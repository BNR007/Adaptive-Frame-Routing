"""UniversalUpscaler application entry point."""

from __future__ import annotations

import argparse
import logging
import math
import signal
import sys
import threading
import time
from pathlib import Path

import cv2

from ai_processor import (
    DEFAULT_AI_TILE_OVERLAP,
    DEFAULT_AI_TILE_SIZE,
    AIProcessor,
    AIProcessorError,
)
from async_pipeline import AsyncFramePipeline, PipelineWorkerError
from capture_manager import CaptureManager
from config import (
    FSR1_LIKE_DEFAULT_EDGE_STRENGTH,
    FSR1_LIKE_DEFAULT_SHARPENING_ENABLED,
    FSR1_LIKE_DEFAULT_SHARPENING_STRENGTH,
    SUPPORTED_CAPTURE_BACKENDS,
    SUPPORTED_UPSCALING_METHODS,
    ApplicationConfig,
    runtime_profile_path,
    upscaling_method_display_name,
)
from frame_processor import FrameProcessor, refine_output
from frame_interpolator import (
    NoOpInterpolator,
    RIFEInterpolator,
    RIFEInterpolatorError,
)
from frame_pacing import (
    DEFAULT_MAX_FRAME_LATENCY_MS,
    FramePacer,
    PresentationFrame,
)
from d3d11_gpu_pipeline import (
    D3D11CapabilityError,
    D3D11GpuError,
    D3D11GpuPipeline,
    validate_gpu_pipeline_request,
)
from hardware_tuner import HardwareTuner
from metrics import PerformanceMetrics, TelemetrySnapshot
from processing_backend import (
    ProcessingBackend,
    ProcessingBackendTuner,
    available_processing_backend_display_names,
    create_processing_backend,
    processing_backend_display_name,
)
from runtime_profile import DEFAULT_RUNTIME_PROFILE, RuntimeProfile
from runtime_recovery import (
    RecoveringCapture,
    RecoveringInterpolator,
    RecoveringProcessor,
    RecoveryController,
    RetryPolicy,
)
from resource_paths import resource_path
from regional_transform import RegionalTransformController
from wgc_capture import WGCError
from window_capture import (
    WindowCaptureError,
    WindowRegionTracker,
    list_visible_windows,
    select_window,
)

logger = logging.getLogger(__name__)

SUPPORTED_PROCESSORS = ("shader", "ai")
DEFAULT_RIFE_MODEL_PATH = (
    resource_path("models", "RIFE_v3.6.onnx")
)


def _raise_keyboard_interrupt(signum, frame) -> None:
    """Route Windows console break events through the normal cleanup path."""

    del signum, frame
    raise KeyboardInterrupt


def _install_console_signal_handlers() -> None:
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _raise_keyboard_interrupt)


def _strength_argument(value: str) -> float:
    try:
        parsed_value = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("strength must be numeric") from error

    if not 0.0 <= parsed_value <= 1.0:
        raise argparse.ArgumentTypeError("strength must be between 0.0 and 1.0")

    return parsed_value


def _ai_tile_argument(value: str) -> str:
    normalized_value = value.strip().lower()
    if normalized_value in ("auto", "off"):
        return normalized_value
    try:
        tile_size = int(normalized_value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "AI tile must be auto, off, or a positive integer."
        ) from error
    if tile_size <= 0:
        raise argparse.ArgumentTypeError("AI tile size must be greater than zero.")
    return str(tile_size)


def _positive_float_argument(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be numeric") from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be a positive finite number")
    return parsed


def _positive_int_argument(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _is_provider_failure(error: BaseException) -> bool:
    message = str(error).lower()
    return any(
        token in message
        for token in ("directml", "dmlexecutionprovider", "execution provider", "device")
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="UniversalUpscaler real-time capture and upscaling",
    )
    parser.add_argument(
        "--retune",
        action="store_true",
        help="Retune processing and OpenCV threads while preserving saved upscaler settings.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Run without cv2.imshow for controlled headless execution.",
    )
    parser.add_argument(
        "--show-performance",
        action="store_true",
        help="Show moving performance telemetry and log a five-second summary.",
    )
    parser.add_argument(
        "--queue-depth",
        type=int,
        default=2,
        help="Bounded newest-frame capture queue depth.",
    )
    parser.add_argument(
        "--target-fps",
        type=_positive_float_argument,
        default=None,
        help="Positive presentation target; auto pacing otherwise measures the source.",
    )
    parser.add_argument(
        "--frame-pacing",
        choices=("auto", "off", "fixed"),
        default="auto",
        help="Measure source cadence, preserve uncapped output, or use a fixed target.",
    )
    parser.add_argument(
        "--max-frame-latency-ms",
        type=_positive_float_argument,
        default=DEFAULT_MAX_FRAME_LATENCY_MS,
        help="Discard presentation output older than this positive latency budget.",
    )
    parser.add_argument(
        "--allow-provider-fallback",
        action="store_true",
        help="Allow repeated DirectML failures to degrade explicitly to CPU.",
    )
    parser.add_argument(
        "--frame-generation",
        choices=("off", "noop", "rife"),
        default="off",
        help=(
            "Disable frame generation, retain the legacy NoOp hook, or generate "
            "one RIFE midpoint between processed frames."
        ),
    )
    parser.add_argument("--rife-model", type=Path, default=None, help="Optional RIFE ONNX model path.")
    parser.add_argument(
        "--generated-frames",
        type=int,
        choices=(1, 2, 3, 4),
        default=1,
        help="Generate this many RIFE frames between consecutive real frames.",
    )
    parser.add_argument(
        "--temporal-stabilization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the shipped whole-frame temporal stabilization heuristic.",
    )
    parser.add_argument(
        "--regional-transform",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use experimental per-region routing, confidence composition, "
            "deadline fallback, and provenance instead of the shipped heuristic."
        ),
    )
    parser.add_argument(
        "--regional-budget-ms",
        type=_positive_float_argument,
        default=None,
        help="Optional regional work budget in milliseconds per generated frame.",
    )
    parser.add_argument(
        "--regional-debug-overlay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Colour generated-frame regions by their recorded provenance.",
    )
    parser.add_argument(
        "--rife-input-width",
        type=_positive_int_argument,
        default=None,
        help="Optional maximum internal RIFE inference width.",
    )
    parser.add_argument(
        "--rife-input-height",
        type=_positive_int_argument,
        default=None,
        help="Optional maximum internal RIFE inference height.",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=0.0,
        help="Warm up processing for this many seconds before session counters begin.",
    )
    parser.add_argument(
        "--preview-fullscreen",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Open the processed preview in borderless fullscreen mode.",
    )
    parser.add_argument(
        "--pipeline",
        choices=("cpu", "d3d11"),
        default="cpu",
        help=(
            "Select the backward-compatible NumPy/OpenCV path or the explicit "
            "WGC/D3D11 GPU-resident scaling and presentation path."
        ),
    )
    parser.add_argument(
        "--processor",
        choices=SUPPORTED_PROCESSORS,
        default="shader",
        help="Process frames with the existing shader/spatial path or one ONNX AI model.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to the single ONNX model required by --processor ai.",
    )
    parser.add_argument(
        "--ai-input-layout",
        choices=("nchw", "nhwc"),
        default="nchw",
        help="Explicit input tensor layout for the ONNX image adapter.",
    )
    parser.add_argument(
        "--ai-output-layout",
        choices=("nchw", "nhwc"),
        default="nchw",
        help="Explicit output tensor layout for the ONNX image adapter.",
    )
    parser.add_argument(
        "--ai-color-order",
        choices=("rgb", "bgr"),
        default="rgb",
        help="Whether the ONNX model interprets its image channels as RGB or BGR.",
    )
    parser.add_argument(
        "--ai-provider",
        choices=("cpu", "directml"),
        default="cpu",
        help=(
            "Select the ONNX Runtime provider for AI processing and live RIFE; "
            "DirectML never falls back silently."
        ),
    )
    parser.add_argument(
        "--ai-device-id",
        type=int,
        default=0,
        help="Non-negative DirectML device ID (ignored by the CPU provider).",
    )
    parser.add_argument(
        "--rife-device-id",
        type=int,
        default=None,
        help=(
            "Optional DirectML device used only for frame generation. "
            "Defaults to --ai-device-id."
        ),
    )
    parser.add_argument(
        "--ai-scale",
        choices=("auto", "1", "2", "3", "4"),
        default="auto",
        help="Automatically detect or explicitly validate the ONNX output scale.",
    )
    parser.add_argument(
        "--ai-input-width",
        type=int,
        default=None,
        help="Resize live captured frames to this width before CPU-side ONNX preprocessing.",
    )
    parser.add_argument(
        "--ai-input-height",
        type=int,
        default=None,
        help="Resize live captured frames to this height before CPU-side ONNX preprocessing.",
    )
    parser.add_argument(
        "--ai-tile",
        type=_ai_tile_argument,
        default="auto",
        metavar="auto|off|SIZE",
        help="Use automatic tiling, disable tiling, or use a fixed square tile size.",
    )
    parser.add_argument(
        "--ai-tile-size",
        type=int,
        default=None,
        help=(
            f"Set a fixed tile size (conservative reference: {DEFAULT_AI_TILE_SIZE}); "
            "equivalent to --ai-tile SIZE."
        ),
    )
    parser.add_argument(
        "--ai-tile-overlap",
        type=int,
        default=DEFAULT_AI_TILE_OVERLAP,
        help="Input-pixel overlap used for feathered tile blending.",
    )
    window_group = parser.add_mutually_exclusive_group()
    window_group.add_argument("--window-title", default=None)
    window_group.add_argument("--window-hwnd", type=int, default=None)
    parser.add_argument("--list-windows", action="store_true")
    parser.add_argument(
        "--capture-backend",
        choices=SUPPORTED_CAPTURE_BACKENDS,
        default=None,
        help="Select and persist the capture backend (WGC requires a selected window).",
    )
    parser.add_argument(
        "--upscaler-method",
        choices=SUPPORTED_UPSCALING_METHODS,
        default=None,
        help="Select and persist the spatial upscaling method.",
    )
    parser.add_argument(
        "--fsr1-like-sharpening",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable the FSR1-like sharpening stage and persist the choice.",
    )
    parser.add_argument(
        "--fsr1-like-sharpening-strength",
        type=_strength_argument,
        default=None,
        metavar="0..1",
        help="Set and persist the conservative FSR1-like sharpening strength.",
    )
    parser.add_argument(
        "--fsr1-like-edge-strength",
        type=_strength_argument,
        default=None,
        metavar="0..1",
        help="Set and persist the FSR1-like edge-adaptive blend strength.",
    )

    args = parser.parse_args()
    if (args.ai_input_width is None) != (args.ai_input_height is None):
        parser.error("--ai-input-width and --ai-input-height must be supplied together.")
    if (args.rife_input_width is None) != (args.rife_input_height is None):
        parser.error(
            "--rife-input-width and --rife-input-height must be supplied together."
        )
    if args.ai_input_width is not None and args.ai_input_width <= 0:
        parser.error("--ai-input-width must be greater than zero.")
    if args.ai_input_height is not None and args.ai_input_height <= 0:
        parser.error("--ai-input-height must be greater than zero.")
    if args.ai_device_id < 0:
        parser.error("--ai-device-id must be zero or greater.")
    if args.rife_device_id is not None and args.rife_device_id < 0:
        parser.error("--rife-device-id must be zero or greater.")
    if args.ai_tile_size is not None and args.ai_tile_size <= 0:
        parser.error("--ai-tile-size must be greater than zero.")
    if args.ai_tile_overlap < 0:
        parser.error("--ai-tile-overlap must be zero or greater.")
    if args.ai_tile_size is not None and args.ai_tile == "off":
        parser.error("--ai-tile-size cannot be supplied with --ai-tile off.")
    fixed_tile_size = args.ai_tile_size
    if args.ai_tile not in ("auto", "off"):
        parsed_tile_size = int(args.ai_tile)
        if fixed_tile_size is not None and fixed_tile_size != parsed_tile_size:
            parser.error("Conflicting fixed AI tile sizes were supplied.")
        fixed_tile_size = parsed_tile_size
    if fixed_tile_size is not None and args.ai_tile_overlap * 2 >= fixed_tile_size:
        parser.error("AI tile size must be greater than twice --ai-tile-overlap.")
    if args.queue_depth <= 0:
        parser.error("--queue-depth must be greater than zero.")
    if not math.isfinite(args.warmup_seconds) or args.warmup_seconds < 0.0:
        parser.error("--warmup-seconds must be zero or a positive finite number.")
    if args.frame_pacing == "fixed" and args.target_fps is None:
        parser.error("--frame-pacing fixed requires --target-fps.")
    return args


def _configure_logging() -> None:
    """Set a clean console logging format for application startup and runtime."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _print_windows() -> None:
    """Print selectable visible windows without initializing capture."""

    windows = list_visible_windows()
    if not windows:
        print("No selectable visible windows found.")
        return

    for index, window in enumerate(windows, start=1):
        line = f"{index}. HWND {window.hwnd} | {window.title}"
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(line.encode(encoding, errors="replace").decode(encoding))


def _load_runtime_profile(profile_file: Path, force_retune: bool) -> RuntimeProfile:
    if not profile_file.exists():
        return DEFAULT_RUNTIME_PROFILE

    # Retuning changes measured runtime choices, not the user's spatial settings.
    del force_retune
    return RuntimeProfile.load(profile_file)


def _updated_runtime_profile(
    runtime_profile: RuntimeProfile,
    **updates,
) -> RuntimeProfile:
    values = {
        "capture_backend": runtime_profile.capture_backend,
        "opencv_threads": runtime_profile.opencv_threads,
        "upscaling_method": runtime_profile.upscaling_method,
        "processing_backend": runtime_profile.processing_backend,
        "fsr1_like_sharpening_enabled": getattr(
            runtime_profile,
            "fsr1_like_sharpening_enabled",
            FSR1_LIKE_DEFAULT_SHARPENING_ENABLED,
        ),
        "fsr1_like_sharpening_strength": getattr(
            runtime_profile,
            "fsr1_like_sharpening_strength",
            FSR1_LIKE_DEFAULT_SHARPENING_STRENGTH,
        ),
        "fsr1_like_edge_strength": getattr(
            runtime_profile,
            "fsr1_like_edge_strength",
            FSR1_LIKE_DEFAULT_EDGE_STRENGTH,
        ),
    }
    values.update(updates)
    return RuntimeProfile(**values)


def _apply_cli_profile_overrides(
    runtime_profile: RuntimeProfile,
    args: argparse.Namespace,
) -> tuple[RuntimeProfile, bool]:
    updates = {}
    argument_to_profile_field = {
        "capture_backend": "capture_backend",
        "upscaler_method": "upscaling_method",
        "fsr1_like_sharpening": "fsr1_like_sharpening_enabled",
        "fsr1_like_sharpening_strength": "fsr1_like_sharpening_strength",
        "fsr1_like_edge_strength": "fsr1_like_edge_strength",
    }

    for argument_name, profile_field in argument_to_profile_field.items():
        argument_value = getattr(args, argument_name, None)

        if argument_value is not None:
            updates[profile_field] = argument_value

    if not updates:
        return runtime_profile, False

    updated_profile = _updated_runtime_profile(runtime_profile, **updates)
    return updated_profile, updated_profile != runtime_profile


def _select_processing_backend(
    runtime_profile: RuntimeProfile,
    test_frame,
    app_config: ApplicationConfig,
    force_retune: bool,
) -> tuple[ProcessingBackend, bool]:
    """Resolve the active processing backend and report whether the profile changed."""

    available_backends = available_processing_backend_display_names()
    logger.info(
        "Available processing backends: %s",
        ", ".join(available_backends) if available_backends else "none",
    )

    if force_retune or runtime_profile.processing_backend == "auto":
        backend = ProcessingBackendTuner().tune(
            test_frame=test_frame,
            output_width=app_config.output_width,
            output_height=app_config.output_height,
            upscaling_method=runtime_profile.upscaling_method,
            fsr1_like_sharpening_enabled=getattr(
                runtime_profile,
                "fsr1_like_sharpening_enabled",
                FSR1_LIKE_DEFAULT_SHARPENING_ENABLED,
            ),
            fsr1_like_sharpening_strength=getattr(
                runtime_profile,
                "fsr1_like_sharpening_strength",
                FSR1_LIKE_DEFAULT_SHARPENING_STRENGTH,
            ),
            fsr1_like_edge_strength=getattr(
                runtime_profile,
                "fsr1_like_edge_strength",
                FSR1_LIKE_DEFAULT_EDGE_STRENGTH,
            ),
        )
        return backend, True

    try:
        backend = create_processing_backend(
            backend_name=runtime_profile.processing_backend,
            output_width=app_config.output_width,
            output_height=app_config.output_height,
            upscaling_method=runtime_profile.upscaling_method,
            fsr1_like_sharpening_enabled=getattr(
                runtime_profile,
                "fsr1_like_sharpening_enabled",
                FSR1_LIKE_DEFAULT_SHARPENING_ENABLED,
            ),
            fsr1_like_sharpening_strength=getattr(
                runtime_profile,
                "fsr1_like_sharpening_strength",
                FSR1_LIKE_DEFAULT_SHARPENING_STRENGTH,
            ),
            fsr1_like_edge_strength=getattr(
                runtime_profile,
                "fsr1_like_edge_strength",
                FSR1_LIKE_DEFAULT_EDGE_STRENGTH,
            ),
        )
    except (RuntimeError, ValueError) as error:
        logger.warning(
            "Saved processing backend '%s' is unavailable: %s",
            runtime_profile.processing_backend,
            error,
        )
        backend = ProcessingBackendTuner().tune(
            test_frame=test_frame,
            output_width=app_config.output_width,
            output_height=app_config.output_height,
            upscaling_method=runtime_profile.upscaling_method,
            fsr1_like_sharpening_enabled=getattr(
                runtime_profile,
                "fsr1_like_sharpening_enabled",
                FSR1_LIKE_DEFAULT_SHARPENING_ENABLED,
            ),
            fsr1_like_sharpening_strength=getattr(
                runtime_profile,
                "fsr1_like_sharpening_strength",
                FSR1_LIKE_DEFAULT_SHARPENING_STRENGTH,
            ),
            fsr1_like_edge_strength=getattr(
                runtime_profile,
                "fsr1_like_edge_strength",
                FSR1_LIKE_DEFAULT_EDGE_STRENGTH,
            ),
        )
        return backend, True

    if not backend.is_available():
        logger.warning(
            "Saved processing backend '%s' is no longer available.",
            runtime_profile.processing_backend,
        )
        backend = ProcessingBackendTuner().tune(
            test_frame=test_frame,
            output_width=app_config.output_width,
            output_height=app_config.output_height,
            upscaling_method=runtime_profile.upscaling_method,
        )
        return backend, True

    logger.info(
        "Loaded processing backend from runtime profile: %s",
        backend.display_name,
    )
    return backend, False


def _create_runtime_processor(
    processor_name: str,
    model_path: Path | None,
    processing_backend: ProcessingBackend | None,
    app_config: ApplicationConfig,
    upscaling_method: str,
    ai_input_layout: str = "nchw",
    ai_output_layout: str = "nchw",
    ai_color_order: str = "rgb",
    ai_provider: str = "cpu",
    ai_device_id: int = 0,
    ai_scale: str | int = "auto",
    ai_input_width: int | None = None,
    ai_input_height: int | None = None,
    ai_tile: str | int = "auto",
    ai_tile_size: int | None = None,
    ai_tile_overlap: int = DEFAULT_AI_TILE_OVERLAP,
):
    """Create the selected frame processor without coupling it to rendering."""

    if processor_name == "shader":
        return FrameProcessor(
            processing_backend=processing_backend,
            output_width=app_config.output_width,
            output_height=app_config.output_height,
            method=upscaling_method,
        )

    if processor_name == "ai":
        processor = AIProcessor(
            model_path=model_path,
            input_layout=ai_input_layout,
            output_layout=ai_output_layout,
            color_order=ai_color_order,
            provider=ai_provider,
            device_id=ai_device_id,
            scale=ai_scale,
            input_width=ai_input_width,
            input_height=ai_input_height,
            tile=ai_tile,
            tile_size=ai_tile_size,
            tile_overlap=ai_tile_overlap,
        )
        processor.initialize()
        return processor

    raise ValueError(f"Unsupported processor: {processor_name}")


def _create_frame_interpolator(
    mode: str,
    *,
    provider: str,
    device_id: int,
    generated_frames: int = 1,
    inference_width: int | None = None,
    inference_height: int | None = None,
    model_path: Path | None = None,
    temporal_stabilization: bool = True,
    regional_transform: bool = False,
    regional_budget_ms: float | None = None,
):
    """Create and initialize the selected optional interpolation hook."""

    if mode == "off":
        return None
    if mode == "noop":
        interpolator = NoOpInterpolator()
    elif mode == "rife":
        # Two evenly-spaced insertions use the CPU regional/optical-flow path
        # at 1/3 and 2/3. The checked-in two-input RIFE exports cannot accept a
        # timestep, so requiring DirectML here is both misleading and can stop
        # a valid 2x run when DirectML is unavailable or silently falls back.
        effective_provider = "cpu" if generated_frames == 2 else provider
        inference_options = {}
        if inference_width is not None or inference_height is not None:
            inference_options = {
                "inference_width": inference_width,
                "inference_height": inference_height,
            }
        interpolator = RIFEInterpolator(
            model_path or DEFAULT_RIFE_MODEL_PATH,
            provider=effective_provider,
            device_id=device_id,
            temporal_stabilization=temporal_stabilization,
            regional_transform=regional_transform,
            regional_budget_ms=regional_budget_ms,
            **inference_options,
        )
    else:
        raise ValueError(f"Unsupported frame-generation mode: {mode}")
    interpolator.initialize()
    return interpolator


def _format_dimensions(dimensions: tuple[int, int] | None) -> str:
    if dimensions is None:
        return "n/a"
    return f"{dimensions[0]}x{dimensions[1]}"


def _draw_preview_overlay(
    frame,
    *,
    capture_backend: str,
    opencv_threads: int,
    capture_dimensions: tuple[int, int] | None,
    window_title: str | None,
    processor_name: str,
    upscaler_name: str,
    telemetry: TelemetrySnapshot | None,
) -> None:
    """Draw the current output presentation rate."""

    presentation_fps = 0.0
    if telemetry is not None:
        presentation_fps = telemetry.presentation_fps or telemetry.fps
    cv2.putText(
        frame,
        f"{presentation_fps:.1f} FPS",
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    run_application(args)


def run_application(
    args: argparse.Namespace,
    *,
    shutdown_event: threading.Event | None = None,
    state_callback=None,
    telemetry_callback=None,
) -> None:
    """Run the shared CLI/GUI engine lifecycle without spawning a subprocess."""

    def publish_state(state: str) -> None:
        if state_callback is not None:
            state_callback(state)

    _configure_logging()
    if threading.current_thread() is threading.main_thread():
        _install_console_signal_handlers()

    if getattr(args, "list_windows", False):
        _print_windows()
        return

    shutdown_event = shutdown_event or threading.Event()
    publish_state("starting")
    recovery = RecoveryController(
        policy=RetryPolicy(),
        shutdown_event=shutdown_event,
        state_callback=lambda state: publish_state(state.value),
    )
    pacer = FramePacer(
        mode=getattr(args, "frame_pacing", "auto"),
        target_fps=getattr(args, "target_fps", None),
        max_frame_latency_ms=getattr(
            args,
            "max_frame_latency_ms",
            DEFAULT_MAX_FRAME_LATENCY_MS,
        ),
        shutdown_event=shutdown_event,
        keep_latest_real=getattr(args, "keep_latest_real", False),
    )

    app_config = ApplicationConfig()
    persist_runtime_profile = bool(getattr(args, "persist_runtime_profile", True))
    profile_file = runtime_profile_path()
    profile_existed_before_launch = profile_file.exists()
    runtime_profile = _load_runtime_profile(
        profile_file=profile_file,
        force_retune=args.retune,
    )
    runtime_profile, cli_profile_changed = _apply_cli_profile_overrides(
        runtime_profile,
        args,
    )

    logger.info("=" * 50)
    logger.info("UniversalUpscaler")
    logger.info("Version: 0.1.0")
    logger.info("Initializing...")
    logger.info(
        "Requested capture backend mode: %s",
        runtime_profile.capture_backend.title(),
    )
    logger.info(
        "Requested upscaling method: %s",
        upscaling_method_display_name(runtime_profile.upscaling_method),
    )
    logger.info(
        "Requested processing backend: %s",
        processing_backend_display_name(runtime_profile.processing_backend),
    )
    processor_name = getattr(args, "processor", "shader")
    model_path = getattr(args, "model", None)
    logger.info("Requested frame processor: %s", processor_name.upper())
    logger.info("Requested runtime pipeline: %s", getattr(args, "pipeline", "cpu").upper())
    logger.info("Preview enabled: %s", not args.no_preview)
    logger.info("=" * 50)

    window_tracker = None
    selected_window = None
    capture_region = app_config.capture_region
    window_title = None

    if getattr(args, "window_title", None) or getattr(args, "window_hwnd", None) is not None:
        try:
            selected_window = select_window(
                title=getattr(args, "window_title", None),
                hwnd=getattr(args, "window_hwnd", None),
            )
            window_tracker = WindowRegionTracker(selected_window)
            capture_region = window_tracker.refresh()
            window_title = selected_window.title
            logger.info("Selected window: %s (HWND %s), client area %sx%s at %s,%s", window_title, selected_window.hwnd, capture_region.width, capture_region.height, capture_region.left, capture_region.top)
        except WindowCaptureError as error:
            logger.error("Unable to select capture window: %s", error)
            return
    else:
        if capture_region is not None:
            logger.info("Capture source: fixed region %sx%s at %s,%s", capture_region.width, capture_region.height, capture_region.left, capture_region.top)

    if processor_name == "ai" and model_path is None:
        logger.error("AI processing requires an ONNX model; use --model PATH.")
        return

    if getattr(args, "pipeline", "cpu") == "d3d11":
        if getattr(args, "frame_generation", "off") != "off":
            logger.error(
                "Live frame generation uses the CPU-frame async runtime; "
                "the D3D11 shader pipeline remains unchanged."
            )
            return
        if processor_name == "ai":
            logger.error(
                "AI processing uses the CPU frame pipeline; --pipeline d3d11 remains the unchanged shader pipeline."
            )
            return

        try:
            validate_gpu_pipeline_request(
                capture_backend=runtime_profile.capture_backend,
                method=runtime_profile.upscaling_method,
                output_width=app_config.output_width,
                output_height=app_config.output_height,
                selected_window=selected_window is not None,
                no_preview=args.no_preview,
            )
        except D3D11CapabilityError as error:
            logger.error("Unable to start explicit D3D11 GPU pipeline: %s", error)
            return

        if runtime_profile.capture_backend == "auto":
            logger.info("D3D11 GPU pipeline resolves capture mode AUTO explicitly to WGC.")
        if args.retune:
            logger.info("--retune applies only to the CPU processing path and is ignored here.")
        if cli_profile_changed and persist_runtime_profile:
            runtime_profile.save(profile_file)
            logger.info("Saved runtime profile overrides to %s", profile_file)

        def run_gpu_pipeline():
            gpu_pipeline = D3D11GpuPipeline(
                hwnd=selected_window.hwnd,
                output_width=app_config.output_width,
                output_height=app_config.output_height,
                method=runtime_profile.upscaling_method,
                fsr1_like_edge_strength=runtime_profile.fsr1_like_edge_strength,
                fsr1_like_sharpening_strength=runtime_profile.fsr1_like_sharpening_strength,
                fsr1_like_sharpening_enabled=runtime_profile.fsr1_like_sharpening_enabled,
                frame_pacer=pacer,
            )
            try:
                return gpu_pipeline.run()
            finally:
                gpu_pipeline.close()

        try:
            try:
                report = run_gpu_pipeline()
            except (D3D11GpuError, WGCError, OSError) as error:
                report = recovery.recover(
                    "renderer",
                    run_gpu_pipeline,
                    initial_error=error,
                )
            logger.info(
                "D3D11 session ended: %s frames in %.2f seconds (%.2f FPS).",
                report.presented_frames,
                report.duration_seconds,
                report.presented_fps,
            )
        except KeyboardInterrupt:
            logger.info("Ctrl+C received; stopping D3D11 GPU pipeline.")
        except (D3D11GpuError, WGCError, OSError, RuntimeError) as error:
            logger.error("D3D11 GPU pipeline failed: %s", error)
        finally:
            pacer.stop()
            recovery.stop()
            recovery.mark_stopped()
        return

    def create_capture_manager():
        return CaptureManager(
            backend=runtime_profile.capture_backend,
            capture_region=capture_region,
            window_hwnd=selected_window.hwnd if selected_window is not None else None,
            fallback_on_explicit_failure=(
                runtime_profile.capture_backend == "wgc"
                and getattr(args, "capture_backend", None) is None
            ),
        )

    try:
        capture = create_capture_manager()
    except (RuntimeError, ValueError) as error:
        logger.error("Unable to initialize capture backend: %s", error)
        return

    logger.info("Active capture backend: %s", capture.backend_name.upper())

    def prepare_capture(component) -> None:
        if window_tracker is None or component.backend_name == "wgc":
            return
        updated_region = window_tracker.refresh()
        if updated_region is not None:
            component.set_capture_region(updated_region)

    capture = RecoveringCapture(
        capture,
        create_capture_manager,
        recovery,
        pre_grab=prepare_capture,
        retryable=lambda error: not isinstance(error, (ValueError, TypeError)),
    )

    try:
        test_frame = capture.grab_frame()
    except (RuntimeError, WindowCaptureError) as error:
        logger.error("Unable to acquire an initial capture frame: %s", error)
        capture.close()
        recovery.stop()
        recovery.mark_stopped()
        return

    processing_backend = None
    if processor_name == "shader":
        processing_backend, processing_profile_changed = _select_processing_backend(
            runtime_profile=runtime_profile,
            test_frame=test_frame,
            app_config=app_config,
            force_retune=args.retune,
        )

        if processing_profile_changed or cli_profile_changed:
            runtime_profile = _updated_runtime_profile(
                runtime_profile,
                processing_backend=processing_backend.backend_name,
            )
            if persist_runtime_profile:
                runtime_profile.save(profile_file)
                logger.info("Saved runtime profile to %s", profile_file)

        logger.info("Active processing backend: %s", processing_backend.display_name)

        if args.retune or not profile_existed_before_launch:
            tuner = HardwareTuner(app_config.opencv_thread_candidates)
            opencv_threads = tuner.find_best_opencv_threads(
                test_frame=test_frame,
                output_width=app_config.output_width,
                output_height=app_config.output_height,
            )
            runtime_profile = _updated_runtime_profile(
                runtime_profile,
                opencv_threads=opencv_threads,
                processing_backend=processing_backend.backend_name,
            )
            if persist_runtime_profile:
                runtime_profile.save(profile_file)
                logger.info("Saved runtime profile to %s", profile_file)
        else:
            opencv_threads = runtime_profile.opencv_threads
            cv2.setNumThreads(opencv_threads)
            logger.info("Loaded runtime profile from %s", profile_file)
    else:
        opencv_threads = runtime_profile.opencv_threads
        cv2.setNumThreads(opencv_threads)
        if cli_profile_changed and persist_runtime_profile:
            runtime_profile.save(profile_file)
            logger.info("Saved runtime profile overrides to %s", profile_file)

    logger.info("Active OpenCV threads: %s", opencv_threads)

    if not args.no_preview:
        logger.info("Press Q to quit the preview.")
    logger.info("=" * 50)

    processor_arguments = dict(
        processor_name=processor_name,
        model_path=model_path,
        processing_backend=processing_backend,
        app_config=app_config,
        upscaling_method=runtime_profile.upscaling_method,
        ai_input_layout=getattr(args, "ai_input_layout", "nchw"),
        ai_output_layout=getattr(args, "ai_output_layout", "nchw"),
        ai_color_order=getattr(args, "ai_color_order", "rgb"),
        ai_provider=getattr(args, "ai_provider", "cpu"),
        ai_device_id=getattr(args, "ai_device_id", 0),
        ai_scale=getattr(args, "ai_scale", "auto"),
        ai_input_width=getattr(args, "ai_input_width", None),
        ai_input_height=getattr(args, "ai_input_height", None),
        ai_tile=getattr(args, "ai_tile", "auto"),
        ai_tile_size=getattr(args, "ai_tile_size", None),
        ai_tile_overlap=getattr(
            args,
            "ai_tile_overlap",
            DEFAULT_AI_TILE_OVERLAP,
        ),
    )

    def create_processor(*, provider: str | None = None):
        arguments = dict(processor_arguments)
        if provider is not None:
            arguments["ai_provider"] = provider
        return _create_runtime_processor(**arguments)

    allow_provider_fallback = bool(getattr(args, "allow_provider_fallback", False))
    try:
        try:
            processor = create_processor()
        except AIProcessorError as error:
            if not (
                processor_name == "ai"
                and getattr(args, "ai_provider", "cpu") == "directml"
                and allow_provider_fallback
                and _is_provider_failure(error)
            ):
                raise
            processor = recovery.recover(
                "inference_provider",
                create_processor,
                initial_error=error,
                retryable=_is_provider_failure,
                fallback=lambda: create_processor(provider="cpu"),
            )
    except (AIProcessorError, ValueError, RuntimeError) as error:
        logger.error("Unable to initialize frame processor: %s", error)
        capture.close()
        if processing_backend is not None:
            processing_backend.close()
        cv2.destroyAllWindows()
        return

    processor = RecoveringProcessor(
        processor,
        create_processor,
        recovery,
        category=("inference_provider" if processor_name == "ai" else "processing"),
        fallback_factory=(
            (lambda: create_processor(provider="cpu"))
            if (
                processor_name == "ai"
                and getattr(args, "ai_provider", "cpu") == "directml"
                and allow_provider_fallback
            )
            else None
        ),
        retryable=lambda error: not isinstance(error, (ValueError, TypeError)),
    )

    active_processor_display_name = (
        processing_backend.display_name
        if processing_backend is not None
        else processor.display_name
    )
    logger.info("Active frame processor: %s", active_processor_display_name)
    show_performance = getattr(args, "show_performance", False)
    metrics = PerformanceMetrics(enabled=show_performance, window_size=60)
    def create_interpolator(*, provider: str | None = None):
        return _create_frame_interpolator(
            getattr(args, "frame_generation", "off"),
            provider=provider or getattr(args, "ai_provider", "cpu"),
            device_id=(
                getattr(args, "rife_device_id", None)
                if getattr(args, "rife_device_id", None) is not None
                else getattr(args, "ai_device_id", 0)
            ),
            inference_width=getattr(args, "rife_input_width", None),
            inference_height=getattr(args, "rife_input_height", None),
            model_path=getattr(args, "rife_model", None),
            generated_frames=getattr(args, "generated_frames", 1),
            temporal_stabilization=getattr(args, "temporal_stabilization", True),
            regional_transform=getattr(args, "regional_transform", False),
            regional_budget_ms=getattr(args, "regional_budget_ms", None),
        )

    try:
        try:
            frame_interpolator = create_interpolator()
        except RIFEInterpolatorError as error:
            if not (
                getattr(args, "frame_generation", "off") == "rife"
                and getattr(args, "ai_provider", "cpu") == "directml"
                and allow_provider_fallback
                and _is_provider_failure(error)
            ):
                raise
            frame_interpolator = recovery.recover(
                "interpolation_provider",
                create_interpolator,
                initial_error=error,
                retryable=_is_provider_failure,
                fallback=lambda: create_interpolator(provider="cpu"),
            )
    except (RIFEInterpolatorError, ValueError, RuntimeError) as error:
        logger.error("Unable to initialize frame generation: %s", error)
        capture.close()
        processor_shutdown = getattr(processor, "shutdown", None)
        if processor_shutdown is not None:
            processor_shutdown()
        if processing_backend is not None:
            processing_close = getattr(processing_backend, "close", None)
            if processing_close is not None:
                processing_close()
        cv2.destroyAllWindows()
        return
    if getattr(args, "frame_generation", "off") == "rife":
        frame_interpolator = RecoveringInterpolator(
            frame_interpolator,
            create_interpolator,
            recovery,
            fallback_factory=(
                (lambda: create_interpolator(provider="cpu"))
                if (
                    getattr(args, "ai_provider", "cpu") == "directml"
                    and allow_provider_fallback
                )
                else None
            ),
        )
    if frame_interpolator is not None:
        logger.info("Selected RIFE model: %s", getattr(args, "rife_model", None) or DEFAULT_RIFE_MODEL_PATH)
        if getattr(args, "generated_frames", 1) == 2:
            logger.info(
                "Two-insertion mode: CPU regional/flow generation at 1/3 and 2/3; "
                "the selected neural provider is not used for these insertions."
            )
        elif getattr(args, "ai_provider", "cpu") == "directml":
            logger.info(
                "Requested frame-generation DirectML device: %s",
                getattr(args, "rife_device_id", None)
                if getattr(args, "rife_device_id", None) is not None
                else getattr(args, "ai_device_id", 0),
            )
        logger.info(
            "Active frame generation: %s (%s)",
            getattr(args, "frame_generation", "off").upper(),
            ", ".join(getattr(frame_interpolator, "active_providers", ()))
            or "no execution provider",
        )

    thread_bound_capture = capture.backend_name == "wgc"
    if thread_bound_capture:
        capture.prepare_thread_handoff()

    def capture_next_frame():
        frame = capture.grab_frame()
        if capture.backend_name == "wgc" and selected_window is not None:
            from window_capture import crop_window_client
            frame = crop_window_client(frame, selected_window.hwnd)
        return frame

    pipeline = AsyncFramePipeline(
        processor,
        capture_source=capture_next_frame,
        capture_shutdown=capture.close,
        capture_shutdown_on_worker=thread_bound_capture,
        frame_interpolator=frame_interpolator,
        generated_frames=getattr(args, "generated_frames", 1),
        queue_depth=getattr(args, "queue_depth", 2),
        collect_telemetry=show_performance,
    )
    recovery.set_on_recovery(pipeline.clear_queued_frames)
    recovery.mark_running()
    warmup_seconds = float(getattr(args, "warmup_seconds", 0.0))
    if warmup_seconds > 0.0:
        warmup_ends_at = time.perf_counter() + warmup_seconds
        last_warmup_second = None
        while not shutdown_event.is_set():
            remaining = max(0, math.ceil(warmup_ends_at - time.perf_counter()))
            if remaining <= 0:
                break
            if remaining != last_warmup_second:
                publish_state(f"warming_up_{remaining}")
                last_warmup_second = remaining
            shutdown_event.wait(min(0.1, max(0.0, warmup_ends_at - time.perf_counter())))
        warmup_interpolator = getattr(frame_interpolator, "warmup", None)
        if not shutdown_event.is_set() and callable(warmup_interpolator):
            publish_state("compiling_frame_generation")
            try:
                warmup_interpolator()
            except Exception as error:
                logger.warning(
                    "Frame-generation warmup failed; continuing normally: %s",
                    error,
                )
    if not shutdown_event.is_set():
        pipeline.start()
        publish_state("running")
    counter_baseline_real = 0
    counter_baseline_generated = 0
    counter_baseline_total = 0
    preview_initialized = False
    game_preview = None
    last_refinement_ms = 0.0

    try:
        try:
            while True:
                if shutdown_event.is_set():
                    break
                if game_preview is not None:
                    if game_preview.stop_requested() or not game_preview.update():
                        break
                processed_batch = pipeline.take_presentation_batch(timeout=0.1)
                if not processed_batch:
                    if pipeline.finished:
                        break
                    continue
                pacing_decisions = pacer.iter_pace_batch(
                    PresentationFrame(
                        processed_frame,
                        processed_frame.captured_at,
                        processed_frame.frame_kind,
                    )
                    for processed_frame in processed_batch
                )
                quit_requested = False
                for decision in pacing_decisions:
                    if not decision.present:
                        continue
                    processed_frame = decision.frame.payload
                    frame = processed_frame.image
                    presented_at = decision.actual_at
                    pacing_snapshot = pacer.snapshot()
                    recovery_snapshot = recovery.snapshot()
                    telemetry_snapshot = None
                    if show_performance:
                        metrics.record(
                            presented_at=presented_at,
                            capture_ms=processed_frame.capture_ms,
                            preprocessing_ms=processed_frame.preprocessing_ms,
                            inference_ms=processed_frame.inference_ms,
                            postprocessing_ms=processed_frame.postprocessing_ms,
                            total_frame_ms=(presented_at - processed_frame.captured_at) * 1000.0,
                            dropped_frames=pipeline.dropped_frames,
                            active_provider=processed_frame.active_provider,
                            capture_dimensions=processed_frame.capture_dimensions,
                            ai_input_dimensions=processed_frame.ai_input_dimensions,
                            ai_output_dimensions=processed_frame.ai_output_dimensions,
                            tile_mode=processed_frame.tile_mode,
                            interpolation_ms=processed_frame.interpolation_ms,
                            interpolation_provider=processed_frame.interpolation_provider,
                            frame_generation=processed_frame.frame_generation,
                            dropped_generated_frames=(
                                pipeline.dropped_generated_frames
                                + pacing_snapshot.generated_frames_dropped_late
                            ),
                            scheduled_presentation_timestamp=decision.scheduled_at,
                            actual_presentation_timestamp=presented_at,
                            pacing_error_ms=decision.pacing_error_ms,
                            late_frames=pacing_snapshot.late_frames,
                            generated_frames_dropped_late=pacing_snapshot.generated_frames_dropped_late,
                            real_frames_dropped_late=pacing_snapshot.real_frames_dropped_late,
                            estimated_source_fps=pacing_snapshot.estimated_source_fps,
                            presentation_fps=pacing_snapshot.presentation_fps,
                            runtime_state=recovery_snapshot.state,
                            recovery_retry_attempts=recovery_snapshot.retry_attempts,
                            successful_recoveries=recovery_snapshot.successful_recoveries,
                            failed_recoveries=recovery_snapshot.failed_recoveries,
                            fallback_activations=recovery_snapshot.fallback_activations,
                            presented_real_frames=(
                                pacing_snapshot.presented_real_frames - counter_baseline_real
                            ),
                            presented_generated_frames=(
                                pacing_snapshot.presented_generated_frames
                                - counter_baseline_generated
                            ),
                            presented_frames=(
                                pacing_snapshot.presented_frames - counter_baseline_total
                            ),
                            generated_frames_requested=(
                                getattr(args, "generated_frames", 1)
                                if getattr(args, "frame_generation", "off") == "rife"
                                else 0
                            ),
                        )
                        if metrics.maybe_log(logger, now=presented_at):
                            logger.info("Preview refinement: previous sample %.2f ms", last_refinement_ms)
                            if processed_frame.regional_provenance_counts:
                                logger.info(
                                    "Regional provenance: %s | routes: %s | admission: %s",
                                    processed_frame.regional_provenance_counts,
                                    processed_frame.regional_route_counts,
                                    processed_frame.regional_admission_reason,
                                )
                        telemetry_snapshot = metrics.snapshot()
                        if telemetry_callback is not None and telemetry_snapshot is not None:
                            telemetry_callback(telemetry_snapshot)

                    if args.no_preview:
                        continue

                    refinement_started = time.perf_counter()
                    frame = refine_output(frame, getattr(args, "output_refinement", 0.0))
                    last_refinement_ms = (time.perf_counter() - refinement_started) * 1000.0
                    if (
                        getattr(args, "regional_debug_overlay", False)
                        and processed_frame.regional_provenance is not None
                    ):
                        frame = RegionalTransformController.provenance_overlay(
                            frame, processed_frame.regional_provenance
                        )
                    if getattr(args, "draw_preview_overlay", True):
                        _draw_preview_overlay(
                            frame,
                            capture_backend=capture.backend_name,
                            opencv_threads=opencv_threads,
                            capture_dimensions=(
                                (
                                    capture.capture_region.width,
                                    capture.capture_region.height,
                                )
                                if getattr(capture, "capture_region", None) is not None
                                else None
                            ),
                            window_title=window_title,
                            processor_name=active_processor_display_name,
                            upscaler_name=upscaling_method_display_name(
                                runtime_profile.upscaling_method
                            ),
                            telemetry=telemetry_snapshot,
                        )

                    preview_title = "UniversalUpscaler Preview"
                    if not preview_initialized:
                        cv2.namedWindow(preview_title, cv2.WINDOW_NORMAL)
                        cv2.imshow(preview_title, frame)
                        cv2.waitKey(1)
                        if getattr(args, "preview_fullscreen", False):
                            cv2.setWindowProperty(
                                preview_title,
                                cv2.WND_PROP_FULLSCREEN,
                                cv2.WINDOW_FULLSCREEN,
                            )
                        if (getattr(args, "preview_fullscreen", False)
                                and selected_window is not None and capture.backend_name == "wgc"):
                            from game_preview import GamePreview
                            game_preview = GamePreview(preview_title, selected_window.hwnd)
                            logger.info("Game preview enabled: Alt-Tab hides output; Ctrl+Alt+Q stops.")
                        preview_initialized = True
                    else:
                        cv2.imshow(preview_title, frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        quit_requested = True
                        break
                if quit_requested:
                    break
            if pipeline.capture_interrupted:
                logger.info("Shutdown requested by user.")
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user.")
        except WindowCaptureError as error:
            logger.warning("Stopping window capture safely: %s", error)
        except PipelineWorkerError as error:
            logger.error("Stopping after asynchronous worker failure: %s", error)

    finally:
        publish_state("stopping")
        pacer.stop()
        recovery.stop()
        try:
            pipeline.stop()
        finally:
            capture.close()
            processor.close()
            if frame_interpolator is not None:
                frame_interpolator.shutdown()
            if processing_backend is not None:
                processing_close = getattr(processing_backend, "close", None)
                if processing_close is not None:
                    processing_close()
            cv2.destroyAllWindows()
            recovery.mark_stopped()
            publish_state("stopped")


if __name__ == "__main__":
    main()
