"""Qt-safe GUI controller and shared-engine adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import logging
from pathlib import Path
import platform
import threading
from types import SimpleNamespace
from typing import Callable

from PySide6.QtCore import QObject, Signal, Slot

import main as engine_main
from resource_paths import resource_path


PROJECT_ROOT = resource_path()
SRVGG_MODEL = resource_path("models", "SRVGGNetCompact_x2.onnx")
RIFE_MODEL = resource_path("models", "RIFE_v3.6.onnx")
RIFE_LITE_MODEL = resource_path("models", "RIFE_v4.25_lite.onnx")
IFRNET_MODEL = resource_path("models", "IFRNet_S_Vimeo90K.onnx")

PRESETS = {
    "performance": {"width": 160, "height": 90, "tile": "off", "latency": 100.0, "rife_latency": 100.0, "queue": 2},
    "fast_quality": {"width": 240, "height": 135, "tile": "off", "latency": 100.0, "rife_latency": 100.0, "queue": 2},
    "balanced": {"width": 320, "height": 180, "tile": "auto", "latency": 100.0, "rife_latency": 100.0, "queue": 2},
    "quality": {"width": 640, "height": 360, "tile": "auto", "latency": 150.0, "rife_latency": 150.0, "queue": 2},
}


@dataclass(frozen=True, slots=True)
class RuntimeConfiguration:
    hwnd: int
    upscaling_mode: str = "shader"
    upscaling_method: str = "fsr1_like"
    frame_generation: str = "off"
    generated_frames: int = 1
    rife_model_path: Path = RIFE_MODEL
    warmup_seconds: float = 5.0
    preset: str = "balanced"
    target_fps: float | None = None
    provider: str = "directml"
    device_id: int = 0
    rife_device_id: int | None = None
    ai_tile: str = "auto"
    ai_tile_overlap: int = 16
    ai_input_width: int = 320
    ai_input_height: int = 180
    frame_pacing: str = "auto"
    max_frame_latency_ms: float = 100.0
    queue_depth: int = 2
    allow_provider_fallback: bool = False
    show_performance_overlay: bool = True
    output_refinement: float = 0.0
    temporal_stabilization: bool = True
    regional_transform: bool = False
    regional_budget_ms: float = 16.0
    regional_debug_overlay: bool = False
    model_path: Path = SRVGG_MODEL
    ai_scale: str = "2"
    ai_input_layout: str = "nchw"
    ai_output_layout: str = "nchw"
    ai_color_order: str = "rgb"
    capture_backend: str = "wgc"
    pipeline: str = "cpu"
    fsr1_like_sharpening: bool = True
    fsr1_like_sharpening_strength: float = 0.2
    fsr1_like_edge_strength: float = 0.35
    research_enabled: bool = False
    research_root: Path = PROJECT_ROOT / "data" / "raw"
    research_machine_id: str = platform.node() or "machine"
    research_application: str = "application"
    research_scene: str = "scene_01"
    research_condition: str = "adaptive_routing"
    research_repetition: int = 1
    research_warmup_seconds: float = 30.0
    research_duration_seconds: float = 60.0

    def validate(self) -> None:
        if not 0.0 <= self.output_refinement <= 0.25:
            raise ValueError("Output refinement must be between 0 and 0.25.")
        if not isinstance(self.temporal_stabilization, bool):
            raise ValueError("Temporal stabilization must be enabled or disabled.")
        if not isinstance(self.regional_transform, bool):
            raise ValueError("Regional transformation must be enabled or disabled.")
        if self.regional_budget_ms <= 0.0:
            raise ValueError("Regional deadline budget must be positive.")
        if self.hwnd <= 0:
            raise ValueError("Select an open target window.")
        if self.upscaling_mode not in ("off", "shader", "ai"):
            raise ValueError("Select a valid upscaling mode.")
        if self.preset not in PRESETS:
            raise ValueError("Select a valid performance preset.")
        if self.frame_pacing not in ("auto", "fixed", "off"):
            raise ValueError("Select a valid frame pacing mode.")
        if self.frame_pacing == "fixed" and self.target_fps is None:
            raise ValueError("Fixed frame pacing requires a target FPS.")
        if self.target_fps is not None and self.target_fps <= 0:
            raise ValueError("Target FPS must be positive.")
        if not 1 <= self.generated_frames <= 4:
            raise ValueError("Generated frames per real frame must be between 1 and 4.")
        if self.warmup_seconds < 0:
            raise ValueError("Warm-up duration cannot be negative.")
        if self.ai_input_width <= 0 or self.ai_input_height <= 0:
            raise ValueError("AI internal dimensions must be positive.")
        if self.ai_tile_overlap < 0 or self.queue_depth <= 0 or self.max_frame_latency_ms <= 0:
            raise ValueError("Advanced numeric values must be positive.")
        if self.provider not in ("cpu", "directml") or self.device_id < 0:
            raise ValueError("Select a valid provider and non-negative device ID.")
        if self.rife_device_id is not None and self.rife_device_id < 0:
            raise ValueError("Select a non-negative frame-generation device ID.")
        if self.capture_backend not in ("auto", "wgc", "dxcam", "mss"):
            raise ValueError("Select a valid capture backend.")
        if self.pipeline not in ("cpu", "d3d11"):
            raise ValueError("Select a valid runtime pipeline.")
        if self.ai_scale not in ("auto", "1", "2", "3", "4"):
            raise ValueError("Select a valid AI output scale.")
        if self.ai_input_layout not in ("nchw", "nhwc") or self.ai_output_layout not in ("nchw", "nhwc"):
            raise ValueError("Select valid AI tensor layouts.")
        if self.ai_color_order not in ("rgb", "bgr"):
            raise ValueError("Select a valid AI color order.")
        for value, name in (
            (self.fsr1_like_sharpening_strength, "FSR sharpening strength"),
            (self.fsr1_like_edge_strength, "FSR edge strength"),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
        if self.research_enabled:
            for value, name in (
                (self.research_machine_id, "machine ID"),
                (self.research_application, "application"),
                (self.research_scene, "scene ID"),
                (self.research_condition, "condition"),
            ):
                if not str(value).strip():
                    raise ValueError(f"Research {name} cannot be empty.")
            if self.research_repetition <= 0:
                raise ValueError("Research repetition must be positive.")
            if self.research_warmup_seconds < 0.0:
                raise ValueError("Research warm-up cannot be negative.")
            if self.research_duration_seconds <= 0.0:
                raise ValueError("Research duration must be positive.")
        if self.upscaling_mode == "ai":
            model_path = Path(self.model_path)
            if not model_path.is_file():
                raise ValueError(f"The selected AI model does not exist: {model_path}")
            if model_path.suffix.lower() != ".onnx":
                raise ValueError("The selected AI model must be an ONNX file.")
        if self.frame_generation == "rife" and not Path(self.rife_model_path).is_file():
            raise ValueError("The selected RIFE model is missing.")
        if self.pipeline == "d3d11":
            if self.upscaling_mode != "shader":
                raise ValueError("The D3D11 pipeline requires Shader upscaling.")
            if self.frame_generation != "off":
                raise ValueError("Frame generation requires the CPU-frame pipeline.")
            if self.capture_backend not in ("auto", "wgc"):
                raise ValueError("The D3D11 pipeline requires Auto or WGC capture.")
            if self.upscaling_method == "bicubic":
                raise ValueError("Bicubic is unavailable in the D3D11 pipeline.")

    def to_engine_args(self) -> SimpleNamespace:
        self.validate()
        method = self.upscaling_method if self.upscaling_mode == "shader" else "bilinear"
        if self.upscaling_mode == "off":
            method = "bilinear"
        rife_preset = PRESETS[self.preset]
        effective_latency = self.max_frame_latency_ms
        return SimpleNamespace(
            retune=False,
            no_preview=False,
            preview_presenter="d3d11" if self.research_enabled else "opencv",
            preview_fullscreen=True,
            show_performance=True,
            draw_preview_overlay=self.show_performance_overlay,
            output_refinement=self.output_refinement,
            temporal_stabilization=self.temporal_stabilization,
            regional_transform=self.regional_transform,
            regional_budget_ms=(
                self.regional_budget_ms if self.regional_transform else None
            ),
            regional_debug_overlay=self.regional_debug_overlay,
            queue_depth=self.queue_depth, frame_generation=self.frame_generation,
            generated_frames=self.generated_frames, warmup_seconds=self.warmup_seconds,
            rife_model=Path(self.rife_model_path),
            rife_device_id=(
                self.device_id if self.rife_device_id is None else self.rife_device_id
            ),
            rife_input_width=(rife_preset["width"] if self.frame_generation == "rife" else None),
            rife_input_height=(rife_preset["height"] if self.frame_generation == "rife" else None),
            pipeline=self.pipeline, processor="ai" if self.upscaling_mode == "ai" else "shader",
            model=Path(self.model_path) if self.upscaling_mode == "ai" else None,
            ai_input_layout=self.ai_input_layout, ai_output_layout=self.ai_output_layout,
            ai_color_order=self.ai_color_order,
            ai_provider=self.provider, ai_device_id=self.device_id,
            ai_scale=self.ai_scale if self.upscaling_mode == "ai" else "auto",
            ai_input_width=self.ai_input_width if self.upscaling_mode == "ai" else None,
            ai_input_height=self.ai_input_height if self.upscaling_mode == "ai" else None,
            ai_tile=self.ai_tile, ai_tile_size=None, ai_tile_overlap=self.ai_tile_overlap,
            window_title=None, window_hwnd=self.hwnd, list_windows=False,
            capture_backend=self.capture_backend, upscaler_method=method,
            fsr1_like_sharpening=self.fsr1_like_sharpening,
            fsr1_like_sharpening_strength=self.fsr1_like_sharpening_strength,
            fsr1_like_edge_strength=self.fsr1_like_edge_strength, target_fps=self.target_fps,
            frame_pacing=self.frame_pacing,
            max_frame_latency_ms=effective_latency,
            allow_provider_fallback=self.allow_provider_fallback,
            persist_runtime_profile=True,
            keep_latest_real=True,
            research_run=self.research_enabled,
            visual_frame_marker=self.research_enabled,
            research_root=Path(self.research_root),
            research_machine_id=self.research_machine_id,
            research_application=self.research_application,
            research_scene=self.research_scene,
            research_condition=self.research_condition,
            research_repetition=self.research_repetition,
            research_warmup_seconds=self.research_warmup_seconds,
            research_duration_seconds=self.research_duration_seconds,
        )


class _SignalLogHandler(logging.Handler):
    def __init__(self, callback: Callable[[str, int], None]) -> None:
        super().__init__(logging.INFO)
        self.callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        self.callback(self.format(record), record.levelno)


class RuntimeWorker(QObject):
    state = Signal(str)
    telemetry = Signal(dict)
    error = Signal(str)
    log = Signal(str)
    finished = Signal()

    def __init__(self, configuration: RuntimeConfiguration, shutdown_event: threading.Event, engine_runner) -> None:
        super().__init__()
        self.configuration = configuration
        self.shutdown_event = shutdown_event
        self.engine_runner = engine_runner
        self._failed = False

    def _on_log(self, message: str, level: int) -> None:
        self.log.emit(message)
        if level >= logging.ERROR:
            self._failed = True
            self.error.emit(message.rsplit("|", 1)[-1].strip())

    @Slot()
    def run(self) -> None:
        handler = _SignalLogHandler(self._on_log)
        logging.getLogger().addHandler(handler)
        try:
            args = self.configuration.to_engine_args()
            self.engine_runner(
                args,
                shutdown_event=self.shutdown_event,
                state_callback=self.state.emit,
                telemetry_callback=self._emit_telemetry,
            )
        except Exception as error:
            self._failed = True
            logging.getLogger(__name__).exception("GUI runtime failed")
            self.error.emit(str(error) or type(error).__name__)
        finally:
            logging.getLogger().removeHandler(handler)
            if self._failed:
                self.state.emit("failed")
            self.finished.emit()

    def _emit_telemetry(self, snapshot) -> None:
        payload = asdict(snapshot) if is_dataclass(snapshot) else dict(snapshot)
        self.telemetry.emit(payload)


class GuiController(QObject):
    state_changed = Signal(str)
    telemetry_changed = Signal(dict)
    error_occurred = Signal(str)
    log_message = Signal(str)
    running_changed = Signal(bool)

    def __init__(self, *, engine_runner=engine_main.run_application, parent=None) -> None:
        super().__init__(parent)
        self.engine_runner = engine_runner
        self._thread: threading.Thread | None = None
        self._worker: RuntimeWorker | None = None
        self._shutdown_event: threading.Event | None = None
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def start(self, configuration: RuntimeConfiguration) -> bool:
        if self._running:
            return False
        configuration.validate()
        self._running = True
        self.running_changed.emit(True)
        self.state_changed.emit("starting")
        self._shutdown_event = threading.Event()
        self._worker = RuntimeWorker(configuration, self._shutdown_event, self.engine_runner)
        self._worker.state.connect(self.state_changed)
        self._worker.telemetry.connect(self.telemetry_changed)
        self._worker.error.connect(self.error_occurred)
        self._worker.log.connect(self.log_message)
        self._worker.finished.connect(self._mark_not_running)
        self._worker.finished.connect(self._cleanup_thread)
        self._thread = threading.Thread(
            target=self._worker.run,
            name="gui-runtime-worker",
            daemon=False,
        )
        self._thread.start()
        return True

    def stop(self) -> bool:
        if not self._running or self._shutdown_event is None:
            return False
        self.state_changed.emit("stopping")
        self._shutdown_event.set()
        return True

    @Slot()
    def _mark_not_running(self) -> None:
        if not self._running:
            return
        self._running = False
        self.running_changed.emit(False)

    @Slot()
    def _cleanup_thread(self) -> None:
        self._worker = None
        self._thread = None
        self._shutdown_event = None

    def shutdown(self, timeout_ms: int = 7000) -> bool:
        if not self._running:
            return True
        thread = self._thread
        self.stop()
        if thread is not None:
            thread.join(timeout_ms / 1000.0)
        stopped = thread is None or not thread.is_alive()
        if stopped:
            self._mark_not_running()
            self._cleanup_thread()
        return stopped
