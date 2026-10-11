"""Regression tests for clean Ctrl+C shutdown."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import sys

import pytest

import main


class _DummyCapture:
    backend_name = "mss"

    def __init__(self) -> None:
        self.closed = False
        self._grab_count = 0

    def grab_frame(self):
        self._grab_count += 1

        if self._grab_count == 1:
            return object()

        raise KeyboardInterrupt

    def close(self) -> None:
        self.closed = True


class _DummyProcessor:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def process(self, frame):
        return frame


class _DummyMetrics:
    def __init__(self, *args, **kwargs) -> None:
        self.pipeline_fps = 0.0
        self.capture_ms = 0.0
        self.upscale_ms = 0.0

    def update(self, *args, **kwargs) -> None:
        pass


class _DummyTuner:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def find_best_opencv_threads(self, *args, **kwargs):
        return 1


class _DummyProfile:
    capture_backend = "auto"
    opencv_threads = 1
    upscaling_method = "bicubic"
    processing_backend = "opencv_cpu"


class _DummyRuntimeProfile:
    def __init__(self, *args, **kwargs) -> None:
        self.capture_backend = kwargs.get("capture_backend", "auto")
        self.opencv_threads = kwargs.get("opencv_threads", 1)
        self.upscaling_method = kwargs.get("upscaling_method", "bicubic")

    def save(self, *args, **kwargs) -> None:
        pass


class _DummyProcessingBackend:
    backend_name = "opencv_cpu"
    display_name = "OpenCV CPU"

    def is_available(self) -> bool:
        return True

    def process(self, frame):
        return frame


def test_main_logs_clean_shutdown_on_keyboard_interrupt(monkeypatch, caplog) -> None:
    monkeypatch.setattr(main, "parse_args", lambda: SimpleNamespace(retune=False, no_preview=True))
    monkeypatch.setattr(main, "_configure_logging", lambda: None)
    monkeypatch.setattr(main, "ApplicationConfig", lambda: SimpleNamespace(
        capture_region=None,
        output_width=1920,
        output_height=1080,
        opencv_thread_candidates=(1,),
        metrics_update_interval=1.0,
    ))
    monkeypatch.setattr(main, "runtime_profile_path", lambda: Path("__nonexistent_runtime_profile.json"))
    monkeypatch.setattr(main, "_load_runtime_profile", lambda profile_file, force_retune: _DummyProfile())
    monkeypatch.setattr(main, "_select_processing_backend", lambda **kwargs: (_DummyProcessingBackend(), False))
    monkeypatch.setattr(main, "RuntimeProfile", _DummyRuntimeProfile)
    monkeypatch.setattr(main, "CaptureManager", lambda **kwargs: _DummyCapture())
    monkeypatch.setattr(main, "HardwareTuner", _DummyTuner)
    monkeypatch.setattr(main, "FrameProcessor", _DummyProcessor)
    monkeypatch.setattr(main, "PerformanceMetrics", _DummyMetrics)
    monkeypatch.setattr(main.cv2, "setNumThreads", lambda *args, **kwargs: None)
    monkeypatch.setattr(main.cv2, "destroyAllWindows", lambda: None)

    with caplog.at_level("INFO"):
        main.main()

    assert "Shutdown requested by user." in caplog.text


def test_cli_accepts_fsr1_like_upscaler_method(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--upscaler-method", "fsr1_like", "--no-preview"],
    )

    args = main.parse_args()

    assert args.upscaler_method == "fsr1_like"
    assert args.no_preview is True


def test_cli_accepts_paired_positive_ai_input_dimensions(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--ai-input-width",
            "320",
            "--ai-input-height",
            "180",
        ],
    )

    args = main.parse_args()

    assert args.ai_input_width == 320
    assert args.ai_input_height == 180


def test_cli_rejects_only_one_ai_input_dimension(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["main.py", "--ai-input-width", "320"])

    with pytest.raises(SystemExit):
        main.parse_args()


@pytest.mark.parametrize(
    ("option", "value"),
    [("--ai-input-width", "0"), ("--ai-input-height", "-1")],
)
def test_cli_rejects_nonpositive_ai_input_dimensions(
    monkeypatch,
    option: str,
    value: str,
) -> None:
    arguments = [
        "main.py",
        "--ai-input-width",
        "320",
        "--ai-input-height",
        "180",
    ]
    arguments[arguments.index(option) + 1] = value
    monkeypatch.setattr(sys, "argv", arguments)

    with pytest.raises(SystemExit):
        main.parse_args()


def test_cli_accepts_auto_and_fixed_ai_tiling(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--ai-tile", "256", "--ai-tile-overlap", "16"],
    )

    fixed_args = main.parse_args()

    assert fixed_args.ai_tile == "256"
    assert fixed_args.ai_tile_overlap == 16

    monkeypatch.setattr(sys, "argv", ["main.py", "--ai-tile", "auto"])
    auto_args = main.parse_args()
    assert auto_args.ai_tile == "auto"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--ai-tile", "0"],
        ["--ai-tile", "invalid"],
        ["--ai-tile", "32", "--ai-tile-overlap", "16"],
        ["--ai-tile", "off", "--ai-tile-size", "64"],
        ["--ai-tile", "64", "--ai-tile-size", "96"],
    ],
)
def test_cli_rejects_invalid_ai_tiling(monkeypatch, arguments: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["main.py", *arguments])

    with pytest.raises(SystemExit):
        main.parse_args()


def test_cli_accepts_performance_telemetry_and_queue_depth(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--show-performance", "--queue-depth", "3"],
    )

    args = main.parse_args()

    assert args.show_performance is True
    assert args.queue_depth == 3


@pytest.mark.parametrize("queue_depth", ["0", "-1"])
def test_cli_rejects_invalid_queue_depth(monkeypatch, queue_depth: str) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--queue-depth", queue_depth],
    )

    with pytest.raises(SystemExit):
        main.parse_args()


@pytest.mark.parametrize("mode", ["off", "noop", "rife"])
def test_cli_accepts_frame_generation_modes(monkeypatch, mode: str) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--frame-generation", mode],
    )

    args = main.parse_args()

    assert args.frame_generation == mode


def test_cli_defaults_frame_generation_to_off(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["main.py"])

    args = main.parse_args()

    assert args.frame_generation == "off"


@pytest.mark.parametrize("amount", (1, 2, 3, 4))
def test_cli_accepts_generated_frame_amount(monkeypatch, amount: int) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--frame-generation", "rife", "--generated-frames", str(amount)],
    )
    args = main.parse_args()
    assert args.generated_frames == amount


def test_cli_accepts_nonnegative_warmup(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["main.py", "--warmup-seconds", "5"])
    assert main.parse_args().warmup_seconds == pytest.approx(5.0)


def test_cli_accepts_separate_rife_device(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--ai-device-id", "1", "--rife-device-id", "0"],
    )
    args = main.parse_args()
    assert args.ai_device_id == 1
    assert args.rife_device_id == 0


def test_cli_accepts_rife_internal_resolution_and_fullscreen_preview(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--rife-input-width",
            "320",
            "--rife-input-height",
            "180",
            "--preview-fullscreen",
        ],
    )
    args = main.parse_args()
    assert (args.rife_input_width, args.rife_input_height) == (320, 180)
    assert args.preview_fullscreen is True


def test_cli_accepts_frame_pacing_latency_and_provider_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--frame-pacing",
            "fixed",
            "--target-fps",
            "59.94",
            "--max-frame-latency-ms",
            "80",
            "--allow-provider-fallback",
        ],
    )
    args = main.parse_args()
    assert args.frame_pacing == "fixed"
    assert args.target_fps == pytest.approx(59.94)
    assert args.max_frame_latency_ms == 80.0
    assert args.allow_provider_fallback is True


@pytest.mark.parametrize(
    "arguments",
    [
        ["--target-fps", "0"],
        ["--target-fps", "nan"],
        ["--max-frame-latency-ms", "-1"],
        ["--frame-pacing", "fixed"],
    ],
)
def test_cli_rejects_invalid_frame_pacing_values(monkeypatch, arguments) -> None:
    monkeypatch.setattr(sys, "argv", ["main.py", *arguments])
    with pytest.raises(SystemExit):
        main.parse_args()


def test_live_rife_factory_uses_selected_provider_and_shuts_down(
    monkeypatch,
) -> None:
    created = []

    class FakeRIFEInterpolator:
        def __init__(
            self,
            model_path,
            *,
            provider,
            device_id,
            temporal_stabilization,
            regional_transform,
            regional_budget_ms,
        ):
            self.model_path = model_path
            self.provider = provider
            self.device_id = device_id
            self.temporal_stabilization = temporal_stabilization
            self.regional_transform = regional_transform
            self.regional_budget_ms = regional_budget_ms
            self.initialized = False
            self.shutdown_calls = 0
            created.append(self)

        def initialize(self):
            self.initialized = True

        def shutdown(self):
            self.shutdown_calls += 1

    monkeypatch.setattr(main, "RIFEInterpolator", FakeRIFEInterpolator)

    interpolator = main._create_frame_interpolator(
        "rife",
        provider="directml",
        device_id=0,
    )
    interpolator.shutdown()

    assert created == [interpolator]
    assert interpolator.model_path == main.DEFAULT_RIFE_MODEL_PATH
    assert interpolator.provider == "directml"
    assert interpolator.device_id == 0
    assert interpolator.temporal_stabilization is True
    assert interpolator.regional_transform is False
    assert interpolator.regional_budget_ms is None
    assert interpolator.initialized is True
    assert interpolator.shutdown_calls == 1


def test_two_insertions_use_cpu_flow_provider_even_when_directml_is_selected(
    monkeypatch,
) -> None:
    created = []

    class FakeRIFEInterpolator:
        def __init__(self, model_path, **options):
            self.model_path = model_path
            self.options = options
            self.initialized = False
            created.append(self)

        def initialize(self):
            self.initialized = True

    monkeypatch.setattr(main, "RIFEInterpolator", FakeRIFEInterpolator)

    interpolator = main._create_frame_interpolator(
        "rife",
        provider="directml",
        device_id=1,
        generated_frames=2,
        regional_transform=True,
        inference_width=160,
        inference_height=90,
    )

    assert created == [interpolator]
    assert interpolator.options["provider"] == "cpu"
    assert interpolator.options["device_id"] == 1
    assert interpolator.options["regional_transform"] is True
    assert interpolator.initialized is True


def test_preview_overlay_starts_with_zero_fps(monkeypatch) -> None:
    lines = []
    monkeypatch.setattr(
        main.cv2,
        "putText",
        lambda frame, text, *args, **kwargs: lines.append(text),
    )

    main._draw_preview_overlay(
        object(),
        capture_backend="wgc",
        opencv_threads=2,
        capture_dimensions=(640, 360),
        window_title=None,
        processor_name="ONNX Runtime AI",
        upscaler_name="AI",
        telemetry=None,
    )

    assert lines == ["Scheduler 0.0/s"]


def test_preview_overlay_shows_current_presentation_fps(
    monkeypatch,
) -> None:
    lines = []
    monkeypatch.setattr(
        main.cv2,
        "putText",
        lambda frame, text, *args, **kwargs: lines.append(text),
    )
    telemetry = main.TelemetrySnapshot(
        fps=30.0,
        capture_ms=1.0,
        preprocessing_ms=2.0,
        inference_ms=3.0,
        postprocessing_ms=4.0,
        total_frame_ms=10.0,
        median_frame_ms=10.0,
        p95_frame_ms=12.0,
        dropped_frames=5,
        active_provider="DmlExecutionProvider",
        capture_dimensions=(640, 360),
        ai_input_dimensions=(320, 180),
        ai_output_dimensions=(640, 360),
        tile_mode="auto (256px, 2 tiles)",
        sample_count=60,
        presentation_fps=59.94,
        presented_real_frames=30,
        presented_generated_frames=120,
        presented_frames=150,
    )

    main._draw_preview_overlay(
        object(),
        capture_backend="wgc",
        opencv_threads=2,
        capture_dimensions=(640, 360),
        window_title="Game",
        processor_name="ONNX Runtime AI",
        upscaler_name="AI",
        telemetry=telemetry,
    )

    assert lines == ["Scheduler 59.9/s"]


def test_preview_overlay_uses_measured_fps_until_presentation_rate_is_ready(
    monkeypatch,
) -> None:
    lines = []
    monkeypatch.setattr(
        main.cv2,
        "putText",
        lambda frame, text, *args, **kwargs: lines.append(text),
    )
    telemetry = main.TelemetrySnapshot(
        fps=24.0,
        capture_ms=1.0,
        preprocessing_ms=0.0,
        inference_ms=0.0,
        postprocessing_ms=0.0,
        total_frame_ms=42.0,
        median_frame_ms=40.0,
        p95_frame_ms=50.0,
        dropped_frames=3,
        active_provider="OpenCV CPU",
        capture_dimensions=(128, 72),
        ai_input_dimensions=(128, 72),
        ai_output_dimensions=(128, 72),
        tile_mode="off",
        sample_count=10,
        interpolation_ms=30.5,
        interpolation_provider="DmlExecutionProvider",
        frame_generation="rife",
        dropped_generated_frames=2,
        presented_real_frames=8,
        presented_generated_frames=7,
    )

    main._draw_preview_overlay(
        object(),
        capture_backend="wgc",
        opencv_threads=2,
        capture_dimensions=(128, 72),
        window_title=None,
        processor_name="OpenCV CPU",
        upscaler_name="Bilinear",
        telemetry=telemetry,
    )

    assert lines == ["Scheduler 24.0/s"]


def test_preview_overlay_distinguishes_calls_from_unique_output(monkeypatch) -> None:
    lines = []
    monkeypatch.setattr(
        main.cv2,
        "putText",
        lambda frame, text, *args, **kwargs: lines.append(text),
    )

    main._draw_preview_overlay(
        object(),
        capture_backend="wgc",
        opencv_threads=2,
        capture_dimensions=(1280, 720),
        window_title="Game",
        processor_name="OpenCV CPU",
        upscaler_name="Bicubic",
        telemetry=None,
        output_analysis={
            "calls_per_second": 60.0,
            "unique_per_second": 31.0,
            "meaningful_per_second": 29.0,
            "presentation_interval_ms": 16.7,
        },
    )

    assert lines == [
        "Scheduler 0.0/s",
        "Preview 60.0/s  Unique 31.0/s  Motion 29.0/s",
        "Gap 16.7 ms",
    ]
