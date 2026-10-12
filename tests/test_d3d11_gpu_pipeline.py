"""GPU architecture tests that do not require a native graphics session."""

from __future__ import annotations

import ctypes
import inspect
from collections import deque
from types import SimpleNamespace

import pytest

import d3d11_gpu_pipeline
import d3d11_shaders
import wgc_capture
from cpu_d3d11_presenter import D3D11CpuFramePresenter, RefreshCadence
from d3d11_gpu_pipeline import (
    D3D11CapabilityError,
    D3D11PresentationError,
    D3D11ScalingPass,
    D3D11ShaderScaler,
    GpuPipelineMetrics,
    PresenterSizeState,
    SUPPORTED_D3D11_SCALERS,
    validate_gpu_pipeline_request,
)
from wgc_capture import D3D11Frame, WGCCaptureBackend, _NativeFrame


def test_gpu_frame_metadata_and_close_owns_one_texture_reference(monkeypatch) -> None:
    released: list[int] = []

    def fake_release(pointer):
        if pointer is not None and pointer.value:
            released.append(pointer.value)
            pointer.value = None

    monkeypatch.setattr(wgc_capture, "_release", fake_release)
    frame = D3D11Frame(
        ctypes.c_void_p(1234),
        width=1280,
        height=720,
        dxgi_format=87,
        sequence=9,
        captured_at=12.5,
    )

    assert frame.texture_pointer.value == 1234
    assert (frame.width, frame.height) == (1280, 720)
    assert frame.format_name == "DXGI_FORMAT_B8G8R8A8_UNORM"
    assert frame.sequence == 9
    assert frame.captured_at == 12.5

    frame.close()
    frame.close()
    assert frame.closed is True
    assert released == [1234]
    with pytest.raises(wgc_capture.WGCError, match="after close"):
        _ = frame.texture_pointer


@pytest.mark.parametrize(
    ("pointer", "width", "height", "dxgi_format", "message"),
    [
        (0, 1, 1, 87, "non-null"),
        (1, 0, 1, 87, "positive"),
        (1, 1, -1, 87, "positive"),
        (1, 1, 1, 28, "B8G8R8A8"),
    ],
)
def test_gpu_frame_metadata_validation(pointer, width, height, dxgi_format, message) -> None:
    with pytest.raises(ValueError, match=message):
        D3D11Frame(
            pointer,
            width=width,
            height=height,
            dxgi_format=dxgi_format,
            sequence=0,
            captured_at=0.0,
        )


@pytest.mark.parametrize("method", ("nearest", "bilinear", "lanczos", "fsr1_like"))
def test_gpu_scaler_capabilities_accept_registered_methods(method: str) -> None:
    assert method in SUPPORTED_D3D11_SCALERS
    validate_gpu_pipeline_request(
        capture_backend="wgc",
        method=method,
        output_width=1920,
        output_height=1080,
        selected_window=True,
    )


@pytest.mark.parametrize("method", ("bicubic", "ai", "frame_generation"))
def test_gpu_scaler_rejects_unimplemented_methods_without_substitution(method: str) -> None:
    with pytest.raises(D3D11CapabilityError, match=method):
        validate_gpu_pipeline_request(
            capture_backend="wgc",
            method=method,
            output_width=1920,
            output_height=1080,
            selected_window=True,
        )


@pytest.mark.parametrize("backend", ("mss", "dxcam"))
def test_gpu_pipeline_rejects_cpu_capture_backends(backend: str) -> None:
    with pytest.raises(D3D11CapabilityError, match="requires selected-window WGC"):
        validate_gpu_pipeline_request(
            capture_backend=backend,
            method="bilinear",
            output_width=1920,
            output_height=1080,
            selected_window=True,
        )


def test_gpu_pipeline_requires_selected_wgc_window_and_valid_output() -> None:
    with pytest.raises(D3D11CapabilityError, match="selected"):
        validate_gpu_pipeline_request(
            capture_backend="wgc",
            method="bilinear",
            output_width=1920,
            output_height=1080,
            selected_window=False,
        )
    with pytest.raises(D3D11CapabilityError, match="positive"):
        validate_gpu_pipeline_request(
            capture_backend="wgc",
            method="bilinear",
            output_width=0,
            output_height=1080,
            selected_window=True,
        )


def test_gpu_pipeline_accepts_frame_pacer() -> None:
    signature = inspect.signature(d3d11_gpu_pipeline.D3D11GpuPipeline)
    assert "frame_pacer" in signature.parameters


def test_gpu_path_rejects_no_preview_instead_of_skipping_presentation() -> None:
    with pytest.raises(D3D11CapabilityError, match="--no-preview"):
        validate_gpu_pipeline_request(
            capture_backend="wgc",
            method="bilinear",
            output_width=1920,
            output_height=1080,
            selected_window=True,
            no_preview=True,
        )


def test_presenter_resize_and_idempotent_close_state() -> None:
    state = PresenterSizeState(1920, 1080)
    state.request(1600, 900)
    assert state.take_pending() == (1600, 900)
    state.commit(1600, 900)
    assert (state.width, state.height) == (1600, 900)
    assert state.take_pending() is None
    state.close()
    state.close()
    assert state.closed is True
    with pytest.raises(D3D11PresentationError, match="closed"):
        state.request(1280, 720)


@pytest.mark.parametrize(
    ("refresh_hz", "target_fps", "expected"),
    [(60, 60, 1), (75, 60, 1), (120, 60, 2), (144, 60, 2), (240, 60, 4)],
)
def test_cpu_presenter_matches_high_refresh_display_to_target(
    refresh_hz: int, target_fps: float, expected: int
) -> None:
    assert (
        D3D11CpuFramePresenter._sync_interval_for_target(refresh_hz, target_fps)
        == expected
    )


@pytest.mark.parametrize(
    ("refresh_hz", "target_fps", "expected"),
    [
        (60, 60, [1, 1, 1, 1, 1]),
        (120, 60, [2, 2, 2, 2, 2]),
        (144, 60, [2, 2, 3, 2, 3]),
        (165, 60, [2, 3, 3, 3]),
        (240, 60, [4, 4, 4, 4]),
    ],
)
def test_refresh_cadence_maps_target_onto_integer_vblanks(
    refresh_hz: int, target_fps: float, expected: list[int]
) -> None:
    cadence = RefreshCadence(refresh_hz, target_fps)
    assert [cadence.next_interval() for _ in expected] == expected


def test_gpu_metrics_report_cpu_side_distributions() -> None:
    metrics = GpuPipelineMetrics()
    metrics.started_at = 10.0
    for value in (1.0, 2.0, 3.0):
        metrics.record(
            acquisition_ms=value,
            scale_submit_ms=value + 1.0,
            present_submit_ms=value + 2.0,
            cpu_loop_ms=value + 3.0,
            source_size=(1280, 720),
            output_size=(1920, 1080),
        )
    report = metrics.report(
        adapter_description="Test adapter",
        replaced_frames=2,
        ended_at=11.0,
    )
    assert report.presented_frames == 3
    assert report.presented_fps == 3.0
    assert report.acquisition.average_ms == 2.0
    assert report.acquisition.median_ms == 2.0
    assert report.replaced_frames == 2
    assert report.gpu_timestamp_queries is False


class _FakeGpuRuntime:
    def __init__(self, frames=(), size=(4, 3)) -> None:
        self.size = size
        self.frames = deque(frames)
        self.released = []
        self.recreated = []
        self.converted = []
        self.closed = False
        self.device_pointer = ctypes.c_void_p(11)
        self.context_pointer = ctypes.c_void_p(22)

    def try_get_frame(self):
        return self.frames.popleft() if self.frames else None

    def frame_to_gpu(self, frame, *, sequence, captured_at):
        self.converted.append((frame, sequence, captured_at))
        return SimpleNamespace(
            width=frame.width,
            height=frame.height,
            sequence=sequence,
            captured_at=captured_at,
        )

    def frame_to_bgr(self, frame):
        raise AssertionError("GPU acquisition must not invoke CPU readback")

    def release_frame(self, frame):
        self.released.append(frame)

    def recreate(self, width, height):
        self.size = (width, height)
        self.recreated.append((width, height))
        self.frames.append(_NativeFrame(pointer=None, width=width, height=height))

    def close(self):
        self.closed = True


@pytest.fixture
def valid_window(monkeypatch):
    monkeypatch.setattr(wgc_capture.win32gui, "IsWindow", lambda hwnd: True)
    monkeypatch.setattr(wgc_capture.win32gui, "IsIconic", lambda hwnd: False)


def test_wgc_gpu_acquisition_drains_to_newest_and_tracks_replacements(valid_window) -> None:
    old = _NativeFrame(pointer=None, width=4, height=3)
    newest = _NativeFrame(pointer=None, width=4, height=3)
    runtime = _FakeGpuRuntime((old, newest))
    backend = WGCCaptureBackend(7, runtime=runtime)

    frame = backend.grab_gpu_frame()

    assert frame.width == 4 and frame.height == 3
    assert frame.sequence == 1
    assert runtime.converted[0][0] is newest
    assert runtime.released == [old, newest]
    assert backend.gpu_replaced_frames == 1
    assert backend.d3d11_device_pointer.value == 11
    assert backend.d3d11_context_pointer.value == 22


def test_wgc_gpu_resize_recreates_pool_before_returning_texture(valid_window) -> None:
    resized = _NativeFrame(pointer=None, width=6, height=5)
    runtime = _FakeGpuRuntime((resized,))
    backend = WGCCaptureBackend(7, runtime=runtime, poll_interval=0)

    frame = backend.grab_gpu_frame()

    assert runtime.recreated == [(6, 5)]
    assert (frame.width, frame.height) == (6, 5)
    assert len(runtime.released) == 2


def test_normal_gpu_module_has_no_numpy_opencv_or_cpu_grab_boundary() -> None:
    for module in (d3d11_gpu_pipeline, d3d11_shaders):
        source = inspect.getsource(module)
        assert "import numpy" not in source
        assert "import cv2" not in source
        assert ".grab_frame()" not in source
        assert "cv2.imshow" not in source


def test_shader_source_and_compilation_are_outside_main_gpu_pipeline() -> None:
    pipeline_source = inspect.getsource(d3d11_gpu_pipeline)
    shader_source = inspect.getsource(d3d11_shaders)

    assert "D3DCompile" not in pipeline_source
    assert "Texture2D<float4> sourceTexture" not in pipeline_source
    assert "D3DCompile" in shader_source
    assert "Texture2D<float4> sourceTexture" in shader_source


def test_original_scaler_name_aliases_reusable_scaling_pass() -> None:
    assert D3D11ShaderScaler is D3D11ScalingPass
    assert hasattr(D3D11ScalingPass, "execute")


@pytest.mark.parametrize(
    ("method", "pixel_entry"),
    (
        ("nearest", b"PSMain"),
        ("bilinear", b"PSMain"),
        ("lanczos", b"PSLanczos"),
        ("fsr1_like", b"PSFsr1Like"),
    ),
)
def test_scaler_method_selects_expected_pixel_shader(
    monkeypatch,
    method: str,
    pixel_entry: bytes,
) -> None:
    captured: dict[str, object] = {}

    def fake_init(self, device, **kwargs) -> None:
        del self, device
        captured.update(kwargs)

    monkeypatch.setattr(d3d11_shaders.D3D11ShaderProgram, "__init__", fake_init)

    d3d11_shaders.D3D11ShaderProgram.fullscreen_scaler(
        ctypes.c_void_p(1),
        method=method,
    )

    assert captured["vertex_entry"] == b"VSMain"
    assert captured["pixel_entry"] == pixel_entry


def test_shader_program_rejects_unknown_scaler_without_fallback() -> None:
    with pytest.raises(d3d11_shaders.D3D11ShaderError, match="unknown"):
        d3d11_shaders.D3D11ShaderProgram.fullscreen_scaler(
            ctypes.c_void_p(1),
            method="unknown",
        )


def test_lanczos_shader_uses_dimensions_four_by_four_taps_and_clamped_edges() -> None:
    source = d3d11_shaders.FULLSCREEN_SCALE_SHADER_SOURCE.decode("ascii")

    assert "float2 sourceSize" in source
    assert "float2 outputSize" in source
    assert "PSLanczos" in source
    assert "for (int y = -1; y <= 2; ++y)" in source
    assert "for (int x = -1; x <= 2; ++x)" in source
    assert "clamp(basePosition" in source
    assert ctypes.sizeof(d3d11_gpu_pipeline._ScalingConstants) == 32


def test_fsr1_like_shader_has_edge_blend_and_locally_clamped_sharpening() -> None:
    source = d3d11_shaders.FULLSCREEN_SCALE_SHADER_SOURCE.decode("ascii")

    assert "PSFsr1Like" in source
    assert "SourceEdgeMask" in source
    assert "gradientX" in source and "gradientY" in source
    assert "lerp(bilinear, nearest, edgeBlend)" in source
    assert "sharpeningStrength * (center.rgb - blurred)" in source
    assert "clamp(sharpened, localMinimum, localMaximum)" in source


def test_scaling_pass_uploads_dimensions_and_fsr1_like_settings(monkeypatch) -> None:
    uploaded: list[tuple[float, ...]] = []
    bound_buffers: list[int] = []

    def fake_vtable_function(pointer, slot, restype, *argtypes):
        del pointer, restype, argtypes

        def call(*args):
            if slot == 48:
                constants = ctypes.cast(
                    args[4],
                    ctypes.POINTER(d3d11_gpu_pipeline._ScalingConstants),
                ).contents
                uploaded.append(
                    (
                        constants.source_width,
                        constants.source_height,
                        constants.output_width,
                        constants.output_height,
                        constants.edge_strength,
                        constants.sharpening_strength,
                        constants.sharpening_enabled,
                        constants.padding,
                    )
                )
            elif slot == 16:
                bound_buffers.append(args[3][0])

        return call

    monkeypatch.setattr(d3d11_gpu_pipeline, "_vtable_function", fake_vtable_function)
    scaling_pass = object.__new__(D3D11ScalingPass)
    scaling_pass._context = ctypes.c_void_p(88)
    scaling_pass._constant_buffer = ctypes.c_void_p(99)
    scaling_pass.fsr1_like_edge_strength = 0.6
    scaling_pass.fsr1_like_sharpening_strength = 0.4
    scaling_pass.fsr1_like_sharpening_enabled = False

    scaling_pass._bind_scaling_constants(
        source_width=1280,
        source_height=720,
        output_width=1920,
        output_height=1080,
    )

    assert uploaded[0] == pytest.approx(
        (1280.0, 720.0, 1920.0, 1080.0, 0.6, 0.4, 0.0, 0.0)
    )
    assert bound_buffers == [99]


def test_fsr1_like_pass_selects_program_and_creates_constant_buffer(monkeypatch) -> None:
    selected_methods: list[str] = []
    created_buffer_sizes: list[int] = []

    class FakeProgram:
        def close(self) -> None:
            pass

    def fake_fullscreen_scaler(cls, device, *, method):
        del cls, device
        selected_methods.append(method)
        return FakeProgram()

    def fake_vtable_function(pointer, slot, restype, *argtypes):
        del pointer, restype, argtypes

        def call(*args):
            if slot == 3:
                desc = ctypes.cast(
                    args[1],
                    ctypes.POINTER(d3d11_gpu_pipeline._D3D11BufferDesc),
                ).contents
                created_buffer_sizes.append(desc.ByteWidth)
            return 0

        return call

    monkeypatch.setattr(
        d3d11_shaders.D3D11ShaderProgram,
        "fullscreen_scaler",
        classmethod(fake_fullscreen_scaler),
    )
    monkeypatch.setattr(d3d11_gpu_pipeline, "D3D11ShaderProgram", d3d11_shaders.D3D11ShaderProgram)
    monkeypatch.setattr(d3d11_gpu_pipeline, "_vtable_function", fake_vtable_function)
    scaling_pass = object.__new__(D3D11ScalingPass)
    scaling_pass.method = "fsr1_like"
    scaling_pass._device = ctypes.c_void_p(1)
    scaling_pass._program = None
    scaling_pass._sampler = ctypes.c_void_p()
    scaling_pass._constant_buffer = ctypes.c_void_p()

    scaling_pass._create_pipeline_state()

    assert selected_methods == ["fsr1_like"]
    assert created_buffer_sizes == [ctypes.sizeof(d3d11_gpu_pipeline._ScalingConstants)]


def test_shader_program_releases_owned_interfaces_once(monkeypatch) -> None:
    released: list[int] = []

    def fake_release(pointer):
        if pointer is not None and pointer.value:
            released.append(pointer.value)
            pointer.value = None

    monkeypatch.setattr(d3d11_shaders, "_release", fake_release)
    program = object.__new__(d3d11_shaders.D3D11ShaderProgram)
    program._vertex_shader = ctypes.c_void_p(101)
    program._pixel_shader = ctypes.c_void_p(202)
    program._closed = False

    program.close()
    program.close()

    assert released == [202, 101]


def test_scaling_pass_releases_shader_resources_once(monkeypatch) -> None:
    events: list[int | str] = []

    def fake_release(pointer):
        if pointer is not None and pointer.value:
            events.append(pointer.value)
            pointer.value = None

    class FakeProgram:
        def close(self) -> None:
            events.append("program")

    monkeypatch.setattr(d3d11_gpu_pipeline, "_release", fake_release)
    scaling_pass = object.__new__(D3D11ScalingPass)
    scaling_pass._closed = False
    scaling_pass._context = ctypes.c_void_p()
    scaling_pass._source_srv = ctypes.c_void_p(101)
    scaling_pass._source_texture = ctypes.c_void_p(202)
    scaling_pass._sampler = ctypes.c_void_p(303)
    scaling_pass._constant_buffer = ctypes.c_void_p(404)
    scaling_pass._program = FakeProgram()
    scaling_pass._device = ctypes.c_void_p(505)

    scaling_pass.close()
    scaling_pass.close()

    assert events == [101, 202, 303, 404, "program", 505]
