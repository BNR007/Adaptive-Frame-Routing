"""Original D3D11 WGC texture -> shader scaler -> swap-chain pipeline.

The normal path in this module never stages or maps frame pixels and never
creates a NumPy image.  One D3D11 device and immediate context are created by
the WGC runtime and deliberately shared by capture, scaling, and presentation.

Timing values are CPU-side API submission/wait measurements.  In particular,
``present_submit_ms`` can include vertical-sync blocking and is not GPU shader
execution time.  This first milestone does not issue D3D11 timestamp queries.
"""

from __future__ import annotations

import ctypes
import logging
import math
import statistics
import time
from ctypes import wintypes
from dataclasses import dataclass
from typing import Any

from config import (
    FSR1_LIKE_DEFAULT_EDGE_STRENGTH,
    FSR1_LIKE_DEFAULT_SHARPENING_ENABLED,
    FSR1_LIKE_DEFAULT_SHARPENING_STRENGTH,
    validate_fsr1_like_sharpening_enabled,
    validate_fsr1_like_strength,
)
from d3d11_shaders import D3D11ShaderError, D3D11ShaderProgram
from frame_pacing import PresentationFrame
from resource_validation import ResourceLease
from wgc_capture import (
    D3D11Frame,
    WGCCaptureBackend,
    WGCError,
    _D3D11Texture2DDesc,
    _DXGISampleDesc,
    _GUID,
    _add_ref,
    _check_hresult,
    _query_interface,
    _release,
    _vtable_function,
)

logger = logging.getLogger(__name__)

DXGI_FORMAT_B8G8R8A8_UNORM = 87
DXGI_FORMAT_NAME = "DXGI_FORMAT_B8G8R8A8_UNORM"
SUPPORTED_D3D11_SCALERS = ("nearest", "bilinear", "lanczos", "fsr1_like")

_D3D11_BIND_CONSTANT_BUFFER = 0x4
_D3D11_BIND_SHADER_RESOURCE = 0x8
_D3D11_USAGE_DEFAULT = 0
_D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST = 4
_D3D11_FILTER_MIN_MAG_MIP_POINT = 0
_D3D11_FILTER_MIN_MAG_MIP_LINEAR = 0x15
_D3D11_TEXTURE_ADDRESS_CLAMP = 3
_D3D11_COMPARISON_NEVER = 1
_DXGI_USAGE_RENDER_TARGET_OUTPUT = 0x20
_DXGI_SCALING_STRETCH = 0
_DXGI_SWAP_EFFECT_FLIP_DISCARD = 4
_DXGI_ALPHA_MODE_IGNORE = 3
_DXGI_MWA_NO_ALT_ENTER = 0x2

_IID_IDXGI_DEVICE = _GUID.from_string("54EC77FA-1377-44E6-8C32-88FD5F44C84C")
_IID_IDXGI_FACTORY2 = _GUID.from_string("50C83A1C-E072-4C48-87B0-3630FA36A6D0")
_IID_D3D11_TEXTURE2D = _GUID.from_string("6F15AAF2-D208-4E89-9AB4-489535D34F9C")


class D3D11GpuError(RuntimeError):
    """Base exception for the GPU-resident runtime path."""


class D3D11CapabilityError(D3D11GpuError):
    """Raised for an explicitly requested unsupported GPU combination."""


class D3D11PresentationError(D3D11GpuError):
    """Raised when the Win32/DXGI presenter cannot continue."""


def validate_gpu_pipeline_request(
    *,
    capture_backend: str,
    method: str,
    output_width: int,
    output_height: int,
    selected_window: bool,
    no_preview: bool = False,
) -> None:
    """Validate explicit GPU mode without silently substituting a CPU path."""

    if capture_backend not in ("auto", "wgc"):
        raise D3D11CapabilityError(
            "The D3D11 GPU path requires selected-window WGC capture; "
            f"capture backend '{capture_backend}' is incompatible."
        )
    if not selected_window:
        raise D3D11CapabilityError(
            "The D3D11 GPU path requires a visible window selected by title or HWND."
        )
    if method not in SUPPORTED_D3D11_SCALERS:
        raise D3D11CapabilityError(
            f"D3D11 GPU scaling does not support '{method}'. Supported methods: "
            f"{', '.join(SUPPORTED_D3D11_SCALERS)}. Select the CPU pipeline for this method."
        )
    if output_width <= 0 or output_height <= 0:
        raise D3D11CapabilityError("D3D11 output dimensions must be positive.")
    if no_preview:
        raise D3D11CapabilityError(
            "The D3D11 GPU path presents through its own swap-chain window and is "
            "incompatible with --no-preview."
        )


@dataclass(frozen=True, slots=True)
class LatencySummary:
    average_ms: float
    median_ms: float
    p95_ms: float


@dataclass(frozen=True, slots=True)
class GpuPipelineReport:
    duration_seconds: float
    presented_frames: int
    presented_fps: float
    acquisition: LatencySummary
    scale_submit: LatencySummary
    present_submit: LatencySummary
    cpu_loop: LatencySummary
    replaced_frames: int
    adapter_description: str
    source_width: int
    source_height: int
    output_width: int
    output_height: int
    source_format: str = DXGI_FORMAT_NAME
    swap_chain_format: str = DXGI_FORMAT_NAME
    vsync_enabled: bool = True
    gpu_timestamp_queries: bool = False


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _summarize(values: list[float]) -> LatencySummary:
    if not values:
        return LatencySummary(0.0, 0.0, 0.0)
    return LatencySummary(
        average_ms=statistics.fmean(values),
        median_ms=statistics.median(values),
        p95_ms=_percentile(values, 0.95),
    )


class GpuPipelineMetrics:
    """Monotonic CPU-side samples for the synchronous GPU loop."""

    def __init__(self) -> None:
        self.started_at = time.perf_counter()
        self.acquisition_ms: list[float] = []
        self.scale_submit_ms: list[float] = []
        self.present_submit_ms: list[float] = []
        self.cpu_loop_ms: list[float] = []
        self.source_width = 0
        self.source_height = 0
        self.output_width = 0
        self.output_height = 0

    def record(
        self,
        *,
        acquisition_ms: float,
        scale_submit_ms: float,
        present_submit_ms: float,
        cpu_loop_ms: float,
        source_size: tuple[int, int],
        output_size: tuple[int, int],
    ) -> None:
        self.acquisition_ms.append(acquisition_ms)
        self.scale_submit_ms.append(scale_submit_ms)
        self.present_submit_ms.append(present_submit_ms)
        self.cpu_loop_ms.append(cpu_loop_ms)
        self.source_width, self.source_height = source_size
        self.output_width, self.output_height = output_size

    def report(
        self,
        *,
        adapter_description: str,
        replaced_frames: int,
        ended_at: float | None = None,
    ) -> GpuPipelineReport:
        finish = time.perf_counter() if ended_at is None else ended_at
        duration = max(0.0, finish - self.started_at)
        count = len(self.cpu_loop_ms)
        return GpuPipelineReport(
            duration_seconds=duration,
            presented_frames=count,
            presented_fps=(count / duration) if duration else 0.0,
            acquisition=_summarize(self.acquisition_ms),
            scale_submit=_summarize(self.scale_submit_ms),
            present_submit=_summarize(self.present_submit_ms),
            cpu_loop=_summarize(self.cpu_loop_ms),
            replaced_frames=replaced_frames,
            adapter_description=adapter_description,
            source_width=self.source_width,
            source_height=self.source_height,
            output_width=self.output_width,
            output_height=self.output_height,
        )


class _D3D11Box(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.UINT),
        ("top", wintypes.UINT),
        ("front", wintypes.UINT),
        ("right", wintypes.UINT),
        ("bottom", wintypes.UINT),
        ("back", wintypes.UINT),
    ]


class _D3D11Viewport(ctypes.Structure):
    _fields_ = [
        ("TopLeftX", ctypes.c_float),
        ("TopLeftY", ctypes.c_float),
        ("Width", ctypes.c_float),
        ("Height", ctypes.c_float),
        ("MinDepth", ctypes.c_float),
        ("MaxDepth", ctypes.c_float),
    ]


class _D3D11SamplerDesc(ctypes.Structure):
    _fields_ = [
        ("Filter", wintypes.UINT),
        ("AddressU", wintypes.UINT),
        ("AddressV", wintypes.UINT),
        ("AddressW", wintypes.UINT),
        ("MipLODBias", ctypes.c_float),
        ("MaxAnisotropy", wintypes.UINT),
        ("ComparisonFunc", wintypes.UINT),
        ("BorderColor", ctypes.c_float * 4),
        ("MinLOD", ctypes.c_float),
        ("MaxLOD", ctypes.c_float),
    ]


class _D3D11BufferDesc(ctypes.Structure):
    _fields_ = [
        ("ByteWidth", wintypes.UINT),
        ("Usage", wintypes.UINT),
        ("BindFlags", wintypes.UINT),
        ("CPUAccessFlags", wintypes.UINT),
        ("MiscFlags", wintypes.UINT),
        ("StructureByteStride", wintypes.UINT),
    ]


class _ScalingConstants(ctypes.Structure):
    _fields_ = [
        ("source_width", ctypes.c_float),
        ("source_height", ctypes.c_float),
        ("output_width", ctypes.c_float),
        ("output_height", ctypes.c_float),
        ("edge_strength", ctypes.c_float),
        ("sharpening_strength", ctypes.c_float),
        ("sharpening_enabled", ctypes.c_float),
        ("padding", ctypes.c_float),
    ]


class _DXGISwapChainDesc1(ctypes.Structure):
    _fields_ = [
        ("Width", wintypes.UINT),
        ("Height", wintypes.UINT),
        ("Format", wintypes.UINT),
        ("Stereo", wintypes.BOOL),
        ("SampleDesc", _DXGISampleDesc),
        ("BufferUsage", wintypes.UINT),
        ("BufferCount", wintypes.UINT),
        ("Scaling", wintypes.UINT),
        ("SwapEffect", wintypes.UINT),
        ("AlphaMode", wintypes.UINT),
        ("Flags", wintypes.UINT),
    ]


class _Luid(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]


class _DXGIAdapterDesc(ctypes.Structure):
    _fields_ = [
        ("Description", wintypes.WCHAR * 128),
        ("VendorId", wintypes.UINT),
        ("DeviceId", wintypes.UINT),
        ("SubSysId", wintypes.UINT),
        ("Revision", wintypes.UINT),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", _Luid),
    ]


def describe_dxgi_adapter(device: ctypes.c_void_p) -> str:
    """Return the adapter attached to the exact D3D11 device WGC owns."""

    dxgi_device = ctypes.c_void_p()
    adapter = ctypes.c_void_p()
    try:
        dxgi_device = _query_interface(device, _IID_IDXGI_DEVICE, "QueryInterface(IDXGIDevice)")
        result = _vtable_function(
            dxgi_device, 7, ctypes.c_long, ctypes.POINTER(ctypes.c_void_p)
        )(dxgi_device, ctypes.byref(adapter))
        _check_hresult(result, "IDXGIDevice.GetAdapter")
        desc = _DXGIAdapterDesc()
        result = _vtable_function(
            adapter, 8, ctypes.c_long, ctypes.POINTER(_DXGIAdapterDesc)
        )(adapter, ctypes.byref(desc))
        _check_hresult(result, "IDXGIAdapter.GetDesc")
        return str(desc.Description).rstrip("\x00") or "Unknown DXGI adapter"
    except WGCError as error:
        raise D3D11GpuError(str(error)) from error
    finally:
        _release(adapter)
        _release(dxgi_device)


class D3D11ScalingPass:
    """Reusable texture-to-render-target fullscreen scaling pass."""

    def __init__(
        self,
        device: ctypes.c_void_p,
        context: ctypes.c_void_p,
        *,
        method: str,
        fsr1_like_edge_strength: float = FSR1_LIKE_DEFAULT_EDGE_STRENGTH,
        fsr1_like_sharpening_strength: float = FSR1_LIKE_DEFAULT_SHARPENING_STRENGTH,
        fsr1_like_sharpening_enabled: bool = FSR1_LIKE_DEFAULT_SHARPENING_ENABLED,
    ) -> None:
        if method not in SUPPORTED_D3D11_SCALERS:
            raise D3D11CapabilityError(f"Unsupported D3D11 scaler method: {method}")
        self.method = method
        self.fsr1_like_edge_strength = validate_fsr1_like_strength(
            fsr1_like_edge_strength,
            "fsr1_like_edge_strength",
        )
        self.fsr1_like_sharpening_strength = validate_fsr1_like_strength(
            fsr1_like_sharpening_strength,
            "fsr1_like_sharpening_strength",
        )
        self.fsr1_like_sharpening_enabled = validate_fsr1_like_sharpening_enabled(
            fsr1_like_sharpening_enabled
        )
        self._device = _add_ref(device)
        self._context = _add_ref(context)
        self._program: D3D11ShaderProgram | None = None
        self._sampler = ctypes.c_void_p()
        self._constant_buffer = ctypes.c_void_p()
        self._source_texture = ctypes.c_void_p()
        self._source_srv = ctypes.c_void_p()
        self._source_signature: tuple[int, int, int] | None = None
        self.resource_generation = 0
        self._closed = False
        try:
            self._create_pipeline_state()
        except Exception:
            self.close()
            raise

    def _create_pipeline_state(self) -> None:
        try:
            self._program = D3D11ShaderProgram.fullscreen_scaler(
                self._device,
                method=self.method,
            )
        except D3D11ShaderError as error:
            raise D3D11CapabilityError(str(error)) from error

        filter_mode = (
            _D3D11_FILTER_MIN_MAG_MIP_POINT
            if self.method == "nearest"
            else _D3D11_FILTER_MIN_MAG_MIP_LINEAR
        )
        sampler_desc = _D3D11SamplerDesc(
            Filter=filter_mode,
            AddressU=_D3D11_TEXTURE_ADDRESS_CLAMP,
            AddressV=_D3D11_TEXTURE_ADDRESS_CLAMP,
            AddressW=_D3D11_TEXTURE_ADDRESS_CLAMP,
            MipLODBias=0.0,
            MaxAnisotropy=1,
            ComparisonFunc=_D3D11_COMPARISON_NEVER,
            BorderColor=(ctypes.c_float * 4)(0.0, 0.0, 0.0, 0.0),
            MinLOD=0.0,
            MaxLOD=3.402823466e38,
        )
        result = _vtable_function(
            self._device,
            23,
            ctypes.c_long,
            ctypes.POINTER(_D3D11SamplerDesc),
            ctypes.POINTER(ctypes.c_void_p),
        )(self._device, ctypes.byref(sampler_desc), ctypes.byref(self._sampler))
        _check_hresult(result, "ID3D11Device.CreateSamplerState")

        if self.method in ("lanczos", "fsr1_like"):
            constant_buffer_desc = _D3D11BufferDesc(
                ByteWidth=ctypes.sizeof(_ScalingConstants),
                Usage=_D3D11_USAGE_DEFAULT,
                BindFlags=_D3D11_BIND_CONSTANT_BUFFER,
                CPUAccessFlags=0,
                MiscFlags=0,
                StructureByteStride=0,
            )
            result = _vtable_function(
                self._device,
                3,
                ctypes.c_long,
                ctypes.POINTER(_D3D11BufferDesc),
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
            )(
                self._device,
                ctypes.byref(constant_buffer_desc),
                None,
                ctypes.byref(self._constant_buffer),
            )
            _check_hresult(result, "ID3D11Device.CreateBuffer(scaling constants)")

    def _bind_scaling_constants(
        self,
        *,
        source_width: int,
        source_height: int,
        output_width: int,
        output_height: int,
    ) -> None:
        if not self._constant_buffer.value:
            return
        constants = _ScalingConstants(
            float(source_width),
            float(source_height),
            float(output_width),
            float(output_height),
            self.fsr1_like_edge_strength,
            self.fsr1_like_sharpening_strength,
            1.0 if self.fsr1_like_sharpening_enabled else 0.0,
            0.0,
        )
        _vtable_function(
            self._context,
            48,
            None,
            ctypes.c_void_p,
            wintypes.UINT,
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.UINT,
            wintypes.UINT,
        )(
            self._context,
            self._constant_buffer,
            0,
            None,
            ctypes.byref(constants),
            0,
            0,
        )
        buffer_array = (ctypes.c_void_p * 1)(self._constant_buffer.value)
        _vtable_function(
            self._context,
            16,
            None,
            wintypes.UINT,
            wintypes.UINT,
            ctypes.POINTER(ctypes.c_void_p),
        )(self._context, 0, 1, buffer_array)

    def _ensure_source(self, frame: D3D11Frame) -> None:
        signature = (frame.width, frame.height, frame.dxgi_format)
        if signature == self._source_signature and self._source_texture.value:
            return

        self._unbind_views()
        _release(self._source_srv)
        _release(self._source_texture)
        desc = _D3D11Texture2DDesc(
            Width=frame.width,
            Height=frame.height,
            MipLevels=1,
            ArraySize=1,
            Format=frame.dxgi_format,
            SampleDesc=_DXGISampleDesc(1, 0),
            Usage=_D3D11_USAGE_DEFAULT,
            BindFlags=_D3D11_BIND_SHADER_RESOURCE,
            CPUAccessFlags=0,
            MiscFlags=0,
        )
        result = _vtable_function(
            self._device,
            5,
            ctypes.c_long,
            ctypes.POINTER(_D3D11Texture2DDesc),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )(self._device, ctypes.byref(desc), None, ctypes.byref(self._source_texture))
        _check_hresult(result, "ID3D11Device.CreateTexture2D(shader source)")
        result = _vtable_function(
            self._device,
            7,
            ctypes.c_long,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )(self._device, self._source_texture, None, ctypes.byref(self._source_srv))
        _check_hresult(result, "ID3D11Device.CreateShaderResourceView")
        self._source_signature = signature
        self.resource_generation += 1

    def execute(
        self,
        frame: D3D11Frame,
        *,
        render_target_view: ctypes.c_void_p,
        output_width: int,
        output_height: int,
    ) -> None:
        if self._closed or self._program is None:
            raise D3D11GpuError("D3D11 scaling pass is closed.")
        if output_width <= 0 or output_height <= 0:
            raise D3D11GpuError("D3D11 render dimensions must be positive.")
        if not render_target_view or not render_target_view.value:
            raise D3D11GpuError("D3D11 render target is unavailable.")
        self._ensure_source(frame)

        source_box = _D3D11Box(0, 0, 0, frame.width, frame.height, 1)
        # CopySubresourceRegion is a GPU-to-GPU copy into one reusable,
        # shader-readable texture.  It also safely decouples the WGC pool
        # surface lifetime from later rendering commands.
        _vtable_function(
            self._context,
            46,
            None,
            ctypes.c_void_p,
            wintypes.UINT,
            wintypes.UINT,
            wintypes.UINT,
            wintypes.UINT,
            ctypes.c_void_p,
            wintypes.UINT,
            ctypes.POINTER(_D3D11Box),
        )(
            self._context,
            self._source_texture,
            0,
            0,
            0,
            0,
            frame.texture_pointer,
            0,
            ctypes.byref(source_box),
        )

        _vtable_function(self._context, 24, None, wintypes.UINT)(
            self._context, _D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST
        )
        _vtable_function(
            self._context, 11, None, ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT
        )(self._context, self._program.vertex_shader, None, 0)
        _vtable_function(
            self._context, 9, None, ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT
        )(self._context, self._program.pixel_shader, None, 0)
        self._bind_scaling_constants(
            source_width=frame.width,
            source_height=frame.height,
            output_width=output_width,
            output_height=output_height,
        )

        srv_array = (ctypes.c_void_p * 1)(self._source_srv.value)
        sampler_array = (ctypes.c_void_p * 1)(self._sampler.value)
        target_array = (ctypes.c_void_p * 1)(render_target_view.value)
        _vtable_function(
            self._context,
            8,
            None,
            wintypes.UINT,
            wintypes.UINT,
            ctypes.POINTER(ctypes.c_void_p),
        )(self._context, 0, 1, srv_array)
        _vtable_function(
            self._context,
            10,
            None,
            wintypes.UINT,
            wintypes.UINT,
            ctypes.POINTER(ctypes.c_void_p),
        )(self._context, 0, 1, sampler_array)
        _vtable_function(
            self._context,
            33,
            None,
            wintypes.UINT,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        )(self._context, 1, target_array, None)

        viewport = _D3D11Viewport(
            0.0,
            0.0,
            float(output_width),
            float(output_height),
            0.0,
            1.0,
        )
        _vtable_function(
            self._context,
            44,
            None,
            wintypes.UINT,
            ctypes.POINTER(_D3D11Viewport),
        )(self._context, 1, ctypes.byref(viewport))
        _vtable_function(self._context, 13, None, wintypes.UINT, wintypes.UINT)(
            self._context, 3, 0
        )
        self._unbind_views()

    def render(
        self,
        frame: D3D11Frame,
        *,
        render_target_view: ctypes.c_void_p,
        output_width: int,
        output_height: int,
    ) -> None:
        """Compatibility wrapper for the original scaler API."""

        self.execute(
            frame,
            render_target_view=render_target_view,
            output_width=output_width,
            output_height=output_height,
        )

    def _unbind_views(self) -> None:
        if not self._context.value:
            return
        null_view = (ctypes.c_void_p * 1)(None)
        _vtable_function(
            self._context,
            8,
            None,
            wintypes.UINT,
            wintypes.UINT,
            ctypes.POINTER(ctypes.c_void_p),
        )(self._context, 0, 1, null_view)
        _vtable_function(
            self._context,
            33,
            None,
            wintypes.UINT,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        )(self._context, 0, None, None)
        if self._constant_buffer.value:
            null_buffer = (ctypes.c_void_p * 1)(None)
            _vtable_function(
                self._context,
                16,
                None,
                wintypes.UINT,
                wintypes.UINT,
                ctypes.POINTER(ctypes.c_void_p),
            )(self._context, 0, 1, null_buffer)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._unbind_views()
        _release(self._source_srv)
        _release(self._source_texture)
        _release(self._sampler)
        _release(self._constant_buffer)
        if self._program is not None:
            self._program.close()
            self._program = None
        _release(self._context)
        _release(self._device)


# Keep the original name import-compatible while callers migrate to the pass API.
D3D11ShaderScaler = D3D11ScalingPass


class PresenterSizeState:
    """Testable resize/close state used by the native presenter."""

    def __init__(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Presenter dimensions must be positive.")
        self.width = width
        self.height = height
        self.pending: tuple[int, int] | None = None
        self.closed = False

    def request(self, width: int, height: int) -> None:
        if self.closed:
            raise D3D11PresentationError("Presenter is closed.")
        if width <= 0 or height <= 0:
            return
        if (width, height) != (self.width, self.height):
            self.pending = (width, height)

    def take_pending(self) -> tuple[int, int] | None:
        value = self.pending
        self.pending = None
        return value

    def commit(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def close(self) -> None:
        self.closed = True
        self.pending = None


_LRESULT = ctypes.c_ssize_t
_WPARAM = ctypes.c_size_t
_LPARAM = ctypes.c_ssize_t
_WNDPROC = ctypes.WINFUNCTYPE(_LRESULT, wintypes.HWND, wintypes.UINT, _WPARAM, _LPARAM)


class _WndClassEx(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.UINT),
        ("style", wintypes.UINT),
        ("lpfnWndProc", _WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
        ("hIconSm", wintypes.HICON),
    ]


class _Rect(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class _Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class _Msg(ctypes.Structure):
    _fields_ = [
        ("hwnd", wintypes.HWND),
        ("message", wintypes.UINT),
        ("wParam", _WPARAM),
        ("lParam", _LPARAM),
        ("time", wintypes.DWORD),
        ("pt", _Point),
        ("lPrivate", wintypes.DWORD),
    ]


class D3D11SwapChainPresenter:
    """Win32 HWND with a two-buffer flip-discard D3D11 swap chain."""

    _WM_DESTROY = 0x0002
    _WM_SIZE = 0x0005
    _WM_CLOSE = 0x0010
    _WM_KEYDOWN = 0x0100
    _WM_QUIT = 0x0012
    _SIZE_MINIMIZED = 1
    _PM_REMOVE = 0x0001
    _SW_SHOW = 5
    _WS_POPUP = 0x80000000
    # A borderless popup has an exact client size even when the hosting Python
    # process is not per-monitor-DPI aware.  Q, Alt+F4/WM_CLOSE, and Ctrl+C are
    # the deliberate shutdown mechanisms.
    _STYLE = _WS_POPUP

    def __init__(
        self,
        device: ctypes.c_void_p,
        context: ctypes.c_void_p,
        *,
        width: int,
        height: int,
        vsync: bool = True,
    ) -> None:
        self._device = _add_ref(device)
        self._context = _add_ref(context)
        self._swap_chain = ctypes.c_void_p()
        self._back_buffer = ctypes.c_void_p()
        self._render_target_view = ctypes.c_void_p()
        self._size = PresenterSizeState(width, height)
        self.vsync = bool(vsync)
        self._quit_requested = False
        self._closed = False
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._class_name = f"UniversalUpscalerD3D11_{id(self):x}"
        self._hinstance = wintypes.HINSTANCE()
        self._class_atom = 0
        self._hwnd = wintypes.HWND()
        self._wnd_proc = _WNDPROC(self._window_proc)
        self._configure_win32_signatures()
        try:
            self._create_window(width, height)
            self._create_swap_chain(width, height)
            self._create_render_target()
        except Exception as error:
            self.close()
            if isinstance(error, D3D11GpuError):
                raise
            raise D3D11PresentationError(
                f"D3D11 presenter initialization failed: {error}"
            ) from error

    def _configure_win32_signatures(self) -> None:
        """Prevent ctypes from truncating Win64 HWND/HINSTANCE parameters."""

        self._user32.DefWindowProcW.argtypes = [
            wintypes.HWND,
            wintypes.UINT,
            _WPARAM,
            _LPARAM,
        ]
        self._user32.DefWindowProcW.restype = _LRESULT
        self._user32.DestroyWindow.argtypes = [wintypes.HWND]
        self._user32.DestroyWindow.restype = wintypes.BOOL
        self._user32.PostQuitMessage.argtypes = [ctypes.c_int]
        self._user32.PostQuitMessage.restype = None
        self._user32.AdjustWindowRectEx.argtypes = [
            ctypes.POINTER(_Rect),
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        self._user32.AdjustWindowRectEx.restype = wintypes.BOOL
        self._user32.CreateWindowExW.argtypes = [
            wintypes.DWORD,
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
            wintypes.DWORD,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.HWND,
            wintypes.HMENU,
            wintypes.HINSTANCE,
            ctypes.c_void_p,
        ]
        self._user32.CreateWindowExW.restype = wintypes.HWND
        self._user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        self._user32.ShowWindow.restype = wintypes.BOOL
        self._user32.UpdateWindow.argtypes = [wintypes.HWND]
        self._user32.UpdateWindow.restype = wintypes.BOOL
        self._user32.SetForegroundWindow.argtypes = [wintypes.HWND]
        self._user32.SetForegroundWindow.restype = wintypes.BOOL
        self._user32.PeekMessageW.argtypes = [
            ctypes.POINTER(_Msg),
            wintypes.HWND,
            wintypes.UINT,
            wintypes.UINT,
            wintypes.UINT,
        ]
        self._user32.PeekMessageW.restype = wintypes.BOOL
        self._user32.TranslateMessage.argtypes = [ctypes.POINTER(_Msg)]
        self._user32.TranslateMessage.restype = wintypes.BOOL
        self._user32.DispatchMessageW.argtypes = [ctypes.POINTER(_Msg)]
        self._user32.DispatchMessageW.restype = _LRESULT
        self._user32.SetWindowPos.argtypes = [
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        ]
        self._user32.SetWindowPos.restype = wintypes.BOOL
        self._user32.IsWindow.argtypes = [wintypes.HWND]
        self._user32.IsWindow.restype = wintypes.BOOL
        self._user32.UnregisterClassW.argtypes = [wintypes.LPCWSTR, wintypes.HINSTANCE]
        self._user32.UnregisterClassW.restype = wintypes.BOOL

    @property
    def render_target_view(self) -> ctypes.c_void_p:
        if self._closed or not self._render_target_view.value:
            raise D3D11PresentationError("Swap-chain render target is unavailable.")
        return ctypes.c_void_p(self._render_target_view.value)

    @property
    def back_buffer(self) -> ctypes.c_void_p:
        if self._closed or not self._back_buffer.value:
            raise D3D11PresentationError("Swap-chain back buffer is unavailable.")
        return ctypes.c_void_p(self._back_buffer.value)

    @property
    def output_size(self) -> tuple[int, int]:
        return self._size.width, self._size.height

    @property
    def quit_requested(self) -> bool:
        return self._quit_requested

    @property
    def hwnd(self) -> int:
        return int(self._hwnd or 0)

    def _window_proc(self, hwnd, message, wparam, lparam):
        if message == self._WM_KEYDOWN and int(wparam) in (ord("Q"), ord("q")):
            self._quit_requested = True
            self._user32.DestroyWindow(hwnd)
            return 0
        if message == self._WM_CLOSE:
            self._quit_requested = True
            self._user32.DestroyWindow(hwnd)
            return 0
        if message == self._WM_DESTROY:
            self._quit_requested = True
            self._user32.PostQuitMessage(0)
            return 0
        if message == self._WM_SIZE and int(wparam) != self._SIZE_MINIMIZED:
            width = int(lparam) & 0xFFFF
            height = (int(lparam) >> 16) & 0xFFFF
            if width > 0 and height > 0:
                try:
                    self._size.request(width, height)
                except D3D11PresentationError:
                    pass
            return 0
        return self._user32.DefWindowProcW(hwnd, message, wparam, lparam)

    def _create_window(self, width: int, height: int) -> None:
        self._kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        self._kernel32.GetModuleHandleW.restype = wintypes.HMODULE
        self._hinstance = self._kernel32.GetModuleHandleW(None)

        window_class = _WndClassEx()
        window_class.cbSize = ctypes.sizeof(_WndClassEx)
        window_class.style = 0
        window_class.lpfnWndProc = self._wnd_proc
        window_class.hInstance = self._hinstance
        window_class.hbrBackground = wintypes.HBRUSH(6)
        window_class.lpszClassName = self._class_name
        self._user32.RegisterClassExW.argtypes = [ctypes.POINTER(_WndClassEx)]
        self._user32.RegisterClassExW.restype = wintypes.ATOM
        self._class_atom = self._user32.RegisterClassExW(ctypes.byref(window_class))
        if not self._class_atom:
            raise D3D11PresentationError(f"RegisterClassExW failed: {ctypes.WinError(ctypes.get_last_error())}")

        rect = _Rect(0, 0, width, height)
        if not self._user32.AdjustWindowRectEx(ctypes.byref(rect), self._STYLE, False, 0):
            raise D3D11PresentationError(f"AdjustWindowRectEx failed: {ctypes.WinError(ctypes.get_last_error())}")
        outer_width = rect.right - rect.left
        outer_height = rect.bottom - rect.top
        self._hwnd = self._user32.CreateWindowExW(
            0,
            self._class_name,
            "UniversalUpscaler Preview",
            self._STYLE,
            0,
            0,
            outer_width,
            outer_height,
            None,
            None,
            self._hinstance,
            None,
        )
        if not self._hwnd:
            raise D3D11PresentationError(f"CreateWindowExW failed: {ctypes.WinError(ctypes.get_last_error())}")
        self._user32.ShowWindow(self._hwnd, self._SW_SHOW)
        self._user32.UpdateWindow(self._hwnd)
        self._user32.SetForegroundWindow(self._hwnd)

    def _create_swap_chain(self, width: int, height: int) -> None:
        dxgi_device = ctypes.c_void_p()
        adapter = ctypes.c_void_p()
        factory = ctypes.c_void_p()
        try:
            dxgi_device = _query_interface(
                self._device, _IID_IDXGI_DEVICE, "QueryInterface(IDXGIDevice)"
            )
            result = _vtable_function(
                dxgi_device, 7, ctypes.c_long, ctypes.POINTER(ctypes.c_void_p)
            )(dxgi_device, ctypes.byref(adapter))
            _check_hresult(result, "IDXGIDevice.GetAdapter")
            result = _vtable_function(
                adapter,
                6,
                ctypes.c_long,
                ctypes.POINTER(_GUID),
                ctypes.POINTER(ctypes.c_void_p),
            )(
                adapter,
                ctypes.byref(_IID_IDXGI_FACTORY2),
                ctypes.byref(factory),
            )
            _check_hresult(result, "IDXGIAdapter.GetParent(IDXGIFactory2)")
            _vtable_function(factory, 8, ctypes.c_long, wintypes.HWND, wintypes.UINT)(
                factory, self._hwnd, _DXGI_MWA_NO_ALT_ENTER
            )

            desc = _DXGISwapChainDesc1(
                Width=width,
                Height=height,
                Format=DXGI_FORMAT_B8G8R8A8_UNORM,
                Stereo=False,
                SampleDesc=_DXGISampleDesc(1, 0),
                BufferUsage=_DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount=2,
                Scaling=_DXGI_SCALING_STRETCH,
                SwapEffect=_DXGI_SWAP_EFFECT_FLIP_DISCARD,
                AlphaMode=_DXGI_ALPHA_MODE_IGNORE,
                Flags=0,
            )
            result = _vtable_function(
                factory,
                15,
                ctypes.c_long,
                ctypes.c_void_p,
                wintypes.HWND,
                ctypes.POINTER(_DXGISwapChainDesc1),
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
            )(
                factory,
                self._device,
                self._hwnd,
                ctypes.byref(desc),
                None,
                None,
                ctypes.byref(self._swap_chain),
            )
            _check_hresult(result, "IDXGIFactory2.CreateSwapChainForHwnd")
        except WGCError as error:
            raise D3D11PresentationError(str(error)) from error
        finally:
            _release(factory)
            _release(adapter)
            _release(dxgi_device)

    def _create_render_target(self) -> None:
        _release(self._back_buffer)
        try:
            result = _vtable_function(
                self._swap_chain,
                9,
                ctypes.c_long,
                wintypes.UINT,
                ctypes.POINTER(_GUID),
                ctypes.POINTER(ctypes.c_void_p),
            )(
                self._swap_chain,
                0,
                ctypes.byref(_IID_D3D11_TEXTURE2D),
                ctypes.byref(self._back_buffer),
            )
            _check_hresult(result, "IDXGISwapChain.GetBuffer")
            result = _vtable_function(
                self._device,
                9,
                ctypes.c_long,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
            )(
                self._device,
                self._back_buffer,
                None,
                ctypes.byref(self._render_target_view),
            )
            _check_hresult(result, "ID3D11Device.CreateRenderTargetView")
        except WGCError as error:
            raise D3D11PresentationError(str(error)) from error

    def pump_messages(self) -> None:
        message = _Msg()
        while self._user32.PeekMessageW(
            ctypes.byref(message), None, 0, 0, self._PM_REMOVE
        ):
            if message.message == self._WM_QUIT:
                self._quit_requested = True
                break
            self._user32.TranslateMessage(ctypes.byref(message))
            self._user32.DispatchMessageW(ctypes.byref(message))

    def apply_pending_resize(self) -> bool:
        pending = self._size.take_pending()
        if pending is None or self._closed:
            return False
        width, height = pending
        self._unbind_render_target()
        _release(self._render_target_view)
        _release(self._back_buffer)
        result = _vtable_function(
            self._swap_chain,
            13,
            ctypes.c_long,
            wintypes.UINT,
            wintypes.UINT,
            wintypes.UINT,
            wintypes.UINT,
            wintypes.UINT,
        )(self._swap_chain, 0, width, height, DXGI_FORMAT_B8G8R8A8_UNORM, 0)
        try:
            _check_hresult(result, "IDXGISwapChain.ResizeBuffers")
            self._create_render_target()
        except WGCError as error:
            raise D3D11PresentationError(str(error)) from error
        self._size.commit(width, height)
        logger.info("D3D11 presenter resized to %sx%s", width, height)
        return True

    def resize(self, width: int, height: int) -> None:
        """Programmatically resize the fixed-style presentation client area."""

        self._size.request(width, height)
        rect = _Rect(0, 0, width, height)
        if not self._user32.AdjustWindowRectEx(ctypes.byref(rect), self._STYLE, False, 0):
            raise D3D11PresentationError(f"AdjustWindowRectEx failed: {ctypes.WinError(ctypes.get_last_error())}")
        outer_width = rect.right - rect.left
        outer_height = rect.bottom - rect.top
        if not self._user32.SetWindowPos(
            self._hwnd, None, 0, 0, outer_width, outer_height, 0x0002 | 0x0004
        ):
            raise D3D11PresentationError(f"SetWindowPos failed: {ctypes.WinError(ctypes.get_last_error())}")

    def present(self) -> None:
        if self._closed:
            raise D3D11PresentationError("Presenter is closed.")
        result = _vtable_function(
            self._swap_chain, 8, ctypes.c_long, wintypes.UINT, wintypes.UINT
        )(self._swap_chain, 1 if self.vsync else 0, 0)
        try:
            _check_hresult(result, "IDXGISwapChain.Present")
        except WGCError as error:
            raise D3D11PresentationError(str(error)) from error

    def _unbind_render_target(self) -> None:
        if self._context.value:
            _vtable_function(
                self._context,
                33,
                None,
                wintypes.UINT,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_void_p,
            )(self._context, 0, None, None)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._size.close()
        self._quit_requested = True
        self._unbind_render_target()
        _release(self._render_target_view)
        _release(self._back_buffer)
        _release(self._swap_chain)
        if self._hwnd and self._user32.IsWindow(self._hwnd):
            self._user32.DestroyWindow(self._hwnd)
        self._hwnd = wintypes.HWND()
        if self._class_atom:
            self._user32.UnregisterClassW(self._class_name, self._hinstance)
            self._class_atom = 0
        _release(self._context)
        _release(self._device)


class D3D11GpuPipeline:
    """Dedicated synchronous GPU-resident WGC scaling/presentation loop."""

    def __init__(
        self,
        *,
        hwnd: int,
        output_width: int,
        output_height: int,
        method: str,
        vsync: bool = True,
        fsr1_like_edge_strength: float = FSR1_LIKE_DEFAULT_EDGE_STRENGTH,
        fsr1_like_sharpening_strength: float = FSR1_LIKE_DEFAULT_SHARPENING_STRENGTH,
        fsr1_like_sharpening_enabled: bool = FSR1_LIKE_DEFAULT_SHARPENING_ENABLED,
        frame_pacer: Any | None = None,
    ) -> None:
        validate_gpu_pipeline_request(
            capture_backend="wgc",
            method=method,
            output_width=output_width,
            output_height=output_height,
            selected_window=True,
        )
        self.capture: WGCCaptureBackend | None = None
        self.scaling_pass: D3D11ScalingPass | None = None
        self.presenter: D3D11SwapChainPresenter | None = None
        self.adapter_description = "Unknown DXGI adapter"
        self.method = method
        self.frame_pacer = frame_pacer
        self._closed = False
        self._validation_resource_lease = ResourceLease("d3d_resources")
        try:
            self.capture = WGCCaptureBackend(hwnd)
            device = self.capture.d3d11_device_pointer
            context = self.capture.d3d11_context_pointer
            self.adapter_description = describe_dxgi_adapter(device)
            self.scaling_pass = D3D11ScalingPass(
                device,
                context,
                method=method,
                fsr1_like_edge_strength=fsr1_like_edge_strength,
                fsr1_like_sharpening_strength=fsr1_like_sharpening_strength,
                fsr1_like_sharpening_enabled=fsr1_like_sharpening_enabled,
            )
            self.presenter = D3D11SwapChainPresenter(
                device,
                context,
                width=output_width,
                height=output_height,
                vsync=vsync,
            )
        except Exception:
            self.close()
            raise

        self._validation_resource_lease.acquire()

        logger.info("Active GPU path: D3D11 GPU")
        logger.info("Active capture backend: WGC")
        logger.info("Active DXGI adapter: %s", self.adapter_description)
        logger.info(
            "D3D11 scaler: %s | output %sx%s | format %s | flip-discard | vsync %s",
            method,
            output_width,
            output_height,
            DXGI_FORMAT_NAME,
            "on" if vsync else "off",
        )
        logger.info("Press Q in the D3D11 presentation window or Ctrl+C to quit.")
        logger.info("GPU timestamp queries: not implemented; metrics are CPU-side only.")

    @property
    def scaler(self) -> D3D11ScalingPass | None:
        """Backward-compatible view of the active scaling pass."""

        return self.scaling_pass

    def run(
        self,
        *,
        duration_seconds: float | None = None,
        warmup_seconds: float = 0.0,
    ) -> GpuPipelineReport:
        if (
            self._closed
            or self.capture is None
            or self.scaling_pass is None
            or self.presenter is None
        ):
            raise D3D11GpuError("D3D11 GPU pipeline is closed or incomplete.")
        if duration_seconds is not None and duration_seconds <= 0.0:
            raise ValueError("duration_seconds must be positive when supplied.")
        if warmup_seconds < 0.0:
            raise ValueError("warmup_seconds cannot be negative.")

        metrics = GpuPipelineMetrics()
        run_started = time.perf_counter()
        collect_after = run_started + warmup_seconds
        metrics.started_at = collect_after
        stop_at = (
            None
            if duration_seconds is None
            else collect_after + duration_seconds
        )
        next_log = run_started + 1.0

        while not self.presenter.quit_requested:
            self.presenter.pump_messages()
            if self.presenter.quit_requested:
                break
            self.presenter.apply_pending_resize()
            if stop_at is not None and time.perf_counter() >= stop_at:
                break

            loop_start = time.perf_counter()
            acquisition_start = loop_start
            frame = self.capture.grab_gpu_frame()
            acquisition_end = time.perf_counter()
            try:
                scale_start = acquisition_end
                output_width, output_height = self.presenter.output_size
                self.scaling_pass.execute(
                    frame,
                    render_target_view=self.presenter.render_target_view,
                    output_width=output_width,
                    output_height=output_height,
                )
                scale_end = time.perf_counter()
                if self.frame_pacer is not None:
                    pacing = self.frame_pacer.pace_batch(
                        [
                            PresentationFrame(
                                frame,
                                acquisition_start,
                                "real",
                            )
                        ]
                    )[0]
                    if not pacing.present:
                        continue
                present_start = time.perf_counter()
                self.presenter.present()
                present_end = time.perf_counter()
            finally:
                frame.close()

            if present_end >= collect_after:
                metrics.record(
                    acquisition_ms=(acquisition_end - acquisition_start) * 1000.0,
                    scale_submit_ms=(scale_end - scale_start) * 1000.0,
                    present_submit_ms=(present_end - present_start) * 1000.0,
                    cpu_loop_ms=(present_end - loop_start) * 1000.0,
                    source_size=(frame.width, frame.height),
                    output_size=self.presenter.output_size,
                )

            now = time.perf_counter()
            if now >= next_log and metrics.cpu_loop_ms:
                report = metrics.report(
                    adapter_description=self.adapter_description,
                    replaced_frames=self.capture.gpu_replaced_frames,
                    ended_at=now,
                )
                logger.info(
                    "D3D11 GPU | WGC %sx%s -> %sx%s | %s | %.1f FPS | "
                    "acquire %.2f ms | scale submit %.2f ms | present call %.2f ms | "
                    "loop %.2f ms | replaced %s",
                    report.source_width,
                    report.source_height,
                    report.output_width,
                    report.output_height,
                    self.method,
                    report.presented_fps,
                    report.acquisition.average_ms,
                    report.scale_submit.average_ms,
                    report.present_submit.average_ms,
                    report.cpu_loop.average_ms,
                    report.replaced_frames,
                )
                next_log = now + 1.0

        return metrics.report(
            adapter_description=self.adapter_description,
            replaced_frames=self.capture.gpu_replaced_frames,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._validation_resource_lease.release()
        # Release consumers before WGC releases the shared device/context.
        if self.presenter is not None:
            self.presenter.close()
        if self.scaling_pass is not None:
            self.scaling_pass.close()
        if self.capture is not None:
            self.capture.close()
        logger.info("D3D11 GPU pipeline closed cleanly.")

    def __enter__(self) -> "D3D11GpuPipeline":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()
