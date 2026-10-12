"""Flip-model D3D11 presentation for CPU-produced BGR frames."""

from __future__ import annotations

import ctypes
from ctypes import wintypes

import cv2
import numpy as np

from d3d11_gpu_pipeline import (
    D3D11PresentationError,
    D3D11SwapChainPresenter,
)
from wgc_capture import _check_hresult, _release, _vtable_function


_D3D11_CREATE_DEVICE_BGRA_SUPPORT = 0x20
_D3D_DRIVER_TYPE_HARDWARE = 1
_D3D_DRIVER_TYPE_WARP = 5


class D3D11CpuFramePresenter:
    """Upload the newest CPU frame into a vsynced flip-discard swap chain."""

    def __init__(self, width: int, height: int, *, vsync: bool = True) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Presenter dimensions must be positive.")
        self._device = ctypes.c_void_p()
        self._context = ctypes.c_void_p()
        self._bgra: np.ndarray | None = None
        self._resized: np.ndarray | None = None
        self._closed = False
        self._create_device()
        try:
            self.presenter = D3D11SwapChainPresenter(
                self._device,
                self._context,
                width=width,
                height=height,
                vsync=vsync,
            )
        except Exception:
            self.close()
            raise

    def _create_device(self) -> None:
        d3d11 = ctypes.WinDLL("d3d11", use_last_error=True)
        create = d3d11.D3D11CreateDevice
        create.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.UINT,
            ctypes.c_void_p,
            wintypes.UINT,
            wintypes.UINT,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(wintypes.UINT),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        create.restype = ctypes.c_long
        feature_level = wintypes.UINT()
        result = -1
        for driver_type in (_D3D_DRIVER_TYPE_HARDWARE, _D3D_DRIVER_TYPE_WARP):
            result = create(
                None,
                driver_type,
                None,
                _D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                None,
                0,
                7,
                ctypes.byref(self._device),
                ctypes.byref(feature_level),
                ctypes.byref(self._context),
            )
            if int(result) >= 0:
                break
        _check_hresult(result, "D3D11CreateDevice(CPU presenter)")

    @property
    def hwnd(self) -> int:
        return self.presenter.hwnd

    @property
    def quit_requested(self) -> bool:
        return self.presenter.quit_requested

    def present_bgr(self, frame: np.ndarray) -> None:
        if self._closed:
            raise D3D11PresentationError("CPU-frame presenter is closed.")
        if frame.ndim != 3 or frame.shape[2] < 3 or frame.dtype != np.uint8:
            raise ValueError("Presenter requires a uint8 BGR image.")
        self.presenter.pump_messages()
        self.presenter.apply_pending_resize()
        if self.presenter.quit_requested:
            return
        width, height = self.presenter.output_size
        source = frame
        if source.shape[:2] != (height, width):
            if self._resized is None or self._resized.shape != (height, width, 3):
                self._resized = np.empty((height, width, 3), dtype=np.uint8)
            source = cv2.resize(
                source,
                (width, height),
                dst=self._resized,
                interpolation=cv2.INTER_LINEAR,
            )
        if self._bgra is None or self._bgra.shape != (height, width, 4):
            self._bgra = np.empty((height, width, 4), dtype=np.uint8)
        cv2.cvtColor(source, cv2.COLOR_BGR2BGRA, dst=self._bgra)
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
            self.presenter.back_buffer,
            0,
            None,
            self._bgra.ctypes.data_as(ctypes.c_void_p),
            width * 4,
            0,
        )
        self.presenter.present()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        presenter = getattr(self, "presenter", None)
        if presenter is not None:
            presenter.close()
        _release(self._context)
        _release(self._device)
        self._bgra = None
        self._resized = None


__all__ = ["D3D11CpuFramePresenter"]
