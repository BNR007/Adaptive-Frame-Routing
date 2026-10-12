"""Compression-tolerant frame identifiers for compositor recordings.

The marker is intentionally drawn after quality analysis.  It measures which
URFTS submissions reached an OBS/window recording; it is not a claim about
physical monitor scanout.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


SYNC_BITS = (1, 0, 1, 1)
DATA_BITS = 16
CHECK_BITS = 4
CELL_COUNT = len(SYNC_BITS) + DATA_BITS + CHECK_BITS


@dataclass(frozen=True, slots=True)
class MarkerGeometry:
    x: int
    y: int
    cell_size: int
    width: int
    height: int


def _checksum(value: int) -> int:
    return (
        (value & 0xF)
        ^ ((value >> 4) & 0xF)
        ^ ((value >> 8) & 0xF)
        ^ ((value >> 12) & 0xF)
    ) & 0xF


def marker_bits(frame_id: int) -> tuple[int, ...]:
    """Return sync, 16-bit Gray payload, and checksum bits, MSB first."""

    value = int(frame_id) & 0xFFFF
    gray = value ^ (value >> 1)
    payload = tuple((gray >> shift) & 1 for shift in range(15, -1, -1))
    check = _checksum(value)
    check_bits = tuple((check >> shift) & 1 for shift in range(3, -1, -1))
    return SYNC_BITS + payload + check_bits


def marker_geometry(frame_shape: tuple[int, ...], *, margin: int = 8) -> MarkerGeometry:
    height, width = frame_shape[:2]
    cell = max(5, min(12, width // 150))
    marker_width = CELL_COUNT * cell
    marker_height = 2 * cell
    x = max(0, width - marker_width - margin)
    y = max(0, min(margin, height - marker_height))
    return MarkerGeometry(x, y, cell, marker_width, marker_height)


def draw_visual_frame_marker(frame: np.ndarray, frame_id: int) -> MarkerGeometry:
    """Draw a two-row marker; the second row is the inverse for validation."""

    if frame.ndim != 3 or frame.shape[2] < 3 or frame.dtype != np.uint8:
        raise ValueError("Visual frame markers require a uint8 BGR image.")
    geometry = marker_geometry(frame.shape)
    bits = marker_bits(frame_id)
    for column, bit in enumerate(bits):
        x0 = geometry.x + column * geometry.cell_size
        x1 = x0 + geometry.cell_size
        for row, row_bit in enumerate((bit, 1 - bit)):
            y0 = geometry.y + row * geometry.cell_size
            y1 = y0 + geometry.cell_size
            value = 255 if row_bit else 0
            frame[y0:y1, x0:x1, :3] = value
    cv2.rectangle(
        frame,
        (geometry.x - 1, geometry.y - 1),
        (geometry.x + geometry.width, geometry.y + geometry.height),
        (127, 127, 127),
        1,
    )
    return geometry


def _gray_to_binary(gray: int) -> int:
    value = gray
    shifted = gray >> 1
    while shifted:
        value ^= shifted
        shifted >>= 1
    return value & 0xFFFF


def decode_visual_frame_marker(frame: np.ndarray) -> int | None:
    """Decode a marker at its deterministic top-right location."""

    if frame.ndim != 3 or frame.shape[2] < 3:
        return None
    geometry = marker_geometry(frame.shape)
    top: list[float] = []
    bottom: list[float] = []
    inset = max(1, geometry.cell_size // 4)
    for column in range(CELL_COUNT):
        x0 = geometry.x + column * geometry.cell_size + inset
        x1 = geometry.x + (column + 1) * geometry.cell_size - inset
        for row, values in ((0, top), (1, bottom)):
            y0 = geometry.y + row * geometry.cell_size + inset
            y1 = geometry.y + (row + 1) * geometry.cell_size - inset
            values.append(float(np.mean(frame[y0:y1, x0:x1, :3])))
    low = min(top[: len(SYNC_BITS)])
    high = max(top[: len(SYNC_BITS)])
    if high - low < 80.0:
        return None
    threshold = (low + high) * 0.5
    bits = tuple(int(value >= threshold) for value in top)
    inverse = tuple(int(value >= threshold) for value in bottom)
    if bits[: len(SYNC_BITS)] != SYNC_BITS:
        return None
    if any(a == b for a, b in zip(bits, inverse, strict=True)):
        return None
    offset = len(SYNC_BITS)
    gray = 0
    for bit in bits[offset : offset + DATA_BITS]:
        gray = (gray << 1) | bit
    value = _gray_to_binary(gray)
    observed_check = 0
    for bit in bits[offset + DATA_BITS :]:
        observed_check = (observed_check << 1) | bit
    return value if observed_check == _checksum(value) else None


__all__ = [
    "MarkerGeometry",
    "decode_visual_frame_marker",
    "draw_visual_frame_marker",
    "marker_bits",
    "marker_geometry",
]
