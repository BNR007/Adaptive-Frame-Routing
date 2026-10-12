from __future__ import annotations

import cv2
import numpy as np

from visual_frame_marker import decode_visual_frame_marker, draw_visual_frame_marker


def test_visual_marker_round_trips_and_survives_jpeg_compression() -> None:
    for frame_id in (0, 1, 2, 255, 256, 32767, 65535):
        frame = np.full((720, 1280, 3), 73, dtype=np.uint8)
        draw_visual_frame_marker(frame, frame_id)
        assert decode_visual_frame_marker(frame) == frame_id
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        assert ok
        compressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        assert decode_visual_frame_marker(compressed) == frame_id


def test_visual_marker_rejects_unmarked_frame() -> None:
    frame = np.full((720, 1280, 3), 73, dtype=np.uint8)
    assert decode_visual_frame_marker(frame) is None
