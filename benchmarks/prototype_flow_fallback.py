"""Compare inexpensive optical-flow fallbacks on one withheld video frame."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_regional_video import psnr, ssim


def warp_pair(first: np.ndarray, last: np.ndarray, width: int) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    height = max(1, round(first.shape[0] * width / first.shape[1]))
    size = (width, height)
    low_a = cv2.resize(first, size, interpolation=cv2.INTER_LINEAR)
    low_b = cv2.resize(last, size, interpolation=cv2.INTER_LINEAR)
    gray_a = cv2.cvtColor(low_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(low_b, cv2.COLOR_BGR2GRAY)
    optical_flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    flow_ab = optical_flow.calc(gray_a, gray_b, None)
    flow_ba = optical_flow.calc(gray_b, gray_a, None)
    rows, columns = np.indices((height, width), dtype=np.float32)
    map_a = np.empty_like(flow_ab)
    map_b = np.empty_like(flow_ba)
    map_a[..., 0] = columns - flow_ab[..., 0] * 0.5
    map_a[..., 1] = rows - flow_ab[..., 1] * 0.5
    map_b[..., 0] = columns - flow_ba[..., 0] * 0.5
    map_b[..., 1] = rows - flow_ba[..., 1] * 0.5
    warped_a = cv2.remap(low_a, map_a, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    warped_b = cv2.remap(low_b, map_b, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    low_result = cv2.addWeighted(warped_a, 0.5, warped_b, 0.5, 0.0)
    result = cv2.resize(low_result, first.shape[1::-1], interpolation=cv2.INTER_LINEAR)
    return result, (time.perf_counter() - started) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--target-frame", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    capture = cv2.VideoCapture(str(args.video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.target_frame - 1)
    frames = []
    for _ in range(3):
        ok, frame = capture.read()
        if not ok:
            raise SystemExit("Unable to read requested triplet")
        frames.append(frame)
    capture.release()
    first, target, last = frames
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    midpoint = cv2.addWeighted(first, 0.5, last, 0.5, 0.0)
    print(f"midpoint: PSNR {psnr(midpoint, target):.3f}, SSIM {ssim(midpoint, target):.5f}")
    cv2.imwrite(str(output / "target.png"), target)
    cv2.imwrite(str(output / "midpoint.png"), midpoint)
    for width in (160, 240, 320, 480):
        candidate, elapsed = warp_pair(first, last, width)
        print(
            f"DIS {width}: {elapsed:.2f} ms, PSNR {psnr(candidate, target):.3f}, "
            f"SSIM {ssim(candidate, target):.5f}"
        )
        cv2.imwrite(str(output / f"dis_{width}.png"), candidate)


if __name__ == "__main__":
    main()
