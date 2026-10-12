"""Evaluate the cheap 0.75 playout reserve against withheld video frames.

Every fourth recorded frame is treated as a source endpoint. The frame at
three quarters of that interval is ground truth for the reserve candidate.
This is an offline quality test, not evidence of display cadence.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
import time

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluate_regional_video import psnr, ssim
from frame_interpolator import RIFEInterpolator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "RIFE_v4.25_lite.onnx",
    )
    parser.add_argument("--provider", choices=("cpu", "directml"), default="cpu")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--inference-width", type=int, default=160)
    parser.add_argument("--inference-height", type=int, default=90)
    parser.add_argument("--budget-ms", type=float, default=34.0)
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "output" / "buffered_reserve_quality",
    )
    return parser.parse_args()


def distribution(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "samples": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p05": ordered[round(0.05 * (len(ordered) - 1))],
        "p95": ordered[round(0.95 * (len(ordered) - 1))],
    }


def read(capture: cv2.VideoCapture):
    ok, frame = capture.read()
    return frame if ok else None


def main() -> None:
    args = parse_args()
    if not args.video.is_file():
        raise SystemExit(f"Video does not exist: {args.video}")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Unable to open video: {args.video}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, args.start_frame))
    interpolator = RIFEInterpolator(
        args.model,
        provider=args.provider,
        device_id=args.device_id,
        inference_width=args.inference_width,
        inference_height=args.inference_height,
        temporal_stabilization=False,
        regional_transform=True,
        regional_budget_ms=args.budget_ms,
    )
    interpolator.initialize()
    interpolator.warmup(3)
    rows: list[dict[str, float]] = []
    first = read(capture)
    try:
        while first is not None and len(rows) < args.samples:
            quarter = read(capture)
            midpoint_target = read(capture)
            reserve_target = read(capture)
            endpoint = read(capture)
            if any(item is None for item in (quarter, midpoint_target, reserve_target, endpoint)):
                break
            started = time.perf_counter()
            midpoint = interpolator.interpolate(first, endpoint)
            midpoint_ms = (time.perf_counter() - started) * 1000.0
            reserve = cv2.addWeighted(midpoint, 0.5, endpoint, 0.5, 0.0)
            direct_blend = cv2.addWeighted(first, 0.25, endpoint, 0.75, 0.0)
            rows.append(
                {
                    "midpoint_ms": midpoint_ms,
                    "reserve_psnr": psnr(reserve, reserve_target),
                    "reserve_ssim": ssim(reserve, reserve_target),
                    "repeat_midpoint_psnr": psnr(midpoint, reserve_target),
                    "repeat_midpoint_ssim": ssim(midpoint, reserve_target),
                    "repeat_endpoint_psnr": psnr(endpoint, reserve_target),
                    "repeat_endpoint_ssim": ssim(endpoint, reserve_target),
                    "direct_blend_psnr": psnr(direct_blend, reserve_target),
                    "direct_blend_ssim": ssim(direct_blend, reserve_target),
                    "midpoint_psnr": psnr(midpoint, midpoint_target),
                    "midpoint_ssim": ssim(midpoint, midpoint_target),
                }
            )
            if len(rows) in (1, 30, 60, 120):
                stem = f"sample_{len(rows):03d}"
                cv2.imwrite(str(args.output_directory / f"{stem}_reserve.png"), reserve)
                cv2.imwrite(str(args.output_directory / f"{stem}_target.png"), reserve_target)
            first = endpoint
    finally:
        interpolator.shutdown()
        capture.release()
    if not rows:
        raise SystemExit("Video did not contain a complete five-frame span.")
    with (args.output_directory / "per_sample.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "video": str(args.video.resolve()),
        "interpretation": "Offline withheld-frame quality; not live display evidence.",
        **{
            field: distribution([float(row[field]) for row in rows])
            for field in rows[0]
        },
        "paired_comparisons": {
            "reserve_psnr_better_than_repeat_midpoint_fraction": statistics.fmean(
                row["reserve_psnr"] > row["repeat_midpoint_psnr"] for row in rows
            ),
            "reserve_psnr_better_than_repeat_endpoint_fraction": statistics.fmean(
                row["reserve_psnr"] > row["repeat_endpoint_psnr"] for row in rows
            ),
            "reserve_minus_repeat_midpoint_psnr_db": statistics.fmean(
                row["reserve_psnr"] - row["repeat_midpoint_psnr"] for row in rows
            ),
            "reserve_minus_repeat_endpoint_psnr_db": statistics.fmean(
                row["reserve_psnr"] - row["repeat_endpoint_psnr"] for row in rows
            ),
        },
    }
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
