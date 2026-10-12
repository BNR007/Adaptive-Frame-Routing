"""Measure the integrated regional interpolator on withheld video frames.

The first and third frames of each triplet simulate a 30 FPS source sampled
from a 60 FPS recording. The middle frame is retained as ground truth. This
measures processing throughput and image stability; it is not a display
scanout or input-to-photon measurement.
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
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from frame_interpolator import RIFEInterpolator
from evaluate_regional_video import psnr, ssim, temporal_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--provider", choices=("cpu", "directml"), default="directml")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--inference-width", type=int, default=160)
    parser.add_argument("--inference-height", type=int, default=90)
    parser.add_argument("--source-width", type=int, default=0)
    parser.add_argument("--source-height", type=int, default=0)
    parser.add_argument("--budget-ms", type=float, default=16.7)
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "output" / "live_regional_evaluation",
    )
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return float(ordered[round(fraction * (len(ordered) - 1))])


def main() -> None:
    args = parse_args()
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Unable to open video: {args.video}")
    recorded_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    recorded_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
    previous_output: np.ndarray | None = None
    previous_midpoint: np.ndarray | None = None
    previous_target: np.ndarray | None = None
    pair_index = 0
    frame_index = max(0, args.start_frame)
    try:
        ok, first = capture.read()
        while ok and len(rows) < args.samples:
            ok_target, target = capture.read()
            ok_last, last = capture.read()
            if not ok_target or not ok_last:
                break
            if args.source_width > 0 and args.source_height > 0:
                size = (args.source_width, args.source_height)
                first = cv2.resize(first, size, interpolation=cv2.INTER_AREA)
                target = cv2.resize(target, size, interpolation=cv2.INTER_AREA)
                last = cv2.resize(last, size, interpolation=cv2.INTER_AREA)
            if pair_index % args.stride == 0:
                midpoint = cv2.addWeighted(first, 0.5, last, 0.5, 0.0)
                started = time.perf_counter()
                candidate = interpolator.interpolate(first, last)
                wall_ms = (time.perf_counter() - started) * 1000.0
                regional = interpolator.last_regional_result
                rows.append(
                    {
                        "frame": float(frame_index + 1),
                        "processing_ms": wall_ms,
                        "psnr": psnr(candidate, target),
                        "ssim": ssim(candidate, target),
                        "temporal_error": temporal_error(
                            previous_output, candidate, previous_target, target
                        ),
                        "midpoint_psnr": psnr(midpoint, target),
                        "midpoint_ssim": ssim(midpoint, target),
                        "midpoint_temporal_error": temporal_error(
                            previous_midpoint, midpoint, previous_target, target
                        ),
                        "model_used": float((interpolator.last_inference_ms or 0.0) > 0.0),
                        "inference_ms": float(interpolator.last_inference_ms or 0.0),
                        "preprocessing_ms": float(interpolator.last_preprocessing_ms or 0.0),
                        "postprocessing_ms": float(interpolator.last_postprocessing_ms or 0.0),
                        "analysis_ms": float(0.0 if regional is None else regional.analysis_ms),
                        "composition_ms": float(0.0 if regional is None else regional.composition_ms),
                        "confidence": float(interpolator.last_interpolation_confidence),
                        "smooth_tiles": float(
                            0 if regional is None else regional.route_counts.get("smooth_motion", 0)
                        ),
                        "deadline_fallback_tiles": float(
                            0 if regional is None else regional.provenance_counts.get("deadline_fallback", 0)
                        ),
                        "admission_fallback_tiles": float(
                            0 if regional is None else regional.provenance_counts.get("admission_fallback", 0)
                        ),
                        "admission_reason": (
                            "none" if regional is None else regional.admission_reason
                        ),
                    }
                )
                previous_output = candidate.copy()
                previous_midpoint = midpoint.copy()
                previous_target = target.copy()
                if len(rows) in (1, 30, 60, 120):
                    sample = output / f"sample_{len(rows):03d}.png"
                    cv2.imwrite(str(sample), candidate)
                    cv2.imwrite(str(output / f"target_{len(rows):03d}.png"), target)
                    cv2.imwrite(str(output / f"midpoint_{len(rows):03d}.png"), midpoint)
                if len(rows) % 20 == 0:
                    print(f"sample {len(rows)}/{args.samples}", flush=True)
            first = last
            pair_index += 1
            frame_index += 2
    finally:
        interpolator.shutdown()
        capture.release()
    if not rows:
        raise SystemExit("Video did not contain enough frames.")
    csv_path = output / "per_frame_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    times = [row["processing_ms"] for row in rows]
    warm_times = times[1:] if len(times) > 1 else times
    report = {
        "video": str(args.video.resolve()),
        "samples": len(rows),
        "budget_ms": args.budget_ms,
        "source_width": int(args.source_width or recorded_width),
        "source_height": int(args.source_height or recorded_height),
        "mean_processing_ms": statistics.mean(warm_times),
        "median_processing_ms": statistics.median(warm_times),
        "p95_processing_ms": percentile(warm_times, 0.95),
        "estimated_processing_fps": 1000.0 / statistics.mean(warm_times),
        "mean_preprocessing_ms": statistics.mean(row["preprocessing_ms"] for row in rows[1:]),
        "mean_inference_ms": statistics.mean(row["inference_ms"] for row in rows[1:]),
        "mean_postprocessing_ms": statistics.mean(row["postprocessing_ms"] for row in rows[1:]),
        "mean_analysis_ms": statistics.mean(row["analysis_ms"] for row in rows[1:]),
        "mean_composition_ms": statistics.mean(row["composition_ms"] for row in rows[1:]),
        "model_use_fraction": statistics.mean(row["model_used"] for row in rows),
        "mean_psnr": statistics.mean(row["psnr"] for row in rows),
        "mean_ssim": statistics.mean(row["ssim"] for row in rows),
        "mean_temporal_error": statistics.mean(row["temporal_error"] for row in rows),
        "midpoint_mean_psnr": statistics.mean(row["midpoint_psnr"] for row in rows),
        "midpoint_mean_ssim": statistics.mean(row["midpoint_ssim"] for row in rows),
        "midpoint_mean_temporal_error": statistics.mean(
            row["midpoint_temporal_error"] for row in rows
        ),
        "interpretation": (
            "Integrated transform throughput and withheld-frame quality; not "
            "display scanout FPS or input-to-photon latency."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
