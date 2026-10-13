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

from async_pipeline import AsyncFramePipeline
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
        "--force-neural-refinement",
        action="store_true",
        help="Use the non-blocking worker's stabilized neural candidate path.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "output" / "live_regional_evaluation",
    )
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return float(ordered[round(fraction * (len(ordered) - 1))])


def phase_step_jerk(
    first: np.ndarray,
    middle: np.ndarray,
    last: np.ndarray,
) -> float:
    """Return disagreement between the two global half-step motion vectors."""

    size = (160, 90)
    window = cv2.createHanningWindow(size, cv2.CV_32F)

    def gray(image: np.ndarray) -> np.ndarray:
        return cv2.resize(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            size,
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)

    first_gray, middle_gray, last_gray = map(gray, (first, middle, last))
    first_step, _ = cv2.phaseCorrelate(first_gray, middle_gray, window)
    second_step, _ = cv2.phaseCorrelate(middle_gray, last_gray, window)
    return float(np.hypot(
        second_step[0] - first_step[0],
        second_step[1] - first_step[1],
    ))


def global_motion_midpoint(first: np.ndarray, last: np.ndarray) -> np.ndarray:
    """Half-warp coherent camera motion while retaining stable HUD pixels."""

    height, width = first.shape[:2]
    size = (160, 90)
    window = cv2.createHanningWindow(size, cv2.CV_32F)
    gray_first = cv2.resize(
        cv2.cvtColor(first, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA
    ).astype(np.float32)
    gray_last = cv2.resize(
        cv2.cvtColor(last, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA
    ).astype(np.float32)
    (shift_x, shift_y), _ = cv2.phaseCorrelate(gray_first, gray_last, window)
    shift_x *= width / size[0]
    shift_y *= height / size[1]
    first_half = cv2.warpAffine(
        first,
        np.float32(((1.0, 0.0, shift_x * 0.5), (0.0, 1.0, shift_y * 0.5))),
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    last_half = cv2.warpAffine(
        last,
        np.float32(((1.0, 0.0, -shift_x * 0.5), (0.0, 1.0, -shift_y * 0.5))),
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    candidate = cv2.addWeighted(first_half, 0.5, last_half, 0.5, 0.0)
    stable = cv2.addWeighted(first, 0.5, last, 0.5, 0.0)
    stable_mask = np.max(cv2.absdiff(first, last), axis=2) <= 8
    np.copyto(candidate, stable, where=stable_mask[..., None])
    return candidate


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
                camera_motion = AsyncFramePipeline._camera_motion_estimate(first, last)
                motion_midpoint = (
                    global_motion_midpoint(first, last)
                    if camera_motion is not None
                    else midpoint
                )
                started = time.perf_counter()
                if args.force_neural_refinement:
                    saved_regional = interpolator.regional_transform
                    saved_stabilization = interpolator.temporal_stabilization
                    try:
                        interpolator.regional_transform = False
                        interpolator.temporal_stabilization = True
                        candidate = interpolator.interpolate(first, last)
                    finally:
                        interpolator.regional_transform = saved_regional
                        interpolator.temporal_stabilization = saved_stabilization
                else:
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
                        "motion_midpoint_psnr": psnr(motion_midpoint, target),
                        "motion_midpoint_ssim": ssim(motion_midpoint, target),
                        "midpoint_temporal_error": temporal_error(
                            previous_midpoint, midpoint, previous_target, target
                        ),
                        "global_motion_pixels": float(
                            0.0 if camera_motion is None else camera_motion[0]
                        ),
                        "global_motion_response": float(
                            0.0 if camera_motion is None else camera_motion[1]
                        ),
                        "candidate_phase_jerk": phase_step_jerk(
                            first, candidate, last
                        ),
                        "target_phase_jerk": phase_step_jerk(first, target, last),
                        "midpoint_phase_jerk": phase_step_jerk(
                            first, midpoint, last
                        ),
                        "motion_midpoint_phase_jerk": phase_step_jerk(
                            first, motion_midpoint, last
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
                    cv2.imwrite(
                        str(output / f"motion_midpoint_{len(rows):03d}.png"),
                        motion_midpoint,
                    )
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
    warm_rows = rows[1:] if len(rows) > 1 else rows
    camera_rows = [row for row in rows if row["global_motion_pixels"] > 0.0]
    candidate_excess_jerk = [
        abs(row["candidate_phase_jerk"] - row["target_phase_jerk"])
        for row in camera_rows
    ]
    midpoint_excess_jerk = [
        abs(row["midpoint_phase_jerk"] - row["target_phase_jerk"])
        for row in camera_rows
    ]
    motion_midpoint_excess_jerk = [
        abs(row["motion_midpoint_phase_jerk"] - row["target_phase_jerk"])
        for row in camera_rows
    ]
    report = {
        "video": str(args.video.resolve()),
        "samples": len(rows),
        "budget_ms": args.budget_ms,
        "force_neural_refinement": bool(args.force_neural_refinement),
        "source_width": int(args.source_width or recorded_width),
        "source_height": int(args.source_height or recorded_height),
        "mean_processing_ms": statistics.mean(warm_times),
        "median_processing_ms": statistics.median(warm_times),
        "p95_processing_ms": percentile(warm_times, 0.95),
        "estimated_processing_fps": 1000.0 / statistics.mean(warm_times),
        "mean_preprocessing_ms": statistics.mean(row["preprocessing_ms"] for row in warm_rows),
        "mean_inference_ms": statistics.mean(row["inference_ms"] for row in warm_rows),
        "mean_postprocessing_ms": statistics.mean(row["postprocessing_ms"] for row in warm_rows),
        "mean_analysis_ms": statistics.mean(row["analysis_ms"] for row in warm_rows),
        "mean_composition_ms": statistics.mean(row["composition_ms"] for row in warm_rows),
        "model_use_fraction": statistics.mean(row["model_used"] for row in rows),
        "mean_psnr": statistics.mean(row["psnr"] for row in rows),
        "mean_ssim": statistics.mean(row["ssim"] for row in rows),
        "mean_temporal_error": statistics.mean(row["temporal_error"] for row in rows),
        "midpoint_mean_psnr": statistics.mean(row["midpoint_psnr"] for row in rows),
        "midpoint_mean_ssim": statistics.mean(row["midpoint_ssim"] for row in rows),
        "midpoint_mean_temporal_error": statistics.mean(
            row["midpoint_temporal_error"] for row in rows
        ),
        "detected_camera_turn_samples": len(camera_rows),
        "camera_turn_mean_psnr": (
            statistics.mean(row["psnr"] for row in camera_rows)
            if camera_rows else None
        ),
        "camera_turn_midpoint_mean_psnr": (
            statistics.mean(row["midpoint_psnr"] for row in camera_rows)
            if camera_rows else None
        ),
        "camera_turn_motion_midpoint_mean_psnr": (
            statistics.mean(row["motion_midpoint_psnr"] for row in camera_rows)
            if camera_rows else None
        ),
        "camera_turn_motion_midpoint_mean_ssim": (
            statistics.mean(row["motion_midpoint_ssim"] for row in camera_rows)
            if camera_rows else None
        ),
        "camera_turn_candidate_mean_excess_phase_jerk": (
            statistics.mean(candidate_excess_jerk)
            if candidate_excess_jerk else None
        ),
        "camera_turn_midpoint_mean_excess_phase_jerk": (
            statistics.mean(midpoint_excess_jerk)
            if midpoint_excess_jerk else None
        ),
        "camera_turn_motion_midpoint_mean_excess_phase_jerk": (
            statistics.mean(motion_midpoint_excess_jerk)
            if motion_midpoint_excess_jerk else None
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
