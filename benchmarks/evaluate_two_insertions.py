"""Evaluate two evenly spaced synthesized frames on withheld video frames.

The benchmark uses every third frame as source endpoints and the two intervening
frames as ground truth. It reports offline transform latency and exact image
uniqueness; it does not claim display scanout or input-to-photon latency.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluate_regional_video import psnr, ssim, temporal_error
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
    parser.add_argument("--budget-ms", type=float, default=16.0)
    parser.add_argument("--samples", type=int, default=240)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "output" / "two_insertions_evaluation",
    )
    return parser.parse_args()


def digest(image: np.ndarray) -> bytes:
    return hashlib.blake2b(memoryview(np.ascontiguousarray(image)), digest_size=16).digest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return float(ordered[round(fraction * (len(ordered) - 1))])


def read(capture: cv2.VideoCapture) -> np.ndarray | None:
    ok, frame = capture.read()
    return frame if ok else None


def main() -> None:
    args = parse_args()
    if not args.video.is_file():
        raise SystemExit(f"Video does not exist: {args.video}")
    if not args.model.is_file():
        raise SystemExit(f"Model does not exist: {args.model}")
    if min(args.inference_width, args.inference_height, args.samples) <= 0:
        raise SystemExit("Resolution and sample count must be positive.")

    output_dir = args.output_directory.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Unable to open video: {args.video}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, args.start_frame))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

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

    rows: list[dict[str, float | str]] = []
    generated_ms: list[float] = []
    emitted_digests: set[bytes] = set()
    unique_outputs = 0
    output_frame_count = 0
    previous_candidates: list[np.ndarray | None] = [None, None]
    previous_targets: list[np.ndarray | None] = [None, None]
    first = read(capture)
    frame_index = max(0, args.start_frame)
    wall_started = time.perf_counter()

    def count_output(image: np.ndarray) -> None:
        nonlocal unique_outputs, output_frame_count
        value = digest(image)
        output_frame_count += 1
        if value not in emitted_digests:
            emitted_digests.add(value)
            unique_outputs += 1

    try:
        while first is not None and len(rows) < args.samples:
            target_one = read(capture)
            target_two = read(capture)
            last = read(capture)
            if target_one is None or target_two is None or last is None:
                break

            pair_started = time.perf_counter()
            started_one = time.perf_counter()
            generated_one = interpolator.interpolate_at(first, last, 1.0 / 3.0)
            time_one = (time.perf_counter() - started_one) * 1000.0
            flow_one = interpolator.last_flow_ms
            regional_one = interpolator.last_regional_result
            method_one = interpolator.last_generation_method

            started_two = time.perf_counter()
            generated_two = interpolator.interpolate_at(first, last, 2.0 / 3.0)
            time_two = (time.perf_counter() - started_two) * 1000.0
            flow_two = interpolator.last_flow_ms
            regional_two = interpolator.last_regional_result
            method_two = interpolator.last_generation_method
            pair_ms = (time.perf_counter() - pair_started) * 1000.0
            generated_ms.extend((time_one, time_two))

            # The first real endpoint is counted once; adjacent pairs share it.
            if not rows:
                count_output(first)
            for image in (generated_one, generated_two, last):
                count_output(image)

            targets = (target_one, target_two)
            candidates = (generated_one, generated_two)
            blend_one = cv2.addWeighted(first, 2.0 / 3.0, last, 1.0 / 3.0, 0.0)
            blend_two = cv2.addWeighted(first, 1.0 / 3.0, last, 2.0 / 3.0, 0.0)
            row: dict[str, float | str] = {
                "source_frame": float(frame_index),
                "pair_processing_ms": pair_ms,
                "first_generated_ms": time_one,
                "second_generated_ms": time_two,
                "first_flow_ms": flow_one,
                "second_flow_ms": flow_two,
                "first_method": method_one,
                "second_method": method_two,
                "first_analysis_ms": float("nan") if regional_one is None else regional_one.analysis_ms,
                "second_analysis_ms": float("nan") if regional_two is None else regional_two.analysis_ms,
                "first_admission_reason": "none" if regional_one is None else regional_one.admission_reason,
                "second_admission_reason": "none" if regional_two is None else regional_two.admission_reason,
                "first_confidence": 0.0 if regional_one is None else float(np.mean(regional_one.confidence)),
                "second_confidence": 0.0 if regional_two is None else float(np.mean(regional_two.confidence)),
                "first_psnr": psnr(generated_one, target_one),
                "second_psnr": psnr(generated_two, target_two),
                "first_ssim": ssim(generated_one, target_one),
                "second_ssim": ssim(generated_two, target_two),
                "first_blend_psnr": psnr(blend_one, target_one),
                "second_blend_psnr": psnr(blend_two, target_two),
                "first_blend_ssim": ssim(blend_one, target_one),
                "second_blend_ssim": ssim(blend_two, target_two),
                "first_temporal_error": temporal_error(
                    previous_candidates[0], generated_one, previous_targets[0], target_one
                ),
                "second_temporal_error": temporal_error(
                    previous_candidates[1], generated_two, previous_targets[1], target_two
                ),
                "pair_unique_generated": float(digest(generated_one) != digest(generated_two)),
                "first_route_tiles": float(
                    0 if regional_one is None else sum(regional_one.route_counts.values())
                ),
                "second_route_tiles": float(
                    0 if regional_two is None else sum(regional_two.route_counts.values())
                ),
                "first_interpolated_tiles": float(
                    0 if regional_one is None else regional_one.provenance_counts.get("interpolated", 0)
                ),
                "second_interpolated_tiles": float(
                    0 if regional_two is None else regional_two.provenance_counts.get("interpolated", 0)
                ),
            }
            rows.append(row)
            for index, (candidate, target) in enumerate(zip(candidates, targets), start=1):
                previous_candidates[index - 1] = candidate.copy()
                previous_targets[index - 1] = target.copy()
            if len(rows) in (1, 30, 60, 120, 240):
                cv2.imwrite(str(output_dir / f"generated_1of3_{len(rows):03d}.png"), generated_one)
                cv2.imwrite(str(output_dir / f"target_1of3_{len(rows):03d}.png"), target_one)
                cv2.imwrite(str(output_dir / f"generated_2of3_{len(rows):03d}.png"), generated_two)
                cv2.imwrite(str(output_dir / f"target_2of3_{len(rows):03d}.png"), target_two)
            if len(rows) % 20 == 0:
                print(f"sample {len(rows)}/{args.samples}", flush=True)
            first = last
            frame_index += 3
    finally:
        elapsed_s = time.perf_counter() - wall_started
        interpolator.shutdown()
        capture.release()

    if not rows:
        raise SystemExit("Video did not contain a complete four-frame span.")

    csv_path = output_dir / "per_pair_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    warm_pair_ms = [float(row["pair_processing_ms"]) for row in rows[1:]] or [
        float(rows[0]["pair_processing_ms"])
    ]
    generated_warm_ms = generated_ms[2:] or generated_ms
    source_pair_fps = source_fps / 3.0 if source_fps > 0.0 else 0.0
    pair_deadline_ms = 1000.0 / source_pair_fps if source_pair_fps > 0.0 else 0.0
    report = {
        "video": args.video.name,
        "model_loaded_for_runtime_compatibility": args.model.name,
        "insertion_method": "OpenCV Farneback bidirectional flow + regional confidence routing",
        "rife_neural_inference_used_for_insertions": False,
        "fractional_positions": [1.0 / 3.0, 2.0 / 3.0],
        "video_fps": source_fps,
        "simulated_source_fps": source_fps / 3.0 if source_fps > 0.0 else 0.0,
        "source_frame_count": source_count,
        "sample_pairs": len(rows),
        "generated_frame_count": 2 * len(rows),
        "generated_unique_fraction_within_pairs": statistics.mean(
            float(row["pair_unique_generated"]) for row in rows
        ),
        "flow_generated_fraction": statistics.mean(
            float(row["first_method"] == "farneback_flow")
            for row in rows
        ) / 2.0 + statistics.mean(
            float(row["second_method"] == "farneback_flow")
            for row in rows
        ) / 2.0,
        "regional_temporal_fallback_fraction": statistics.mean(
            float(row["first_method"] == "regional_temporal_fallback")
            for row in rows
        ) / 2.0 + statistics.mean(
            float(row["second_method"] == "regional_temporal_fallback")
            for row in rows
        ) / 2.0,
        "flow_rejected_fraction": statistics.mean(
            float(row["first_method"] == "farneback_rejected")
            for row in rows
        ) / 2.0 + statistics.mean(
            float(row["second_method"] == "farneback_rejected")
            for row in rows
        ) / 2.0,
        "distinct_output_images": unique_outputs,
        "counted_output_images": output_frame_count,
        "distinct_output_fraction_exact_hash": unique_outputs / max(1, output_frame_count),
        "mean_generated_latency_ms": statistics.mean(generated_warm_ms),
        "median_generated_latency_ms": statistics.median(generated_warm_ms),
        "p95_generated_latency_ms": percentile(generated_warm_ms, 0.95),
        "mean_pair_latency_ms": statistics.mean(warm_pair_ms),
        "p95_pair_latency_ms": percentile(warm_pair_ms, 0.95),
        "offline_generated_throughput_fps": 2000.0 / statistics.mean(warm_pair_ms),
        "source_pair_cadence_fps": source_pair_fps,
        "pair_deadline_ms_for_source_cadence": pair_deadline_ms,
        "pair_deadline_hit_fraction": (
            sum(value <= pair_deadline_ms for value in warm_pair_ms) / len(warm_pair_ms)
            if pair_deadline_ms > 0.0
            else 0.0
        ),
        "offline_total_sequence_throughput_fps": 3000.0 / statistics.mean(warm_pair_ms),
        "elapsed_seconds": elapsed_s,
        "mean_psnr_1of3": statistics.mean(float(row["first_psnr"]) for row in rows),
        "mean_psnr_2of3": statistics.mean(float(row["second_psnr"]) for row in rows),
        "mean_ssim_1of3": statistics.mean(float(row["first_ssim"]) for row in rows),
        "mean_ssim_2of3": statistics.mean(float(row["second_ssim"]) for row in rows),
        "mean_blend_psnr_1of3": statistics.mean(float(row["first_blend_psnr"]) for row in rows),
        "mean_blend_psnr_2of3": statistics.mean(float(row["second_blend_psnr"]) for row in rows),
        "mean_blend_ssim_1of3": statistics.mean(float(row["first_blend_ssim"]) for row in rows),
        "mean_blend_ssim_2of3": statistics.mean(float(row["second_blend_ssim"]) for row in rows),
        "interpretation": (
            "Offline generated-image uniqueness, processing latency, and quality against "
            "withheld frames. FPS is transform throughput, not display presentation, "
            "scanout, or input-to-photon latency. Exact image hashes count any pixel "
            "difference as unique, including source noise."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
