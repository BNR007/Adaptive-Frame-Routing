"""Evaluate regional interpolation against withheld real video frames.

Every odd frame is treated as ground truth.  The surrounding even frames are
given to the interpolation model, then the raw candidate, shipped heuristic,
and regional controller are compared against the withheld frame.  This is an
offline quality experiment; its processing rate is not a live presentation FPS
claim.
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
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from frame_interpolator import RIFEInterpolator
from regional_transform import RegionRoute, RegionalTransformController


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "IFRNet_S_Vimeo90K.onnx",
    )
    parser.add_argument("--provider", choices=("cpu", "directml"), default="directml")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--inference-width", type=_positive_int, default=240)
    parser.add_argument("--inference-height", type=_positive_int, default=135)
    parser.add_argument("--max-samples", type=_positive_int, default=120)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--sample-stride", type=_positive_int, default=1)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "output" / "regional_video_evaluation",
    )
    return parser.parse_args()


def psnr(candidate: np.ndarray, target: np.ndarray) -> float:
    error = np.mean((candidate.astype(np.float32) - target.astype(np.float32)) ** 2)
    if error <= 1e-12:
        return 99.0
    return float(10.0 * np.log10((255.0**2) / error))


def ssim(candidate: np.ndarray, target: np.ndarray) -> float:
    first = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY).astype(np.float32)
    second = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY).astype(np.float32)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mean_a = cv2.GaussianBlur(first, (11, 11), 1.5)
    mean_b = cv2.GaussianBlur(second, (11, 11), 1.5)
    variance_a = cv2.GaussianBlur(first * first, (11, 11), 1.5) - mean_a**2
    variance_b = cv2.GaussianBlur(second * second, (11, 11), 1.5) - mean_b**2
    covariance = cv2.GaussianBlur(first * second, (11, 11), 1.5) - mean_a * mean_b
    numerator = (2.0 * mean_a * mean_b + c1) * (2.0 * covariance + c2)
    denominator = (mean_a**2 + mean_b**2 + c1) * (
        variance_a + variance_b + c2
    )
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def temporal_error(
    previous: np.ndarray | None,
    current: np.ndarray,
    previous_target: np.ndarray | None,
    target: np.ndarray,
) -> float:
    if previous is None or previous_target is None:
        return 0.0
    candidate_change = cv2.absdiff(current, previous)
    target_change = cv2.absdiff(target, previous_target)
    return float(np.mean(cv2.absdiff(candidate_change, target_change)))


def _read_frame(capture: cv2.VideoCapture) -> np.ndarray | None:
    ok, frame = capture.read()
    return frame if ok else None


def _working_frames(
    raw: np.ndarray,
    first: np.ndarray,
    last: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = (width, height)
    return (
        cv2.resize(raw, size, interpolation=cv2.INTER_AREA),
        cv2.resize(first, size, interpolation=cv2.INTER_AREA),
        cv2.resize(last, size, interpolation=cv2.INTER_AREA),
    )


def _restore_full_regional(
    low_generated: np.ndarray,
    pixel_confidence: np.ndarray,
    endpoint_fallback: np.ndarray,
    first: np.ndarray,
    last: np.ndarray,
    *,
    scene_cut: bool,
) -> np.ndarray:
    if scene_cut:
        return last.copy()
    height, width = first.shape[:2]
    generated = cv2.resize(
        low_generated, (width, height), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
    confidence = cv2.resize(
        pixel_confidence, (width, height), interpolation=cv2.INTER_LINEAR
    )
    midpoint = cv2.addWeighted(first, 0.5, last, 0.5, 0.0).astype(np.float32)
    endpoint_weight = cv2.resize(
        endpoint_fallback, (width, height), interpolation=cv2.INTER_LINEAR
    )
    stable_fallback = (
        last.astype(np.float32) * endpoint_weight[..., None]
        + midpoint * (1.0 - endpoint_weight[..., None])
    )
    return np.rint(
        generated * confidence[..., None]
        + stable_fallback * (1.0 - confidence[..., None])
    ).astype(np.uint8)


def _summary(rows: list[dict[str, float]], prefix: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric in ("psnr", "ssim", "temporal_error", "processing_ms"):
        values = [row[f"{prefix}_{metric}"] for row in rows]
        result[f"mean_{metric}"] = float(statistics.mean(values))
        result[f"median_{metric}"] = float(statistics.median(values))
        ordered = sorted(values)
        result[f"p95_{metric}"] = float(ordered[round(0.95 * (len(ordered) - 1))])
    return result


def main() -> None:
    args = parse_args()
    if not args.video.is_file():
        raise SystemExit(f"Video does not exist: {args.video}")
    if not args.model.is_file():
        raise SystemExit(f"Model does not exist: {args.model}")
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Unable to open video: {args.video}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, args.start_frame))

    interpolator = RIFEInterpolator(
        args.model,
        provider=args.provider,
        device_id=args.device_id,
        inference_width=args.inference_width,
        inference_height=args.inference_height,
        temporal_stabilization=False,
    )
    interpolator.initialize()
    interpolator.warmup(3)
    regional = RegionalTransformController()
    rows: list[dict[str, float]] = []
    previous_outputs: dict[str, np.ndarray | None] = {
        "midpoint": None,
        "raw": None,
        "legacy": None,
        "regional": None,
    }
    previous_target: np.ndarray | None = None
    first = _read_frame(capture)
    source_index = max(0, args.start_frame)
    pair_index = 0
    try:
        while first is not None and len(rows) < args.max_samples:
            target = _read_frame(capture)
            last = _read_frame(capture)
            if target is None or last is None:
                break
            if pair_index % args.sample_stride:
                first = last
                source_index += 2
                pair_index += 1
                continue

            midpoint_started = time.perf_counter()
            midpoint = cv2.addWeighted(first, 0.5, last, 0.5, 0.0)
            midpoint_ms = (time.perf_counter() - midpoint_started) * 1000.0

            raw_started = time.perf_counter()
            raw = interpolator.interpolate(first, last)
            raw_ms = (time.perf_counter() - raw_started) * 1000.0

            low_raw, low_first, low_last = _working_frames(
                raw,
                first,
                last,
                args.inference_width,
                args.inference_height,
            )
            legacy_started = time.perf_counter()
            low_legacy, _fraction, legacy_mask = RIFEInterpolator._confidence_composite(
                low_raw.copy(), low_first, low_last
            )
            legacy = cv2.resize(
                low_legacy, first.shape[1::-1], interpolation=cv2.INTER_LINEAR
            )
            full_legacy_mask = cv2.resize(
                legacy_mask,
                first.shape[1::-1],
                interpolation=cv2.INTER_NEAREST,
            )
            full_midpoint = cv2.addWeighted(first, 0.5, last, 0.5, 0.0)
            cv2.copyTo(full_midpoint, full_legacy_mask, legacy)
            legacy_ms = raw_ms + (time.perf_counter() - legacy_started) * 1000.0

            regional_started = time.perf_counter()
            analysis = regional.analyze(low_first, low_last)
            regional_selected = regional.allocate(analysis, None)
            regional_selected &= analysis.routes == RegionRoute.SMOOTH_MOTION
            regional_model_used = bool(np.any(regional_selected))
            regional_candidate = (
                low_raw
                if regional_model_used
                else cv2.addWeighted(low_first, 0.5, low_last, 0.5, 0.0)
            )
            regional_result = regional.compose_interpolation(
                regional_candidate.copy(), low_first, low_last, analysis=analysis
            )
            regional_frame = _restore_full_regional(
                regional_candidate,
                regional_result.pixel_confidence,
                regional_result.endpoint_fallback,
                first,
                last,
                scene_cut=analysis.scene_cut,
            )
            regional_ms = (
                raw_ms if regional_model_used else 0.0
            ) + (time.perf_counter() - regional_started) * 1000.0
            regional.observe_real_frame(
                cv2.resize(
                    target,
                    (args.inference_width, args.inference_height),
                    interpolation=cv2.INTER_AREA,
                )
            )

            candidates = {
                "midpoint": (midpoint, midpoint_ms),
                "raw": (raw, raw_ms),
                "legacy": (legacy, legacy_ms),
                "regional": (regional_frame, regional_ms),
            }
            row: dict[str, float] = {"frame": float(source_index + 1)}
            for name, (candidate, processing_ms) in candidates.items():
                row[f"{name}_psnr"] = psnr(candidate, target)
                row[f"{name}_ssim"] = ssim(candidate, target)
                row[f"{name}_temporal_error"] = temporal_error(
                    previous_outputs[name], candidate, previous_target, target
                )
                row[f"{name}_processing_ms"] = processing_ms
                previous_outputs[name] = candidate.copy()
            row["regional_interpolated_tiles"] = float(
                regional_result.provenance_counts.get("interpolated", 0)
            )
            row["regional_model_used"] = float(regional_model_used)
            row["regional_blended_tiles"] = float(
                regional_result.provenance_counts.get("blended", 0)
            )
            row["regional_fallback_tiles"] = float(
                regional_result.provenance_counts.get("spatial", 0)
                + regional_result.provenance_counts.get("midpoint_blend", 0)
                + regional_result.provenance_counts.get("deadline_fallback", 0)
                + regional_result.provenance_counts.get("real", 0)
            )
            for route in RegionRoute:
                key = route.name.lower()
                row[f"regional_route_{key}"] = float(
                    regional_result.route_counts.get(key, 0)
                )
            rows.append(row)
            print(
                f"sample {len(rows)}/{args.max_samples}: frame {source_index + 1}, "
                f"routes={regional_result.route_counts}, "
                f"provenance={regional_result.provenance_counts}",
                flush=True,
            )

            if len(rows) in (1, 10, 30):
                sample_dir = output / f"sample_{len(rows):03d}"
                sample_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(sample_dir / "ground_truth.png"), target)
                for name, (candidate, _processing_ms) in candidates.items():
                    cv2.imwrite(str(sample_dir / f"{name}.png"), candidate)
                overlay = RegionalTransformController.provenance_overlay(
                    regional_frame, regional_result.provenance
                )
                cv2.imwrite(str(sample_dir / "regional_provenance.png"), overlay)

            previous_target = target.copy()
            first = last
            source_index += 2
            pair_index += 1
    finally:
        capture.release()
        interpolator.shutdown()

    if not rows:
        raise SystemExit("Video did not contain enough frames for evaluation.")
    csv_path = output / "per_frame_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "video": str(args.video.resolve()),
        "model": str(args.model.resolve()),
        "provider": args.provider,
        "source_fps": source_fps,
        "source_frame_count": total_frames,
        "samples": len(rows),
        "inference_resolution": [args.inference_width, args.inference_height],
        "interpretation": (
            "Offline withheld-middle-frame quality and processing throughput; "
            "not display scanout FPS or input-to-photon latency."
        ),
        "methods": {
            name: _summary(rows, name)
            for name in ("midpoint", "raw", "legacy", "regional")
        },
    }
    report_path = output / "summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Per-frame metrics: {csv_path}")
    print(f"Visual samples: {output}")


if __name__ == "__main__":
    main()
