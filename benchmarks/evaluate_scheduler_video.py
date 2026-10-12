"""Real-time output-scheduler ablation on a recorded video.

This benchmark feeds a fixed recording to the asynchronous interpolation
pipeline at a configured source cadence and paces accepted output at a fixed
target.  It compares the former replace-all output handoff with the bounded
reserved-batch handoff.  Results are software presentation decisions, not
display scanout or input-to-photon measurements.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import statistics
import sys
import threading
import time

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from async_pipeline import AsyncFramePipeline
from frame_interpolator import RIFEInterpolator
from frame_pacing import FramePacer, PresentationFrame


class IdentityProcessor:
    def process(self, frame):
        return frame

    def close(self) -> None:
        return None


class MidpointInterpolator:
    """Cheap deterministic control that isolates scheduling from inference."""

    produces_intermediate_frame = True
    active_providers = ("scheduler-control",)
    last_generation_method = "midpoint_control"
    last_total_ms = 0.0

    def interpolate(self, frame_a, frame_b, timestep: float = 0.5):
        started = time.perf_counter()
        result = cv2.addWeighted(
            frame_a, 1.0 - float(timestep), frame_b, float(timestep), 0.0
        )
        self.last_total_ms = (time.perf_counter() - started) * 1000.0
        return result

    def shutdown(self) -> None:
        return None


class LegacyReplacementPipeline(AsyncFramePipeline):
    """Pre-fix output policy used only as the ablation control."""

    def _publish_results(self, results):
        with self._result_condition:
            if self._presentation_results:
                self._result_replacements += len(self._presentation_results)
                self._dropped_generated_frames += sum(
                    result.frame_kind == "generated"
                    for result in self._presentation_results
                )
                self._presentation_results.clear()
            self._presentation_results.extend(results)
            self._result_condition.notify_all()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--provider", choices=("cpu", "directml"), default="cpu")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--source-width", type=int, default=1280)
    parser.add_argument("--source-height", type=int, default=720)
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--target-fps", type=float, default=60.0)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--inference-width", type=int, default=160)
    parser.add_argument("--inference-height", type=int, default=90)
    parser.add_argument("--budget-ms", type=float, default=16.0)
    parser.add_argument(
        "--scheduler-only",
        action="store_true",
        help="Use a deterministic midpoint to isolate queue and pacing policy.",
    )
    parser.add_argument("--submit-overhead-ms", type=float, default=1.0)
    parser.add_argument(
        "--stalls-ms",
        default="0,10,25,50,100",
        help="Comma-separated presenter stalls injected once per second.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "output" / "scheduler_video_evaluation",
    )
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[round(fraction * (len(ordered) - 1))])


def run_condition(
    args: argparse.Namespace, policy: str, stall_ms: float
) -> dict[str, object]:
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {args.video}")
    recorded_fps = float(capture.get(cv2.CAP_PROP_FPS) or args.source_fps)
    source_step = max(1, round(recorded_fps / args.source_fps))

    if args.scheduler_only:
        interpolator = MidpointInterpolator()
    else:
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
    pipeline_type = (
        LegacyReplacementPipeline if policy == "legacy_replace_all" else AsyncFramePipeline
    )
    pipeline = pipeline_type(
        IdentityProcessor(),
        frame_interpolator=interpolator,
        generated_frames=1,
        queue_depth=2,
    )
    pacer = FramePacer(
        mode="fixed",
        target_fps=args.target_fps,
        max_frame_latency_ms=100.0,
        keep_latest_real=True,
    )
    producer_done = threading.Event()
    producer_error: list[str] = []

    def produce() -> None:
        next_submit = time.perf_counter()
        submitted = 0
        try:
            while submitted < round(args.duration * args.source_fps):
                ok, frame = capture.read()
                if not ok:
                    break
                for _ in range(source_step - 1):
                    if not capture.grab():
                        break
                frame = cv2.resize(
                    frame,
                    (args.source_width, args.source_height),
                    interpolation=cv2.INTER_AREA,
                )
                delay = next_submit - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                pipeline.submit(frame)
                submitted += 1
                next_submit += 1.0 / args.source_fps
        except BaseException as error:  # benchmark must report worker failures
            producer_error.append(repr(error))
        finally:
            producer_done.set()

    pipeline.start()
    started = time.perf_counter()
    producer = threading.Thread(target=produce, name=f"video-producer-{policy}")
    producer.start()
    presented_kinds: list[str] = []
    presentation_times: list[float] = []
    sequence_ids: list[int] = []
    max_output_queue = 0
    next_stall_at = round(args.target_fps)
    try:
        while True:
            max_output_queue = max(max_output_queue, pipeline.queue_sizes().output)
            batch = pipeline.take_presentation_batch(timeout=0.05)
            for decision in pacer.iter_pace_batch(
                PresentationFrame(
                    result,
                    result.captured_at,
                    result.frame_kind,
                    result.sequence_id,
                )
                for result in batch
            ):
                if not decision.present:
                    continue
                if args.submit_overhead_ms > 0:
                    time.sleep(args.submit_overhead_ms / 1000.0)
                presented_kinds.append(decision.frame.frame_kind)
                presentation_times.append(time.perf_counter())
                sequence_ids.append(decision.frame.source_sequence_id)
                if stall_ms > 0.0 and len(presented_kinds) >= next_stall_at:
                    time.sleep(stall_ms / 1000.0)
                    next_stall_at += round(args.target_fps)

            sizes = pipeline.queue_sizes()
            if (
                producer_done.is_set()
                and (
                    pipeline.processed_frames + pipeline.input_replacements
                    >= pipeline.submitted_frames
                )
                and sizes.input == 0
                and sizes.output == 0
            ):
                break
    finally:
        producer.join(timeout=2.0)
        elapsed = time.perf_counter() - started
        pacer.stop()
        pipeline.stop()
        interpolator.shutdown()
        capture.release()

    if producer_error:
        raise RuntimeError(producer_error[0])
    intervals_ms = [
        (right - left) * 1000.0
        for left, right in zip(presentation_times, presentation_times[1:])
    ]
    snapshot = pacer.snapshot()
    paired_generated = sum(
        kind == "generated" and index + 1 < len(sequence_ids)
        and sequence_ids[index + 1] == sequence_ids[index]
        for index, kind in enumerate(presented_kinds)
    )
    return {
        "policy": policy,
        "injected_stall_ms": stall_ms,
        "elapsed_seconds": elapsed,
        "submitted_real_frames": pipeline.submitted_frames,
        "processed_real_frames": pipeline.processed_frames,
        "presented_real_frames": presented_kinds.count("processed"),
        "presented_generated_frames": presented_kinds.count("generated"),
        "paired_generated_frames": paired_generated,
        "software_presentation_fps": (
            (len(presentation_times) - 1)
            / (presentation_times[-1] - presentation_times[0])
            if len(presentation_times) > 1
            else 0.0
        ),
        "median_interval_ms": statistics.median(intervals_ms) if intervals_ms else 0.0,
        "p95_interval_ms": percentile(intervals_ms, 0.95),
        "p99_interval_ms": percentile(intervals_ms, 0.99),
        "input_queue_replacements": pipeline.input_replacements,
        "output_queue_replacements": pipeline.result_replacements,
        "pipeline_generated_drops": pipeline.dropped_generated_frames,
        "stale_reserved_batches": pipeline.stale_reserved_batches,
        "max_output_queue_frames": max_output_queue,
        "pacer_generated_drops": snapshot.generated_frames_dropped_late,
        "pacer_real_drops": snapshot.real_frames_dropped_late,
        "cadence_misses": snapshot.cadence_misses,
    }


def main() -> None:
    args = parse_args()
    if not args.video.is_file():
        raise SystemExit(f"Video does not exist: {args.video}")
    if not args.model.is_file():
        raise SystemExit(f"Model does not exist: {args.model}")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    stalls = [float(value) for value in args.stalls_ms.split(",") if value.strip()]
    results = [
        run_condition(args, policy, stall_ms)
        for stall_ms in stalls
        for policy in ("legacy_replace_all", "reserved_generated_batch")
    ]
    report = {
        "video": str(args.video.resolve()),
        "source_size": [args.source_width, args.source_height],
        "source_fps": args.source_fps,
        "target_fps": args.target_fps,
        "duration_seconds": args.duration,
        "scheduler_only": bool(args.scheduler_only),
        "interpretation": (
            "Software scheduler ablation; not display scanout or input-to-photon measurement."
        ),
        "conditions": results,
    }
    output = args.output_directory / "summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
