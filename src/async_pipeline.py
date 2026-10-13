"""Low-latency asynchronous capture and frame-processing primitives."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import logging
import math
import threading
import time
import traceback
from typing import Any, Callable

import cv2
import numpy as np

from resource_validation import QueueSizes


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    """A captured image and the monotonic metadata needed to trace it."""

    sequence_id: int
    image: Any
    captured_at: float
    capture_ms: float


@dataclass(frozen=True, slots=True)
class ProcessedFrame:
    """A processed image with end-to-end timing and adapter metadata."""

    sequence_id: int
    image: Any
    captured_at: float
    processing_started_at: float
    processed_at: float
    capture_ms: float
    content_timestamp: float | None = None
    preprocessing_ms: float | None = None
    inference_ms: float | None = None
    postprocessing_ms: float | None = None
    active_provider: str = "unknown"
    capture_dimensions: tuple[int, int] | None = None
    ai_input_dimensions: tuple[int, int] | None = None
    ai_output_dimensions: tuple[int, int] | None = None
    tile_mode: str = "off"
    interpolation_ms: float | None = None
    interpolation_provider: str = "none"
    frame_generation: str = "off"
    generation_method: str = "none"
    frame_kind: str = "processed"
    generation_position: float = 1.0
    regional_provenance: Any | None = None
    regional_confidence: Any | None = None
    regional_route_counts: dict[str, int] | None = None
    regional_provenance_counts: dict[str, int] | None = None
    regional_admission_reason: str | None = None
    global_motion_pixels: float | None = None
    global_motion_response: float | None = None
    reserve_suppressed: bool = False

    @property
    def frame_age_ms(self) -> float:
        """Return the age when processing began (time waiting for the worker)."""

        return (self.processing_started_at - self.captured_at) * 1000.0

    @property
    def processing_ms(self) -> float:
        return (self.processed_at - self.processing_started_at) * 1000.0

    @property
    def end_to_end_ms(self) -> float:
        return (self.processed_at - self.captured_at) * 1000.0


class DeferredRefinement:
    """Atomically choose a neural refinement or its cadence-safe fallback.

    The processing worker publishes the fallback immediately.  A separate
    worker may publish a refined image until the presenter resolves this
    object.  Resolution closes the slot, so a late inference can never mutate
    pixels which are already being uploaded or displayed.
    """

    def __init__(self, fallback: np.ndarray) -> None:
        self._fallback = fallback
        self._refined: np.ndarray | None = None
        self._metadata: dict[str, Any] = {}
        self._resolved = False
        self._lock = threading.Lock()

    def publish(self, image: np.ndarray, metadata: dict[str, Any]) -> bool:
        with self._lock:
            if self._resolved:
                return False
            self._refined = image
            self._metadata = dict(metadata)
            return True

    def resolve(self) -> tuple[np.ndarray, bool, dict[str, Any]]:
        with self._lock:
            self._resolved = True
            if self._refined is not None:
                return self._refined, True, dict(self._metadata)
            return self._fallback, False, {}

    def discard(self) -> None:
        with self._lock:
            self._resolved = True
            self._refined = None
            self._metadata.clear()


@dataclass(frozen=True, slots=True)
class _RefinementJob:
    frame_a: np.ndarray
    frame_b: np.ndarray
    target: DeferredRefinement
    camera_motion: bool = False


@dataclass(frozen=True, slots=True)
class WorkerFailure:
    """Serializable details for a capture or processing-worker failure."""

    stage: str
    exception_type: str
    message: str
    traceback_text: str


class PipelineWorkerError(RuntimeError):
    """Raised on the controlling thread when a pipeline worker has failed."""

    def __init__(self, failure: WorkerFailure) -> None:
        self.failure = failure
        super().__init__(
            f"{failure.stage} worker failed with {failure.exception_type}: "
            f"{failure.message}"
        )


class LatestFrameHandoff:
    """A bounded condition queue whose consumer always takes the newest item."""

    def __init__(self, capacity: int = 1) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("Frame queue capacity must be an integer.")
        if capacity <= 0:
            raise ValueError("Frame queue capacity must be greater than zero.")
        self.capacity = capacity
        self._condition = threading.Condition()
        self._items: deque[CapturedFrame] = deque()
        self._closed = False
        self._replacements = 0

    @property
    def replacements(self) -> int:
        with self._condition:
            return self._replacements

    @property
    def pending_count(self) -> int:
        with self._condition:
            return len(self._items)

    def publish(self, item: CapturedFrame) -> bool:
        """Publish an item, dropping the oldest item only when capacity is full."""

        with self._condition:
            if self._closed:
                raise RuntimeError("Cannot publish to a closed frame handoff.")

            replaced = len(self._items) >= self.capacity
            if replaced:
                self._items.popleft()
                self._replacements += 1
            self._items.append(item)
            self._condition.notify()
            return replaced

    def receive(self) -> CapturedFrame | None:
        """Wait for and consume only the newest item, dropping queued stale items."""

        with self._condition:
            self._condition.wait_for(lambda: self._items or self._closed)
            if not self._items:
                return None

            item = self._items.pop()
            stale_count = len(self._items)
            if stale_count:
                self._replacements += stale_count
                self._items.clear()
            return item

    def close(self, *, discard: bool = True) -> None:
        with self._condition:
            self._closed = True
            if discard and self._items:
                self._replacements += len(self._items)
                self._items.clear()
            self._condition.notify_all()

    def clear(self) -> int:
        """Discard queued stale input while keeping the handoff open."""

        with self._condition:
            count = len(self._items)
            if count:
                self._replacements += count
                self._items.clear()
            self._condition.notify_all()
            return count


class AsyncFramePipeline:
    """Run optional capture and required processing workers with newest-frame flow."""

    def __init__(
        self,
        processor: Any,
        *,
        capture_source: Callable[[], Any] | None = None,
        capture_shutdown: Callable[[], None] | None = None,
        capture_shutdown_on_worker: bool = False,
        frame_interpolator: Any | None = None,
        generated_frames: int = 1,
        buffered_playout: bool = False,
        queue_depth: int = 2,
        output_max_age_ms: float = 100.0,
        collect_telemetry: bool = False,
        clock: Callable[[], float] = time.perf_counter,
        worker_name: str = "frame-processing-worker",
        capture_worker_name: str = "frame-capture-worker",
    ) -> None:
        if isinstance(queue_depth, bool) or not isinstance(queue_depth, int):
            raise TypeError("Queue depth must be an integer.")
        if queue_depth <= 0:
            raise ValueError("Queue depth must be greater than zero.")
        if isinstance(generated_frames, bool) or not isinstance(generated_frames, int):
            raise TypeError("Generated frame count must be an integer.")
        if not 1 <= generated_frames <= 4:
            raise ValueError("Generated frame count must be between 1 and 4.")
        if (
            isinstance(output_max_age_ms, bool)
            or not isinstance(output_max_age_ms, (int, float))
            or not math.isfinite(float(output_max_age_ms))
            or output_max_age_ms <= 0.0
        ):
            raise ValueError("Output maximum age must be positive.")
        self._processor = processor
        self._capture_source = capture_source
        self._capture_shutdown = capture_shutdown
        self._capture_shutdown_on_worker = bool(capture_shutdown_on_worker)
        self._frame_interpolator = frame_interpolator
        self.generated_frames = generated_frames
        self.buffered_playout = bool(buffered_playout)
        self.output_max_age_ms = float(output_max_age_ms)
        self._previous_processed_image: Any | None = None
        self._previous_captured_at: float | None = None
        self._collect_telemetry = bool(collect_telemetry)
        self._clock = clock
        self._input = LatestFrameHandoff(queue_depth)
        self._result_condition = threading.Condition()
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._presentation_results: deque[ProcessedFrame] = deque()
        self._failure: WorkerFailure | None = None
        self._next_sequence_id = 0
        self._submitted_frames = 0
        self._processed_frames = 0
        self._result_replacements = 0
        self._dropped_generated_frames = 0
        self._stale_reserved_batches = 0
        self._interpolated_frames = 0
        self._interpolation_failures = 0
        self._refinement_submitted = 0
        self._refinement_completed_on_time = 0
        self._refinement_model_rejections = 0
        self._refinement_applied = 0
        self._refinement_fallback_presented = 0
        self._refinement_late_drops = 0
        self._refinement_job_replacements = 0
        self._refinement_condition = threading.Condition()
        self._refinement_job: _RefinementJob | None = None
        self._refinement_closed = False
        # Fractional 1/3 + 2/3 generation is allowed to prove that it fits the
        # live deadline once.  If it does not, preserve the known-good midpoint
        # path for the rest of the session instead of repeatedly starving
        # capture and publishing an already-stale three-frame batch.
        self._two_frame_pair_cost_ms: float | None = None
        self._two_frame_deadline_missed = False
        self._last_global_motion_pixels = 0.0
        self._last_global_motion_response = 0.0
        self._last_reserve_suppressed = False
        self._started = False
        self._stopped = False
        self._capture_finished = capture_source is None
        self._processing_finished = False
        self._capture_interrupted = False
        self._capture_shutdown_called = False
        self._processing_thread = threading.Thread(
            target=self._run_processing_worker,
            name=worker_name,
            daemon=False,
        )
        self._refinement_thread = (
            threading.Thread(
                target=self._run_refinement_worker,
                name="neural-refinement-worker",
                daemon=False,
            )
            if frame_interpolator is not None and self.buffered_playout
            else None
        )
        self._capture_thread = (
            threading.Thread(
                target=self._run_capture_worker,
                name=capture_worker_name,
                daemon=False,
            )
            if capture_source is not None
            else None
        )

    @property
    def queue_depth(self) -> int:
        return self._input.capacity

    @property
    def input_replacements(self) -> int:
        return self._input.replacements

    @property
    def result_replacements(self) -> int:
        with self._result_condition:
            return self._result_replacements

    @property
    def dropped_frames(self) -> int:
        return self.input_replacements + self.result_replacements

    @property
    def dropped_generated_frames(self) -> int:
        with self._result_condition:
            return self._dropped_generated_frames

    @property
    def stale_reserved_batches(self) -> int:
        with self._result_condition:
            return self._stale_reserved_batches

    @property
    def interpolated_frames(self) -> int:
        with self._state_lock:
            return self._interpolated_frames

    @property
    def interpolation_failures(self) -> int:
        with self._state_lock:
            return self._interpolation_failures

    @property
    def refinement_submitted(self) -> int:
        with self._state_lock:
            return self._refinement_submitted

    @property
    def refinement_applied(self) -> int:
        with self._state_lock:
            return self._refinement_applied

    @property
    def refinement_completed_on_time(self) -> int:
        with self._state_lock:
            return self._refinement_completed_on_time

    @property
    def refinement_model_rejections(self) -> int:
        with self._state_lock:
            return self._refinement_model_rejections

    @property
    def refinement_fallback_presented(self) -> int:
        with self._state_lock:
            return self._refinement_fallback_presented

    @property
    def refinement_late_drops(self) -> int:
        with self._state_lock:
            return self._refinement_late_drops

    @property
    def refinement_job_replacements(self) -> int:
        with self._state_lock:
            return self._refinement_job_replacements

    @property
    def submitted_frames(self) -> int:
        with self._state_lock:
            return self._submitted_frames

    @property
    def processed_frames(self) -> int:
        with self._state_lock:
            return self._processed_frames

    @property
    def capture_interrupted(self) -> bool:
        with self._state_lock:
            return self._capture_interrupted

    @property
    def finished(self) -> bool:
        with self._state_lock:
            return self._capture_finished and self._processing_finished

    @property
    def worker_threads_alive(self) -> bool:
        capture_alive = (
            self._capture_thread is not None and self._capture_thread.is_alive()
        )
        refinement_alive = (
            self._refinement_thread is not None
            and self._refinement_thread.is_alive()
        )
        return capture_alive or self._processing_thread.is_alive() or refinement_alive

    def queue_sizes(self) -> QueueSizes:
        """Return an atomic-enough validation snapshot of bounded queue sizes."""

        with self._result_condition:
            output_count = len(self._presentation_results)
        return QueueSizes(input=self._input.pending_count, output=output_count)

    def start(self) -> None:
        if self._started:
            raise RuntimeError("The asynchronous pipeline has already been started.")
        if self._stopped:
            raise RuntimeError("A stopped asynchronous pipeline cannot be restarted.")
        self._started = True
        if self._refinement_thread is not None:
            self._refinement_thread.start()
        self._processing_thread.start()
        if self._capture_thread is not None:
            self._capture_thread.start()

    def _next_packet(
        self,
        image: Any,
        *,
        captured_at: float,
        capture_ms: float,
    ) -> CapturedFrame:
        with self._state_lock:
            sequence_id = self._next_sequence_id
            self._next_sequence_id += 1
            self._submitted_frames += 1
        return CapturedFrame(
            sequence_id=sequence_id,
            image=image,
            captured_at=captured_at,
            capture_ms=float(capture_ms),
        )

    def submit(
        self,
        image: Any,
        *,
        captured_at: float | None = None,
        capture_ms: float = 0.0,
    ) -> int:
        """Submit a captured frame manually and return its sequence ID."""

        self.raise_if_failed()
        if not self._started or self._stopped:
            raise RuntimeError("The asynchronous pipeline is not running.")

        packet = self._next_packet(
            image,
            captured_at=self._clock() if captured_at is None else captured_at,
            capture_ms=capture_ms,
        )
        try:
            self._input.publish(packet)
        except RuntimeError:
            self.raise_if_failed()
            raise
        return packet.sequence_id

    def take_latest(self, timeout: float | None = 0.0) -> ProcessedFrame | None:
        """Consume the next presentation result without busy waiting."""

        self.raise_if_failed()
        with self._result_condition:
            if not self._presentation_results and timeout != 0.0:
                self._result_condition.wait_for(
                    lambda: (
                        bool(self._presentation_results)
                        or self._failure is not None
                        or self._processing_finished
                    ),
                    timeout=timeout,
                )
            result = (
                self._presentation_results.popleft()
                if self._presentation_results
                else None
            )

        self.raise_if_failed()
        return result

    def take_presentation_batch(
        self,
        timeout: float | None = 0.0,
    ) -> list[ProcessedFrame]:
        """Atomically consume one complete ordered presentation batch.

        The generated members and following real endpoint must be removed
        under one condition lock.  Unlocking between individual dequeues lets
        the producer replace the endpoint and can splice different source
        intervals into one presentation batch.
        """

        self.raise_if_failed()
        with self._result_condition:
            if not self._presentation_results and timeout != 0.0:
                self._result_condition.wait_for(
                    lambda: (
                        bool(self._presentation_results)
                        or self._failure is not None
                        or self._processing_finished
                    ),
                    timeout=timeout,
                )
            if not self._presentation_results:
                results: list[ProcessedFrame] = []
            else:
                # A short presenter stall must not force a stale reserved real
                # endpoint onto the screen before a newer complete batch.  If
                # a successor exists, retire the whole expired batch together
                # so temporal order cannot jump backward and then forward.
                while self._presentation_results:
                    queued = list(self._presentation_results)
                    first_batch_end = self._complete_batch_length(queued)
                    if first_batch_end >= len(queued):
                        break
                    endpoint = queued[first_batch_end - 1]
                    age_ms = (self._clock() - endpoint.captured_at) * 1000.0
                    if age_ms <= self.output_max_age_ms:
                        break
                    stale_batch = [
                        self._presentation_results.popleft()
                        for _ in range(first_batch_end)
                    ]
                    self._result_replacements += len(stale_batch)
                    self._dropped_generated_frames += sum(
                        result.frame_kind == "generated"
                        for result in stale_batch
                    )
                    self._stale_reserved_batches += 1

                results = []
                while self._presentation_results:
                    result = self._presentation_results.popleft()
                    results.append(result)
                    if result.frame_kind != "generated":
                        break

        self.raise_if_failed()
        return results

    @staticmethod
    def _complete_batch_length(results: list[ProcessedFrame]) -> int:
        return next(
            (
                index + 1
                for index, result in enumerate(results)
                if result.frame_kind != "generated"
            ),
            len(results),
        )

    def clear_queued_frames(self) -> int:
        """Clear stale input and presentation output during recovery."""

        cleared = self._input.clear()
        with self._result_condition:
            result_count = len(self._presentation_results)
            if result_count:
                self._result_replacements += result_count
                self._dropped_generated_frames += sum(
                    result.frame_kind == "generated"
                    for result in self._presentation_results
                )
                self._presentation_results.clear()
            self._previous_processed_image = None
            self._previous_captured_at = None
            self._result_condition.notify_all()
        return cleared + result_count

    def raise_if_failed(self) -> None:
        with self._result_condition:
            failure = self._failure
        if failure is not None:
            raise PipelineWorkerError(failure)

    def _call_capture_shutdown(self) -> None:
        if self._capture_shutdown is None or self._capture_shutdown_called:
            return
        self._capture_shutdown_called = True
        self._capture_shutdown()

    def stop(self, timeout: float = 5.0) -> None:
        """Wake, unblock, and join all workers. Safe to call repeatedly."""

        if self._stopped:
            return
        self._stopped = True
        self._stop_event.set()
        if not self._capture_shutdown_on_worker:
            self._call_capture_shutdown()
        self._input.close(discard=True)
        with self._refinement_condition:
            self._refinement_closed = True
            if self._refinement_job is not None:
                self._refinement_job.target.discard()
                self._refinement_job = None
            self._refinement_condition.notify_all()
        with self._result_condition:
            self._result_condition.notify_all()

        if not self._started:
            self._previous_processed_image = None
            return
        deadline = self._clock() + timeout
        threads = [
            thread
            for thread in (
                self._capture_thread,
                self._processing_thread,
                self._refinement_thread,
            )
            if thread is not None
        ]
        for thread in threads:
            remaining = max(0.0, deadline - self._clock())
            thread.join(timeout=remaining)
        alive = [thread.name for thread in threads if thread.is_alive()]
        if alive:
            raise RuntimeError(
                "Pipeline workers did not stop within "
                f"{timeout:.1f} seconds: {', '.join(alive)}."
            )
        self._previous_processed_image = None

    @staticmethod
    def _image_dimensions(image: Any) -> tuple[int, int] | None:
        shape = getattr(image, "shape", None)
        if shape is None or len(shape) < 2:
            return None
        return int(shape[1]), int(shape[0])

    def _processor_metadata(
        self,
        captured_image: Any,
        processed_image: Any,
    ) -> dict[str, Any]:
        active_providers = getattr(self._processor, "active_providers", ())
        if active_providers:
            active_provider = str(active_providers[0])
        else:
            backend = getattr(self._processor, "processing_backend", None)
            active_provider = str(
                getattr(
                    backend,
                    "display_name",
                    getattr(self._processor, "display_name", type(self._processor).__name__),
                )
            )
        configured_tile_mode = getattr(self._processor, "tile_mode", "off")
        selected_tile_size = getattr(self._processor, "selected_tile_size", None)
        tiles_processed = getattr(self._processor, "tiles_processed", 1)
        tile_mode = str(configured_tile_mode)
        if selected_tile_size is not None:
            tile_mode += f" ({selected_tile_size}px, {tiles_processed} tiles)"
        capture_dimensions = self._image_dimensions(captured_image)
        ai_input_dimensions = getattr(
            self._processor,
            "ai_input_dimensions",
            capture_dimensions,
        )
        ai_output_dimensions = getattr(
            self._processor,
            "output_dimensions",
            self._image_dimensions(processed_image),
        )
        return {
            "preprocessing_ms": getattr(
                self._processor,
                "last_preprocessing_ms",
                None,
            ),
            "inference_ms": getattr(self._processor, "last_inference_ms", None),
            "postprocessing_ms": getattr(
                self._processor,
                "last_postprocessing_ms",
                None,
            ),
            "active_provider": active_provider,
            "capture_dimensions": capture_dimensions,
            "ai_input_dimensions": ai_input_dimensions,
            "ai_output_dimensions": ai_output_dimensions,
            "tile_mode": tile_mode,
        }

    def _frame_generation_metadata(self) -> dict[str, Any]:
        if self._frame_interpolator is None:
            return {
                "interpolation_ms": None,
                "interpolation_provider": "none",
                "frame_generation": "off",
            }
        active_providers = getattr(
            self._frame_interpolator,
            "active_providers",
            (),
        )
        generation_method = getattr(
            self._frame_interpolator, "last_generation_method", "rife_midpoint"
        )
        return {
            "interpolation_ms": getattr(
                self._frame_interpolator,
                "last_total_ms",
                getattr(self._frame_interpolator, "last_inference_ms", None),
            ),
            "interpolation_provider": (
                "OpenCV regional/flow CPU"
                if generation_method in {
                    "farneback_flow",
                    "farneback_rejected",
                    "regional_temporal_fallback",
                    "deadline_degraded_midpoint",
                }
                else str(active_providers[0]) if active_providers else "none"
            ),
            "frame_generation": (
                "rife"
                if getattr(
                    self._frame_interpolator,
                    "produces_intermediate_frame",
                    False,
                )
                else "noop"
            ),
            "generation_method": generation_method,
        }

    def _submit_refinement(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        target: DeferredRefinement,
        *,
        camera_motion: bool = False,
    ) -> None:
        """Submit newest-only neural work without delaying the cadence path."""

        with self._refinement_condition:
            if self._refinement_closed:
                target.discard()
                return
            if self._refinement_job is not None:
                self._refinement_job.target.discard()
                with self._state_lock:
                    self._refinement_job_replacements += 1
            self._refinement_job = _RefinementJob(
                frame_a=frame_a,
                frame_b=frame_b,
                target=target,
                camera_motion=camera_motion,
            )
            with self._state_lock:
                self._refinement_submitted += 1
            self._refinement_condition.notify()

    def resolve_refinement(self, frame: ProcessedFrame) -> ProcessedFrame:
        """Resolve deferred pixels immediately before presentation/measurement."""

        if not isinstance(frame.image, DeferredRefinement):
            return frame
        image, refined, metadata = frame.image.resolve()
        with self._state_lock:
            if refined:
                self._refinement_applied += 1
            else:
                self._refinement_fallback_presented += 1
        if not refined:
            return replace(
                frame,
                image=image,
                generation_method="deadline_fallback_midpoint",
                interpolation_ms=0.0,
                interpolation_provider="OpenCV cadence fallback",
                regional_admission_reason="neural_not_ready_at_presentation",
            )
        return replace(
            frame,
            image=image,
            interpolation_ms=metadata.get("interpolation_ms"),
            interpolation_provider=metadata.get(
                "interpolation_provider", frame.interpolation_provider
            ),
            generation_method=metadata.get(
                "generation_method", "neural_refined_midpoint"
            ),
            regional_provenance=metadata.get("regional_provenance"),
            regional_confidence=metadata.get("regional_confidence"),
            regional_route_counts=metadata.get("regional_route_counts"),
            regional_provenance_counts=metadata.get("regional_provenance_counts"),
            regional_admission_reason=metadata.get(
                "regional_admission_reason", "neural_ready_before_presentation"
            ),
        )

    @staticmethod
    def discard_refinement(frame: ProcessedFrame) -> None:
        """Close deferred work for a pacing decision that will not be shown."""

        if isinstance(frame.image, DeferredRefinement):
            frame.image.discard()

    def _run_refinement_worker(self) -> None:
        """Run best-effort interpolation independently of capture/processing."""

        while True:
            with self._refinement_condition:
                self._refinement_condition.wait_for(
                    lambda: self._refinement_job is not None or self._refinement_closed
                )
                if self._refinement_job is None and self._refinement_closed:
                    return
                job = self._refinement_job
                self._refinement_job = None
            if job is None:
                continue
            started = self._clock()
            try:
                image = self._frame_interpolator.interpolate(job.frame_a, job.frame_b)
                interpolation_ms = (self._clock() - started) * 1000.0
                inference_ms = getattr(
                    self._frame_interpolator, "last_inference_ms", None
                )
                if inference_ms is None or float(inference_ms) <= 0.0:
                    # The regional gate returned a spatial/endpoint fallback.
                    # Do not mislabel that completion as a neural refinement;
                    # the already-published cadence fallback remains active.
                    with self._state_lock:
                        self._refinement_model_rejections += 1
                    continue
                regional = getattr(
                    self._frame_interpolator, "last_regional_result", None
                )
                generation_method = getattr(
                    self._frame_interpolator,
                    "last_generation_method",
                    "neural_refined_midpoint",
                )
                if job.camera_motion:
                    generation_method = "camera_motion_neural_refinement"
                active_providers = getattr(
                    self._frame_interpolator, "active_providers", ()
                )
                metadata = {
                    "interpolation_ms": interpolation_ms,
                    "interpolation_provider": (
                        str(active_providers[0]) if active_providers else "unknown"
                    ),
                    "generation_method": generation_method,
                    "regional_provenance": (
                        regional.provenance if regional is not None else None
                    ),
                    "regional_confidence": (
                        regional.confidence if regional is not None else None
                    ),
                    "regional_route_counts": (
                        regional.route_counts if regional is not None else None
                    ),
                    "regional_provenance_counts": (
                        regional.provenance_counts if regional is not None else None
                    ),
                    "regional_admission_reason": (
                        regional.admission_reason
                        if regional is not None
                        else "neural_ready_before_presentation"
                    ),
                }
                if job.target.publish(image, metadata):
                    with self._state_lock:
                        self._refinement_completed_on_time += 1
                else:
                    with self._state_lock:
                        self._refinement_late_drops += 1
            except Exception as error:
                with self._state_lock:
                    self._interpolation_failures += 1
                logger.warning(
                    "Neural refinement failed; cadence fallback remains active: %s",
                    error,
                )

    def _generate_intermediate_frames(
        self,
        frame_a: Any,
        frame_b: Any,
    ) -> list[tuple[float, Any, Any | None]]:
        """Generate one to four ordered frames with a midpoint-only interpolator."""

        interpolate = self._frame_interpolator.interpolate

        if (
            self.generated_frames == 2
            and not self.buffered_playout
            and self._two_frame_deadline_missed
        ):
            image = interpolate(frame_a, frame_b)
            # Keep the downgrade visible to telemetry and evaluation.  This is
            # one midpoint, not a successful two-insertion result.
            try:
                self._frame_interpolator.last_generation_method = (
                    "deadline_degraded_midpoint"
                )
            except (AttributeError, TypeError):
                pass
            return [
                (
                    0.5,
                    image,
                    getattr(self._frame_interpolator, "last_regional_result", None),
                )
            ]

        if self.generated_frames == 2 and self.buffered_playout:
            camera_motion = self._camera_motion_estimate(frame_a, frame_b)
            if isinstance(frame_a, np.ndarray) and isinstance(frame_b, np.ndarray):
                # Publish a cheap, correctly timed midpoint now.  Neural work
                # runs on its own newest-only worker and may replace these
                # pixels atomically until the presenter resolves the frame.
                midpoint_fallback = cv2.addWeighted(
                    frame_a, 0.5, frame_b, 0.5, 0.0
                )
                midpoint = DeferredRefinement(midpoint_fallback)
                self._submit_refinement(
                    frame_a,
                    frame_b,
                    midpoint,
                    camera_motion=camera_motion is not None,
                )
                if camera_motion is not None:
                    motion_pixels, response = camera_motion
                    self._last_global_motion_pixels = motion_pixels
                    self._last_global_motion_response = response
                    self._last_reserve_suppressed = True
                    return [(0.5, midpoint, None)]

                self._last_global_motion_pixels = 0.0
                self._last_global_motion_response = 0.0
                self._last_reserve_suppressed = False
                reserve = cv2.addWeighted(
                    midpoint_fallback, 0.5, frame_b, 0.5, 0.0
                )
                return [(0.5, midpoint, None), (0.75, reserve, None)]

            if camera_motion is not None:
                motion_pixels, response = camera_motion
                self._last_global_motion_pixels = motion_pixels
                self._last_global_motion_response = response
                self._last_reserve_suppressed = True
                midpoint = interpolate(frame_a, frame_b)
                try:
                    self._frame_interpolator.last_generation_method = (
                        "camera_motion_midpoint"
                    )
                except (AttributeError, TypeError):
                    pass
                # A reserve at 0.75 occupies a complete refresh interval but
                # advances only a quarter of the source interval. During a pan
                # this creates visible slow-fast motion. Keep a single,
                # geometrically placed midpoint so 30 Hz input remains evenly
                # spaced at 60 Hz output.
                return [(0.5, midpoint, None)]

            self._last_global_motion_pixels = 0.0
            self._last_global_motion_response = 0.0
            self._last_reserve_suppressed = False
            midpoint = interpolate(frame_a, frame_b)
            midpoint_regional = getattr(
                self._frame_interpolator, "last_regional_result", None
            )
            # The reserve exists to bridge a missed display slot, not to justify
            # a second neural inference.  Blend the already generated midpoint
            # toward the real endpoint for a sub-millisecond 0.75 candidate.
            # Generic/string test doubles retain the old callable fallback.
            if hasattr(midpoint, "shape") and hasattr(frame_b, "shape"):
                reserve = cv2.addWeighted(midpoint, 0.5, frame_b, 0.5, 0.0)
            else:
                interpolate_at = getattr(self._frame_interpolator, "interpolate_at", None)
                if callable(interpolate_at):
                    reserve = interpolate_at(frame_a, frame_b, 0.75)
                else:
                    reserve = interpolate(midpoint, frame_b)
            return [
                (0.5, midpoint, midpoint_regional),
                (0.75, reserve, midpoint_regional),
            ]

        if self.generated_frames == 2:
            interpolate_at = getattr(self._frame_interpolator, "interpolate_at", None)
            if callable(interpolate_at):
                generated = []
                for position in (1.0 / 3.0, 2.0 / 3.0):
                    image = interpolate_at(frame_a, frame_b, position)
                    generated.append(
                        (
                            position,
                            image,
                            getattr(self._frame_interpolator, "last_regional_result", None),
                        )
                    )
                return generated

        def generate(first, second):
            image = interpolate(first, second)
            return image, getattr(
                self._frame_interpolator, "last_regional_result", None
            )

        midpoint, midpoint_regional = generate(frame_a, frame_b)
        if self.generated_frames == 1:
            return [(0.5, midpoint, midpoint_regional)]

        quarter, quarter_regional = generate(frame_a, midpoint)
        three_quarter, three_quarter_regional = generate(midpoint, frame_b)
        if self.generated_frames == 2:
            return [
                (0.25, quarter, quarter_regional),
                (0.75, three_quarter, three_quarter_regional),
            ]
        if self.generated_frames == 3:
            return [
                (0.25, quarter, quarter_regional),
                (0.5, midpoint, midpoint_regional),
                (0.75, three_quarter, three_quarter_regional),
            ]

        one_eighth, one_eighth_regional = generate(frame_a, quarter)
        three_eighths, three_eighths_regional = generate(quarter, midpoint)
        five_eighths, five_eighths_regional = generate(midpoint, three_quarter)
        seven_eighths, seven_eighths_regional = generate(three_quarter, frame_b)
        return [
            (0.125, one_eighth, one_eighth_regional),
            (0.375, three_eighths, three_eighths_regional),
            (0.625, five_eighths, five_eighths_regional),
            (0.875, seven_eighths, seven_eighths_regional),
        ]

    @staticmethod
    def _camera_motion_estimate(
        frame_a: Any,
        frame_b: Any,
    ) -> tuple[float, float] | None:
        """Detect dominant camera motion cheaply at a fixed tiny resolution.

        Phase correlation is deliberately computed on a tiny image. A broad
        active area plus a coherent translation distinguishes camera motion
        from a moving object, animated HUD, or capture noise.
        """

        if not (
            isinstance(frame_a, np.ndarray)
            and isinstance(frame_b, np.ndarray)
            and frame_a.shape == frame_b.shape
            and frame_a.ndim == 3
            and frame_a.dtype == np.uint8
            and frame_b.dtype == np.uint8
        ):
            return None
        height, width = frame_a.shape[:2]
        scale = min(1.0, 160.0 / max(width, 1), 90.0 / max(height, 1))
        small_size = (
            max(32, int(round(width * scale))),
            max(18, int(round(height * scale))),
        )
        first = cv2.resize(frame_a, small_size, interpolation=cv2.INTER_AREA)
        second = cv2.resize(frame_b, small_size, interpolation=cv2.INTER_AREA)
        gray_a = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
        difference = cv2.absdiff(gray_a, gray_b)
        active_fraction = float(np.mean(difference >= 8))
        # At 160x90 a coherent camera pan often changes only 6-12% of pixels
        # by eight luma levels because large underwater regions are smooth.
        # Requiring a larger area missed the exact Subnautica turns this gate
        # is intended to stabilize.
        if active_fraction < 0.06:
            return None
        window = cv2.createHanningWindow(small_size, cv2.CV_32F)
        (shift_x, shift_y), response = cv2.phaseCorrelate(
            gray_a.astype(np.float32),
            gray_b.astype(np.float32),
            window,
        )
        if not np.isfinite((shift_x, shift_y, response)).all():
            return None
        scale_x = width / small_size[0]
        scale_y = height / small_size[1]
        full_shift_x = float(shift_x * scale_x)
        full_shift_y = float(shift_y * scale_y)
        motion_pixels = math.hypot(full_shift_x, full_shift_y)
        if response < 0.15 or motion_pixels < 1.25:
            return None
        return motion_pixels, float(response)

    def _publish_results(self, results: list[ProcessedFrame]) -> None:
        """Publish bounded ordered batches without pre-empting a midpoint.

        A generated frame and its following real frame form one presentation
        batch.  Once such a batch reaches the output handoff it must get one
        chance to reach the pacer; otherwise a newer worker result can erase
        the midpoint before the presenter even evaluates its deadline.

        Keep that oldest generated batch plus one newest successor.  Any
        superseded successor is replaced atomically, so overload still cannot
        grow into an unbounded FIFO queue.  Real-only output retains the
        original newest-frame behaviour.
        """

        with self._result_condition:
            if self._presentation_results:
                queued = list(self._presentation_results)
                first_batch_end = self._complete_batch_length(queued)
                reserved_batch = queued[:first_batch_end]
                has_reserved_generated = any(
                    result.frame_kind == "generated"
                    for result in reserved_batch
                )

                if has_reserved_generated:
                    superseded = queued[first_batch_end:]
                    if superseded:
                        self._result_replacements += len(superseded)
                        self._dropped_generated_frames += sum(
                            result.frame_kind == "generated"
                            for result in superseded
                        )
                        self._presentation_results.clear()
                        self._presentation_results.extend(reserved_batch)
                else:
                    self._result_replacements += len(queued)
                    self._dropped_generated_frames += sum(
                        result.frame_kind == "generated" for result in queued
                    )
                    self._presentation_results.clear()
            self._presentation_results.extend(results)
            self._result_condition.notify_all()

    @staticmethod
    def _detached_image(image: Any) -> Any:
        copy = getattr(image, "copy", None)
        return copy() if callable(copy) else image

    def _record_failure(self, stage: str, error: BaseException) -> None:
        failure = WorkerFailure(
            stage=stage,
            exception_type=type(error).__name__,
            message=str(error),
            traceback_text="".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
        )
        with self._result_condition:
            if self._failure is None:
                self._failure = failure
            self._result_condition.notify_all()
        self._stop_event.set()
        self._input.close(discard=True)

    def _run_capture_worker(self) -> None:
        if self._capture_source is None:
            return
        try:
            while not self._stop_event.is_set():
                captured_at = self._clock()
                image = self._capture_source()
                captured_end = self._clock()
                if self._stop_event.is_set():
                    break
                packet = self._next_packet(
                    image,
                    captured_at=captured_at,
                    capture_ms=(captured_end - captured_at) * 1000.0,
                )
                try:
                    self._input.publish(packet)
                except RuntimeError:
                    if self._stop_event.is_set():
                        break
                    raise
        except KeyboardInterrupt:
            with self._state_lock:
                self._capture_interrupted = True
            self._stop_event.set()
        except BaseException as error:
            if not self._stop_event.is_set():
                self._record_failure("capture", error)
        finally:
            if self._capture_shutdown_on_worker:
                self._call_capture_shutdown()
            with self._state_lock:
                self._capture_finished = True
            self._input.close(discard=False)
            with self._result_condition:
                self._result_condition.notify_all()

    def _run_processing_worker(self) -> None:
        try:
            while True:
                captured = self._input.receive()
                if captured is None:
                    return

                processing_started_at = self._clock()
                current_image = self._processor.process(captured.image)
                generates_middle_frame = bool(
                    self._frame_interpolator is not None
                    and getattr(
                        self._frame_interpolator,
                        "produces_intermediate_frame",
                        False,
                    )
                )
                generated_images: list[tuple[float, Any, Any | None]] = []
                interpolation_ms = None
                if generates_middle_frame and self._previous_processed_image is not None:
                    try:
                        interpolation_started = self._clock()
                        generated_images = self._generate_intermediate_frames(
                            self._previous_processed_image,
                            current_image,
                        )
                        interpolation_ms = (self._clock() - interpolation_started) * 1000.0
                        if (
                            self.generated_frames == 2
                            and not self.buffered_playout
                            and len(generated_images) == 2
                        ):
                            self._two_frame_pair_cost_ms = interpolation_ms
                            configured_budget = getattr(
                                self._frame_interpolator,
                                "regional_budget_ms",
                                None,
                            )
                            pair_budget_ms = max(
                                1000.0 / 30.0,
                                2.0 * float(configured_budget)
                                if configured_budget is not None
                                else 0.0,
                            )
                            if interpolation_ms > pair_budget_ms:
                                self._two_frame_deadline_missed = True
                                logger.warning(
                                    "Two-insertion generation took %.2f ms (budget %.2f ms); "
                                    "degrading to the stable 1x midpoint path for this session.",
                                    interpolation_ms,
                                    pair_budget_ms,
                                )
                        with self._state_lock:
                            self._interpolated_frames += len(generated_images)
                    except Exception as error:
                        with self._state_lock:
                            self._interpolation_failures += 1
                        logger.warning(
                            "RIFE interpolation failed; presenting the processed frame: %s",
                            error,
                        )
                elif (
                    self._frame_interpolator is not None
                    and self._previous_processed_image is not None
                ):
                    current_image = self._frame_interpolator.interpolate(
                        self._previous_processed_image,
                        current_image,
                    )
                self._previous_processed_image = (
                    self._detached_image(current_image)
                    if generates_middle_frame
                    else current_image
                )
                previous_captured_at = self._previous_captured_at
                self._previous_captured_at = captured.captured_at
                processed_at = self._clock()
                metadata = (
                    self._processor_metadata(captured.image, current_image)
                    if self._collect_telemetry
                    else {}
                )
                frame_generation_metadata = self._frame_generation_metadata()
                if interpolation_ms is not None and self.generated_frames > 1:
                    frame_generation_metadata["interpolation_ms"] = interpolation_ms
                current_result = ProcessedFrame(
                    sequence_id=captured.sequence_id,
                    image=current_image,
                    captured_at=captured.captured_at,
                    processing_started_at=processing_started_at,
                    processed_at=processed_at,
                    capture_ms=captured.capture_ms,
                    content_timestamp=captured.captured_at,
                    global_motion_pixels=self._last_global_motion_pixels,
                    global_motion_response=self._last_global_motion_response,
                    reserve_suppressed=self._last_reserve_suppressed,
                    **metadata,
                    **frame_generation_metadata,
                )
                results = []
                for generation_position, generated_image, regional_result in generated_images:
                    is_buffered_reserve = (
                        self.buffered_playout and generation_position > 0.5 + 1e-6
                    )
                    candidate_metadata = dict(frame_generation_metadata)
                    if is_buffered_reserve:
                        candidate_metadata["generation_method"] = "buffered_reserve_blend"
                    regional_provenance = (
                        regional_result.provenance
                        if regional_result is not None
                        else None
                    )
                    regional_provenance_counts = (
                        regional_result.provenance_counts
                        if regional_result is not None
                        else None
                    )
                    regional_admission_reason = (
                        regional_result.admission_reason
                        if regional_result is not None
                        else None
                    )
                    if is_buffered_reserve:
                        # The reserve is a new full-frame blend. Do not copy the
                        # midpoint's interpolation provenance onto different pixels.
                        regional_provenance = None
                        reserve_shape = getattr(generated_image, "shape", ())
                        reserve_pixels = (
                            int(reserve_shape[0]) * int(reserve_shape[1])
                            if len(reserve_shape) >= 2
                            else 1
                        )
                        regional_provenance_counts = {
                            "reserve_blend": reserve_pixels
                        }
                        regional_admission_reason = "buffered_reserve_blend"
                    results.append(
                        ProcessedFrame(
                            sequence_id=captured.sequence_id,
                            image=generated_image,
                            captured_at=captured.captured_at,
                            processing_started_at=processing_started_at,
                            processed_at=processed_at,
                            capture_ms=captured.capture_ms,
                            content_timestamp=(
                                captured.captured_at
                                if previous_captured_at is None
                                else previous_captured_at
                                + generation_position
                                * (captured.captured_at - previous_captured_at)
                            ),
                            **metadata,
                            **candidate_metadata,
                            frame_kind="generated",
                            generation_position=generation_position,
                            global_motion_pixels=self._last_global_motion_pixels,
                            global_motion_response=self._last_global_motion_response,
                            reserve_suppressed=self._last_reserve_suppressed,
                            regional_provenance=regional_provenance,
                            regional_confidence=(
                                regional_result.confidence
                                if regional_result is not None
                                else None
                            ),
                            regional_route_counts=(
                                regional_result.route_counts
                                if regional_result is not None
                                else None
                            ),
                            regional_provenance_counts=regional_provenance_counts,
                            regional_admission_reason=regional_admission_reason,
                        )
                    )
                results.append(current_result)
                with self._state_lock:
                    self._processed_frames += 1
                self._publish_results(results)
        except BaseException as error:
            if not self._stop_event.is_set():
                self._record_failure("processing", error)
        finally:
            with self._state_lock:
                self._processing_finished = True
            with self._result_condition:
                self._result_condition.notify_all()
