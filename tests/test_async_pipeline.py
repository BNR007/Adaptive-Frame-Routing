"""Tests for the bounded latest-frame asynchronous pipeline."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from async_pipeline import (
    AsyncFramePipeline,
    CapturedFrame,
    LatestFrameHandoff,
    PipelineWorkerError,
    ProcessedFrame,
)
from frame_interpolator import NoOpInterpolator


def test_latest_frame_handoff_replaces_stale_input() -> None:
    handoff = LatestFrameHandoff()
    first = CapturedFrame(0, "old", 1.0, 2.0)
    second = CapturedFrame(1, "new", 2.0, 3.0)

    assert handoff.publish(first) is False
    assert handoff.publish(second) is True

    received = handoff.receive()
    assert received is second
    assert handoff.replacements == 1


def test_pipeline_processes_newest_waiting_frame_with_sequence_metadata() -> None:
    processing_started = threading.Event()
    release_first = threading.Event()
    processed_sequences: list[int] = []

    class BlockingProcessor:
        def process(self, frame: int) -> int:
            processed_sequences.append(frame)
            if frame == 0:
                processing_started.set()
                assert release_first.wait(timeout=2.0)
            return frame * 10

    pipeline = AsyncFramePipeline(BlockingProcessor())
    pipeline.start()
    try:
        assert pipeline.submit(0) == 0
        assert processing_started.wait(timeout=2.0)
        assert pipeline.submit(1) == 1
        assert pipeline.submit(2) == 2
        release_first.set()

        deadline = time.perf_counter() + 2.0
        newest = None
        while time.perf_counter() < deadline:
            candidate = pipeline.take_latest(timeout=0.05)
            if candidate is not None and candidate.sequence_id == 2:
                newest = candidate
                break

        assert newest is not None
        assert newest.image == 20
        assert processed_sequences == [0, 2]
        assert pipeline.input_replacements == 1
        assert newest.processing_ms >= 0.0
        assert newest.end_to_end_ms >= newest.processing_ms
    finally:
        release_first.set()
        pipeline.stop()


def test_pipeline_surfaces_typed_worker_failure() -> None:
    class FailingProcessor:
        def process(self, frame) -> None:
            raise ValueError(f"bad frame {frame}")

    pipeline = AsyncFramePipeline(FailingProcessor())
    pipeline.start()
    try:
        pipeline.submit(7)

        with pytest.raises(PipelineWorkerError) as error_info:
            pipeline.take_latest(timeout=2.0)

        failure = error_info.value.failure
        assert failure.stage == "processing"
        assert failure.exception_type == "ValueError"
        assert failure.message == "bad frame 7"
        assert "ValueError: bad frame 7" in failure.traceback_text
    finally:
        pipeline.stop()


def test_pipeline_stop_wakes_idle_worker() -> None:
    class IdentityProcessor:
        def process(self, frame):
            return frame

    pipeline = AsyncFramePipeline(IdentityProcessor())
    pipeline.start()

    started = time.perf_counter()
    pipeline.stop(timeout=1.0)

    assert time.perf_counter() - started < 1.0


def test_bounded_handoff_drops_oldest_and_consumer_takes_newest() -> None:
    handoff = LatestFrameHandoff(capacity=2)
    frames = [CapturedFrame(index, index, 1.0, 0.0) for index in range(3)]

    assert handoff.publish(frames[0]) is False
    assert handoff.publish(frames[1]) is False
    assert handoff.pending_count == 2
    assert handoff.publish(frames[2]) is True

    assert handoff.receive() is frames[2]
    assert handoff.pending_count == 0
    assert handoff.replacements == 2


@pytest.mark.parametrize("queue_depth", [0, -1])
def test_pipeline_rejects_invalid_queue_depth(queue_depth: int) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        AsyncFramePipeline(object(), queue_depth=queue_depth)


def test_capture_and_processing_threads_drop_stale_frames_and_clean_up() -> None:
    processing_started = threading.Event()
    release_processing = threading.Event()
    capture_finished = threading.Event()
    captured_value = 0
    processed_values: list[int] = []

    def capture_source() -> int:
        nonlocal captured_value
        value = captured_value
        captured_value += 1
        if value == 0:
            return value
        assert processing_started.wait(timeout=2.0)
        if value <= 3:
            return value
        capture_finished.set()
        raise KeyboardInterrupt

    class BlockingProcessor:
        def process(self, frame: int) -> int:
            processed_values.append(frame)
            if frame == 0:
                processing_started.set()
                assert release_processing.wait(timeout=2.0)
            return frame

    pipeline = AsyncFramePipeline(
        BlockingProcessor(),
        capture_source=capture_source,
        queue_depth=2,
    )
    pipeline.start()
    try:
        assert capture_finished.wait(timeout=2.0)
        release_processing.set()

        results = []
        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            result = pipeline.take_latest(timeout=0.05)
            if result is not None:
                results.append(result)
            if pipeline.finished:
                break

        assert processed_values == [0, 3]
        assert pipeline.submitted_frames == 4
        assert pipeline.processed_frames == 2
        assert pipeline.input_replacements == 2
        assert pipeline.capture_interrupted is True
        assert results[-1].sequence_id == 3
    finally:
        release_processing.set()
        pipeline.stop(timeout=1.0)

    assert pipeline.worker_threads_alive is False


def test_pipeline_stop_calls_capture_shutdown_and_is_idempotent() -> None:
    shutdown_called = threading.Event()

    def capture_source():
        shutdown_called.wait(timeout=2.0)
        raise RuntimeError("capture closed")

    pipeline = AsyncFramePipeline(
        processor=object(),
        capture_source=capture_source,
        capture_shutdown=shutdown_called.set,
    )
    pipeline.start()

    pipeline.stop(timeout=1.0)
    pipeline.stop(timeout=1.0)

    assert shutdown_called.is_set()
    assert pipeline.worker_threads_alive is False


def test_thread_bound_capture_shutdown_runs_on_capture_worker() -> None:
    shutdown_threads = []

    class Identity:
        def process(self, frame):
            return frame

    pipeline = AsyncFramePipeline(
        Identity(),
        capture_source=lambda: object(),
        capture_shutdown=lambda: shutdown_threads.append(threading.current_thread().name),
        capture_shutdown_on_worker=True,
    )
    pipeline.start()
    try:
        assert pipeline.take_latest(timeout=1.0) is not None
    finally:
        pipeline.stop(timeout=2.0)

    assert shutdown_threads == ["frame-capture-worker"]
    assert pipeline.worker_threads_alive is False


def test_processed_frame_snapshots_ai_telemetry_without_racing() -> None:
    class TelemetryProcessor:
        active_providers = ("DmlExecutionProvider", "CPUExecutionProvider")
        tile_mode = "auto"
        selected_tile_size = 256
        tiles_processed = 2
        last_preprocessing_ms = 1.25
        last_inference_ms = 4.5
        last_postprocessing_ms = 0.75
        ai_input_dimensions = (320, 180)
        output_dimensions = (640, 360)

        def process(self, frame):
            return frame

    frame = type("Frame", (), {"shape": (180, 320, 3)})()
    pipeline = AsyncFramePipeline(
        TelemetryProcessor(),
        collect_telemetry=True,
    )
    pipeline.start()
    try:
        pipeline.submit(frame)
        result = pipeline.take_latest(timeout=1.0)
        assert result is not None
        assert result.preprocessing_ms == 1.25
        assert result.inference_ms == 4.5
        assert result.postprocessing_ms == 0.75
        assert result.active_provider == "DmlExecutionProvider"
        assert result.capture_dimensions == (320, 180)
        assert result.ai_input_dimensions == (320, 180)
        assert result.ai_output_dimensions == (640, 360)
        assert result.tile_mode == "auto (256px, 2 tiles)"
    finally:
        pipeline.stop()


def test_noop_interpolator_hook_preserves_async_output_and_frame_order() -> None:
    interpolator = NoOpInterpolator()
    interpolator.initialize()

    class IdentityProcessor:
        def process(self, frame):
            return frame

    pipeline = AsyncFramePipeline(
        IdentityProcessor(),
        frame_interpolator=interpolator,
    )
    pipeline.start()
    try:
        pipeline.submit("first")
        first = pipeline.take_latest(timeout=1.0)
        assert first is not None
        assert first.image == "first"

        pipeline.submit("second")
        second = pipeline.take_latest(timeout=1.0)
        assert second is not None
        assert second.image == "second"
    finally:
        pipeline.stop()
        interpolator.shutdown()


class _MidpointInterpolator:
    produces_intermediate_frame = True
    active_providers = ("CPUExecutionProvider",)
    last_inference_ms = 4.25

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = []
        self.shutdown_calls = 0

    def interpolate(self, frame_a, frame_b):
        self.calls.append((frame_a, frame_b))
        if self.fail:
            raise RuntimeError("interpolation unavailable")
        return f"middle({frame_a},{frame_b})"

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def _wait_for_processed_frames(
    pipeline: AsyncFramePipeline,
    count: int,
) -> None:
    deadline = time.perf_counter() + 2.0
    while pipeline.processed_frames < count and time.perf_counter() < deadline:
        time.sleep(0.005)
    assert pipeline.processed_frames >= count


def test_rife_frame_generation_presents_a_middle_b_in_order() -> None:
    interpolator = _MidpointInterpolator()

    class IdentityProcessor:
        def process(self, frame):
            return frame

    pipeline = AsyncFramePipeline(
        IdentityProcessor(),
        frame_interpolator=interpolator,
        collect_telemetry=True,
    )
    pipeline.start()
    try:
        pipeline.submit("A")
        first = pipeline.take_latest(timeout=1.0)
        assert first is not None
        assert (first.image, first.frame_kind) == ("A", "processed")

        pipeline.submit("B")
        middle = pipeline.take_latest(timeout=1.0)
        current = pipeline.take_latest(timeout=1.0)

        assert middle is not None
        assert current is not None
        assert (middle.image, middle.frame_kind) == (
            "middle(A,B)",
            "generated",
        )
        assert (current.image, current.frame_kind) == ("B", "processed")
        assert middle.sequence_id == current.sequence_id == 1
        assert middle.frame_generation == "rife"
        assert middle.interpolation_ms == 4.25
        assert middle.interpolation_provider == "CPUExecutionProvider"
        assert pipeline.interpolated_frames == 1
    finally:
        pipeline.stop()


@pytest.mark.parametrize("generated_frames", (1, 2, 3, 4))
def test_rife_generation_amount_publishes_requested_ordered_batch(
    generated_frames: int,
) -> None:
    class IdentityProcessor:
        def process(self, frame):
            return frame

    pipeline = AsyncFramePipeline(
        IdentityProcessor(),
        frame_interpolator=_MidpointInterpolator(),
        generated_frames=generated_frames,
    )
    pipeline.start()
    try:
        pipeline.submit("A")
        assert pipeline.take_latest(timeout=1.0).image == "A"
        pipeline.submit("B")
        batch = pipeline.take_presentation_batch(timeout=1.0)

        assert len(batch) == generated_frames + 1
        assert [frame.frame_kind for frame in batch] == [
            *("generated" for _ in range(generated_frames)),
            "processed",
        ]
        assert [frame.generation_position for frame in batch[:-1]] == sorted(
            frame.generation_position for frame in batch[:-1]
        )
        assert batch[-1].image == "B"
        assert pipeline.interpolated_frames == generated_frames
    finally:
        pipeline.stop()


def test_two_generated_frames_use_uniform_third_and_two_third_positions():
    class FractionalInterpolator(_MidpointInterpolator):
        last_generation_method = "farneback_flow"

        def __init__(self):
            super().__init__()
            self.positions = []

        def interpolate_at(self, frame_a, frame_b, position):
            self.positions.append(position)
            return f"at({position:.6f})"

    interpolator = FractionalInterpolator()

    class IdentityProcessor:
        def process(self, frame):
            return frame

    pipeline = AsyncFramePipeline(
        IdentityProcessor(), frame_interpolator=interpolator, generated_frames=2
    )
    pipeline.start()
    try:
        pipeline.submit("A")
        assert pipeline.take_latest(timeout=1.0).image == "A"
        pipeline.submit("B")
        batch = pipeline.take_presentation_batch(timeout=1.0)
        assert [item.generation_position for item in batch[:-1]] == pytest.approx(
            [1.0 / 3.0, 2.0 / 3.0]
        )
        assert [item.image for item in batch[:-1]] == [
            "at(0.333333)",
            "at(0.666667)",
        ]
        assert interpolator.positions == pytest.approx([1.0 / 3.0, 2.0 / 3.0])
        assert batch[0].interpolation_provider == "OpenCV regional/flow CPU"
        assert batch[0].generation_method == "farneback_flow"
    finally:
        pipeline.stop()


def test_buffered_two_candidate_mode_keeps_midpoint_and_adds_late_reserve():
    class FractionalInterpolator(_MidpointInterpolator):
        last_generation_method = "buffered_candidates"

        def interpolate_at(self, frame_a, frame_b, position):
            return f"reserve({position:.2f})"

    class IdentityProcessor:
        def process(self, frame):
            return frame

    pipeline = AsyncFramePipeline(
        IdentityProcessor(),
        frame_interpolator=FractionalInterpolator(),
        generated_frames=2,
        buffered_playout=True,
    )
    pipeline.start()
    try:
        pipeline.submit("A")
        assert pipeline.take_latest(timeout=1.0).image == "A"
        pipeline.submit("B")
        batch = pipeline.take_presentation_batch(timeout=1.0)
        assert [item.generation_position for item in batch] == pytest.approx(
            [0.5, 0.75, 1.0]
        )
        assert batch[0].image == "middle(A,B)"
        assert batch[1].image == "reserve(0.75)"
        assert batch[2].image == "B"
    finally:
        pipeline.stop()


def test_buffered_reserve_uses_one_model_call_for_image_frames():
    class CountingInterpolator:
        produces_intermediate_frame = True
        active_providers = ("test",)
        last_generation_method = "counting"
        last_total_ms = 0.0
        last_regional_result = None

        def __init__(self):
            self.calls = 0

        def interpolate(self, frame_a, frame_b):
            self.calls += 1
            self.last_regional_result = SimpleNamespace(
                provenance=np.full((2, 2), 2, dtype=np.uint8),
                confidence=np.full((2, 2), 0.8, dtype=np.float32),
                route_counts={"smooth_motion": 4},
                provenance_counts={"interpolated": 4},
                admission_reason="within_budget",
            )
            return ((frame_a.astype(np.uint16) + frame_b.astype(np.uint16)) // 2).astype(
                np.uint8
            )

        def interpolate_at(self, frame_a, frame_b, position):
            raise AssertionError("buffer reserve must not run a second model inference")

    class IdentityProcessor:
        def process(self, frame):
            return frame

    interpolator = CountingInterpolator()
    pipeline = AsyncFramePipeline(
        IdentityProcessor(),
        frame_interpolator=interpolator,
        generated_frames=2,
        buffered_playout=True,
    )
    first = np.zeros((4, 4, 3), dtype=np.uint8)
    second = np.full((4, 4, 3), 100, dtype=np.uint8)
    pipeline.start()
    try:
        pipeline.submit(first)
        first_result = pipeline.take_latest(timeout=1.0)
        assert first_result is not None
        pipeline.submit(second)
        batch = pipeline.take_presentation_batch(timeout=1.0)
        assert interpolator.calls == 1
        assert [item.generation_position for item in batch] == pytest.approx(
            [0.5, 0.75, 1.0]
        )
        assert np.all(batch[0].image == 50)
        assert np.all(batch[1].image == 75)
        assert batch[0].generation_method == "counting"
        assert batch[0].regional_provenance_counts == {"interpolated": 4}
        assert batch[1].generation_method == "buffered_reserve_blend"
        assert batch[1].regional_provenance_counts == {"reserve_blend": 4}
        assert batch[1].regional_admission_reason == "buffered_reserve_blend"
        assert (
            first_result.content_timestamp
            < batch[0].content_timestamp
            < batch[1].content_timestamp
            < batch[2].content_timestamp
        )
    finally:
        pipeline.stop()


def test_two_frame_deadline_miss_degrades_to_stable_midpoint():
    class SlowFractionalInterpolator(_MidpointInterpolator):
        last_generation_method = "farneback_flow"
        regional_budget_ms = 16.0

        def __init__(self):
            super().__init__()
            self.fractional_calls = 0
            self.midpoint_calls = 0

        def interpolate_at(self, frame_a, frame_b, position):
            self.fractional_calls += 1
            time.sleep(0.025)
            return f"fractional({position:.3f})"

        def interpolate(self, frame_a, frame_b):
            self.midpoint_calls += 1
            return f"midpoint({frame_a},{frame_b})"

    class IdentityProcessor:
        def process(self, frame):
            return frame

    interpolator = SlowFractionalInterpolator()
    pipeline = AsyncFramePipeline(
        IdentityProcessor(), frame_interpolator=interpolator, generated_frames=2
    )
    pipeline.start()
    try:
        pipeline.submit("A")
        assert pipeline.take_latest(timeout=1.0).image == "A"

        pipeline.submit("B")
        first_batch = pipeline.take_presentation_batch(timeout=1.0)
        assert len(first_batch) == 3
        assert interpolator.fractional_calls == 2

        pipeline.submit("C")
        degraded_batch = pipeline.take_presentation_batch(timeout=1.0)
        assert len(degraded_batch) == 2
        assert degraded_batch[0].generation_position == pytest.approx(0.5)
        assert degraded_batch[0].generation_method == "deadline_degraded_midpoint"
        assert interpolator.fractional_calls == 2
        assert interpolator.midpoint_calls == 1
    finally:
        pipeline.stop()


@pytest.mark.parametrize("generated_frames", (0, 5))
def test_rife_generation_amount_rejects_out_of_range_values(generated_frames: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 4"):
        AsyncFramePipeline(object(), generated_frames=generated_frames)


def test_rife_failure_immediately_falls_back_to_current_frame(caplog) -> None:
    interpolator = _MidpointInterpolator(fail=True)

    class IdentityProcessor:
        def process(self, frame):
            return frame

    pipeline = AsyncFramePipeline(
        IdentityProcessor(),
        frame_interpolator=interpolator,
    )
    pipeline.start()
    try:
        pipeline.submit("A")
        assert pipeline.take_latest(timeout=1.0).image == "A"
        with caplog.at_level("WARNING"):
            pipeline.submit("B")
            fallback = pipeline.take_latest(timeout=1.0)

        assert fallback is not None
        assert fallback.image == "B"
        assert fallback.frame_kind == "processed"
        assert pipeline.interpolation_failures == 1
        assert pipeline.interpolated_frames == 0
        assert "presenting the processed frame" in caplog.text
        pipeline.raise_if_failed()
    finally:
        pipeline.stop()


def test_frame_generation_reserves_one_generated_batch_and_keeps_newest_successor() -> None:
    interpolator = _MidpointInterpolator()

    class IdentityProcessor:
        def process(self, frame):
            return frame

    pipeline = AsyncFramePipeline(
        IdentityProcessor(),
        frame_interpolator=interpolator,
    )
    pipeline.start()
    try:
        for index, frame in enumerate(("A", "B", "C", "D"), start=1):
            pipeline.submit(frame)
            _wait_for_processed_frames(pipeline, index)

        middle = pipeline.take_latest(timeout=1.0)
        current = pipeline.take_latest(timeout=1.0)
        newest_middle = pipeline.take_latest(timeout=1.0)
        newest_current = pipeline.take_latest(timeout=1.0)

        # B's midpoint was already admitted to the output handoff, so C and D
        # cannot erase it before the pacer gets one chance to present it.
        assert middle is not None and middle.image == "middle(A,B)"
        assert current is not None and current.image == "B"
        # The single successor slot still uses newest-frame semantics: C's
        # complete batch is replaced by D's complete batch.
        assert newest_middle is not None and newest_middle.image == "middle(C,D)"
        assert newest_current is not None and newest_current.image == "D"
        assert pipeline.result_replacements == 3
        assert pipeline.dropped_generated_frames == 1
    finally:
        pipeline.stop()


def test_presentation_batch_dequeue_is_atomic_against_publisher() -> None:
    """A producer cannot replace an endpoint after its midpoint was dequeued."""

    interpolator = _MidpointInterpolator()

    class IdentityProcessor:
        def process(self, frame):
            return frame

    pipeline = AsyncFramePipeline(
        IdentityProcessor(),
        frame_interpolator=interpolator,
    )
    pipeline.start()
    try:
        pipeline.submit("A")
        _wait_for_processed_frames(pipeline, 1)
        assert pipeline.take_presentation_batch(timeout=1.0)[0].image == "A"

        pipeline.submit("B")
        _wait_for_processed_frames(pipeline, 2)
        batch = pipeline.take_presentation_batch(timeout=1.0)

        assert [result.image for result in batch] == ["middle(A,B)", "B"]
        assert len({result.sequence_id for result in batch}) == 1
    finally:
        pipeline.stop()


def _presentation_result(sequence_id: int, kind: str, captured_at: float = 0.0):
    return ProcessedFrame(
        sequence_id=sequence_id,
        image=f"{kind}-{sequence_id}",
        captured_at=captured_at,
        processing_started_at=captured_at,
        processed_at=captured_at,
        capture_ms=0.0,
        frame_kind=kind,
    )


@pytest.mark.parametrize("generated_count", (1, 2, 3, 4))
def test_output_handoff_is_bounded_and_preserves_batch_sequence_integrity(
    generated_count: int,
) -> None:
    pipeline = AsyncFramePipeline(
        object(), generated_frames=generated_count, clock=lambda: 0.0
    )

    for sequence_id in range(100):
        pipeline._publish_results([
            *(
                _presentation_result(sequence_id, "generated")
                for _ in range(generated_count)
            ),
            _presentation_result(sequence_id, "processed"),
        ])
        assert pipeline.queue_sizes().output <= 2 * (generated_count + 1)

    oldest = pipeline.take_presentation_batch()
    newest = pipeline.take_presentation_batch()
    assert {result.sequence_id for result in oldest} == {0}
    assert {result.sequence_id for result in newest} == {99}
    assert pipeline.result_replacements == 98 * (generated_count + 1)
    assert pipeline.dropped_generated_frames == 98 * generated_count


def test_stalled_consumer_discards_whole_stale_reserved_batch() -> None:
    now = [0.0]
    pipeline = AsyncFramePipeline(
        object(),
        output_max_age_ms=50.0,
        clock=lambda: now[0],
    )
    pipeline._publish_results([
        _presentation_result(1, "generated", captured_at=0.0),
        _presentation_result(1, "processed", captured_at=0.0),
    ])
    pipeline._publish_results([
        _presentation_result(2, "generated", captured_at=0.09),
        _presentation_result(2, "processed", captured_at=0.09),
    ])

    now[0] = 0.10
    selected = pipeline.take_presentation_batch()

    assert {result.sequence_id for result in selected} == {2}
    assert pipeline.stale_reserved_batches == 1
    assert pipeline.result_replacements == 2
    assert pipeline.dropped_generated_frames == 1


def test_real_only_output_retains_newest_frame_semantics() -> None:
    pipeline = AsyncFramePipeline(object())
    for sequence_id in range(100):
        pipeline._publish_results([
            _presentation_result(sequence_id, "processed")
        ])

    selected = pipeline.take_presentation_batch()
    assert [result.sequence_id for result in selected] == [99]
    assert pipeline.result_replacements == 99
    assert pipeline.dropped_generated_frames == 0


@pytest.mark.parametrize("invalid", (True, 0, -1, float("nan"), float("inf"), "100"))
def test_output_max_age_rejects_invalid_values(invalid) -> None:
    with pytest.raises(ValueError, match="maximum age"):
        AsyncFramePipeline(object(), output_max_age_ms=invalid)


def test_frame_generation_shutdown_releases_workers_and_previous_frame() -> None:
    interpolator = _MidpointInterpolator()

    class IdentityProcessor:
        def process(self, frame):
            return frame

    pipeline = AsyncFramePipeline(
        IdentityProcessor(),
        frame_interpolator=interpolator,
    )
    pipeline.start()
    pipeline.submit("A")
    assert pipeline.take_latest(timeout=1.0) is not None

    pipeline.stop(timeout=1.0)
    interpolator.shutdown()

    assert pipeline.worker_threads_alive is False
    assert pipeline._previous_processed_image is None
    assert interpolator.shutdown_calls == 1
