from __future__ import annotations

import threading
import time

import pytest

from frame_pacing import FramePacer, PresentationFrame


class FakeClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value
        self.waits: list[float] = []

    def __call__(self) -> float:
        return self.value

    def wait(self, delay: float) -> bool:
        self.waits.append(delay)
        self.value += delay
        return False


def frame(
    value: str,
    captured_at: float,
    kind: str = "real",
    sequence_id: int | None = None,
) -> PresentationFrame:
    return PresentationFrame(value, captured_at, kind, sequence_id)


def test_fixed_scheduling_is_monotonic_and_evenly_spaced() -> None:
    clock = FakeClock()
    pacer = FramePacer(mode="fixed", target_fps=20, clock=clock, waiter=clock.wait)
    decisions = []
    for index in range(4):
        decisions.extend(pacer.pace_batch([frame(str(index), 100.0 + index * 0.05)]))
    scheduled = [decision.scheduled_at for decision in decisions]
    assert scheduled == pytest.approx([100.0, 100.05, 100.10, 100.15])
    assert all(right > left for left, right in zip(scheduled, scheduled[1:]))


def test_midpoint_ordering_and_even_spacing() -> None:
    clock = FakeClock()
    pacer = FramePacer(mode="fixed", target_fps=60, clock=clock, waiter=clock.wait)
    first = pacer.pace_batch([frame("A", 100.0)])[0]
    batch = pacer.pace_batch(
        [frame("M", 100.01, "generated"), frame("B", 100.01)]
    )
    assert [decision.frame.payload for decision in [first, *batch]] == ["A", "M", "B"]
    assert [decision.scheduled_at for decision in [first, *batch]] == pytest.approx(
        [100.0, 100.0 + 1 / 60, 100.0 + 2 / 60]
    )


def test_late_generated_frame_is_dropped_before_ready_real_frame() -> None:
    clock = FakeClock()
    pacer = FramePacer(mode="fixed", target_fps=10, clock=clock, waiter=clock.wait)
    pacer.pace_batch([frame("A", 100.0)])
    clock.value = 100.21
    decisions = pacer.pace_batch(
        [frame("M", 100.20, "generated"), frame("B", 100.20)]
    )
    assert decisions[0].present is False
    assert decisions[0].reason == "generated_late"
    assert decisions[1].present is True
    assert pacer.snapshot().generated_frames_dropped_late == 1


@pytest.mark.parametrize("generated_count", (1, 2, 3, 4))
def test_late_batch_is_reanchored_when_latency_budget_can_preserve_generation(
    generated_count: int,
) -> None:
    clock = FakeClock()
    pacer = FramePacer(
        mode="fixed",
        target_fps=60,
        max_frame_latency_ms=1000,
        clock=clock,
        waiter=clock.wait,
    )
    pacer.pace_batch([frame("A", 100.0)])
    clock.value = 100.3

    decisions = pacer.pace_batch([
        *(frame(f"G{index}", 100.3, "generated") for index in range(generated_count)),
        frame("B", 100.3),
    ])

    assert all(decision.present for decision in decisions)
    assert decisions[0].scheduled_at == pytest.approx(100.3)
    assert decisions[-1].scheduled_at == pytest.approx(
        100.3 + generated_count / 60
    )
    snapshot = pacer.snapshot()
    assert snapshot.presented_generated_frames == generated_count
    assert snapshot.presented_real_frames == 2
    assert snapshot.generated_frames_dropped_late == 0


def test_maximum_latency_drops_real_and_generated_output() -> None:
    clock = FakeClock(2.0)
    pacer = FramePacer(
        mode="fixed",
        target_fps=60,
        max_frame_latency_ms=25,
        clock=clock,
        waiter=clock.wait,
    )
    decisions = pacer.pace_batch(
        [frame("M", 1.0, "generated"), frame("B", 1.0)]
    )
    assert not any(decision.present for decision in decisions)
    snapshot = pacer.snapshot()
    assert snapshot.generated_frames_dropped_late == 1
    assert snapshot.stale_generated_frames_dropped == 0
    assert snapshot.real_frames_dropped_late == 1


def test_stale_generated_frame_is_identified_separately() -> None:
    clock = FakeClock(2.0)
    pacer = FramePacer(
        mode="off",
        max_frame_latency_ms=25,
        clock=clock,
        waiter=clock.wait,
    )

    decision = pacer.pace_batch([frame("M", 1.0, "generated")])[0]

    assert not decision.present
    assert decision.reason == "stale_generated"
    assert pacer.snapshot().stale_generated_frames_dropped == 1


def test_pacing_off_preserves_uncapped_behavior_without_waits() -> None:
    clock = FakeClock()
    pacer = FramePacer(mode="off", clock=clock, waiter=clock.wait)
    for index in range(3):
        decision = pacer.pace_batch([frame(str(index), clock.value)])[0]
        assert decision.present
    assert clock.waits == []


def test_pacing_off_presents_generated_and_real_frames() -> None:
    clock = FakeClock()
    pacer = FramePacer(mode="off", clock=clock, waiter=clock.wait)

    decisions = pacer.pace_batch(
        [frame("M", clock.value, "generated"), frame("B", clock.value)]
    )

    assert [decision.present for decision in decisions] == [True, True]
    snapshot = pacer.snapshot()
    assert snapshot.presented_generated_frames == 1
    assert snapshot.presented_real_frames == 1
    assert snapshot.generated_frames_dropped_late == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_fps": 0},
        {"target_fps": -1},
        {"target_fps": float("nan")},
        {"max_frame_latency_ms": 0},
        {"mode": "fixed"},
        {"mode": "invalid"},
    ],
)
def test_invalid_pacing_configuration_is_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        FramePacer(**kwargs)


def test_shutdown_interrupts_timing_wait_immediately() -> None:
    shutdown = threading.Event()
    pacer = FramePacer(
        mode="fixed",
        target_fps=1,
        max_frame_latency_ms=2000,
        shutdown_event=shutdown,
    )
    pacer.pace_batch([frame("A", time.perf_counter())])
    result = []
    worker = threading.Thread(
        target=lambda: result.extend(
            pacer.pace_batch([frame("B", time.perf_counter())])
        )
    )
    worker.start()
    time.sleep(0.02)
    pacer.stop()
    worker.join(timeout=0.5)
    assert not worker.is_alive()
    assert result and result[0].reason == "shutdown"


def test_waiting_uses_one_blocking_wait_and_never_busy_spins() -> None:
    clock = FakeClock()
    pacer = FramePacer(mode="fixed", target_fps=30, clock=clock, waiter=clock.wait)
    pacer.pace_batch([frame("A", clock.value)])
    pacer.pace_batch([frame("B", clock.value)])
    assert len(clock.waits) == 1
    assert clock.waits[0] == pytest.approx(1 / 30)


def test_auto_mode_remains_uncapped_until_source_rate_is_reliable() -> None:
    clock = FakeClock()
    pacer = FramePacer(mode="auto", clock=clock, waiter=clock.wait)
    for index in range(5):
        clock.value = 100.0 + index / 30
        pacer.pace_batch([frame(str(index), clock.value)])
    assert clock.waits == []
    assert pacer.snapshot().estimated_source_fps == pytest.approx(30.0)


def test_source_rate_accounts_for_newest_frame_sequence_gaps() -> None:
    clock = FakeClock()
    pacer = FramePacer(mode="auto", clock=clock, waiter=clock.wait)
    for index in range(6):
        sequence_id = index * 4
        clock.value = 100.0 + sequence_id / 60
        pacer.pace_batch([
            frame(str(index), clock.value, sequence_id=sequence_id)
        ])

    assert pacer.snapshot().estimated_source_fps == pytest.approx(60.0)


def test_auto_mode_adds_one_slot_only_when_generated_cadence_becomes_reliable() -> None:
    clock = FakeClock()
    pacer = FramePacer(mode="auto", clock=clock, waiter=clock.wait)
    for index in range(5):
        clock.value = 100.0 + index / 30
        pacer.pace_batch([frame(str(index), clock.value)])

    clock.value = 100.0 + 5 / 30
    decisions = pacer.pace_batch([
        frame("M", clock.value, "generated"),
        frame("B", clock.value),
    ])

    assert decisions[0].present
    assert decisions[1].present
    assert clock.waits[-2:] == pytest.approx([1 / 60, 1 / 60])
    assert decisions[1].actual_at - decisions[0].actual_at == pytest.approx(1 / 60)


def test_four_generated_frames_are_evenly_paced_and_counted_by_kind() -> None:
    clock = FakeClock()
    pacer = FramePacer(mode="fixed", target_fps=150, clock=clock, waiter=clock.wait)
    decisions = pacer.pace_batch(
        [
            frame("G1", clock.value, "generated"),
            frame("G2", clock.value, "generated"),
            frame("G3", clock.value, "generated"),
            frame("G4", clock.value, "generated"),
            frame("B", clock.value),
        ]
    )

    assert all(decision.present for decision in decisions)
    snapshot = pacer.snapshot()
    assert snapshot.presented_generated_frames == 4
    assert snapshot.presented_real_frames == 1
    assert snapshot.presented_frames == 5


def test_live_pacing_yields_before_waiting_for_next_frame():
    clock = FakeClock()
    pacer = FramePacer(mode="fixed", target_fps=60, clock=clock, waiter=clock.wait)
    decisions = pacer.iter_pace_batch([
        frame("M", 100.0, "generated"), frame("B", 100.0)])
    first = next(decisions)
    assert first.present and first.frame.payload == "M"
    assert clock.waits == []
    second = next(decisions)
    assert second.present and second.frame.payload == "B"
    assert clock.value - first.actual_at == pytest.approx(1 / 60)


def test_gui_overload_keeps_latest_real_but_drops_late_generated():
    clock = FakeClock(100.5)
    pacer = FramePacer(mode="fixed", target_fps=60, clock=clock,
                      waiter=clock.wait, keep_latest_real=True)
    decisions = pacer.pace_batch([
        frame("M", 100.0, "generated"), frame("B", 100.0)])
    assert not decisions[0].present
    assert decisions[1].present
    assert clock.waits == []
    assert pacer.snapshot().presented_real_frames == 1


def test_vsynced_presenter_owns_cadence_without_double_dropping_generation():
    clock = FakeClock(100.0)
    pacer = FramePacer(
        mode="fixed",
        target_fps=60,
        max_frame_latency_ms=100,
        clock=clock,
        waiter=clock.wait,
        keep_latest_real=True,
        presenter_controls_cadence=True,
    )
    pacer.pace_batch([frame("A", 100.0)])

    # Simulate the caller spending several refresh intervals inside Present.
    # The software schedule is now historically late, but the fresh midpoint
    # must still reach the swap chain instead of being discarded twice.
    clock.value = 100.05
    decisions = pacer.iter_pace_batch([
        frame("M", 100.04, "generated"),
        frame("B", 100.04),
    ])
    midpoint = next(decisions)
    assert midpoint.present
    assert midpoint.frame.payload == "M"
    clock.value += 1 / 60
    real = next(decisions)
    assert real.present
    assert real.frame.payload == "B"
    assert clock.waits == []
    snapshot = pacer.snapshot()
    assert snapshot.generated_frames_dropped_late == 0
    assert snapshot.target_interval_ms == pytest.approx(1000 / 60)


def test_vsynced_presenter_still_rejects_stale_generated_frame():
    clock = FakeClock(100.2)
    pacer = FramePacer(
        mode="fixed",
        target_fps=60,
        max_frame_latency_ms=100,
        clock=clock,
        waiter=clock.wait,
        presenter_controls_cadence=True,
    )

    decision = pacer.pace_batch([frame("M", 100.0, "generated")])[0]

    assert not decision.present
    assert decision.reason == "stale_generated"
