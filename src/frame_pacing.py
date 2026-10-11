"""Reusable low-latency presentation scheduling.

The pacer owns only timing policy.  Capture, processing, and rendering remain
independent, and callers retain control of the actual presentation operation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import statistics
import threading
import time
from typing import Any, Callable, Iterable


PACING_MODES = ("auto", "off", "fixed")
DEFAULT_MAX_FRAME_LATENCY_MS = 100.0


@dataclass(frozen=True, slots=True)
class PresentationFrame:
    payload: Any
    captured_at: float
    frame_kind: str = "real"
    source_sequence_id: int | None = None


@dataclass(frozen=True, slots=True)
class PacingDecision:
    frame: PresentationFrame
    present: bool
    scheduled_at: float
    actual_at: float
    pacing_error_ms: float
    reason: str = ""


@dataclass(frozen=True, slots=True)
class PacingSnapshot:
    scheduled_presentation_timestamp: float = 0.0
    actual_presentation_timestamp: float = 0.0
    pacing_error_ms: float = 0.0
    late_frames: int = 0
    generated_frames_dropped_late: int = 0
    stale_generated_frames_dropped: int = 0
    real_frames_dropped_late: int = 0
    estimated_source_fps: float = 0.0
    presentation_fps: float = 0.0
    presented_frames: int = 0
    presented_real_frames: int = 0
    presented_generated_frames: int = 0
    presentation_interval_ms: float = 0.0
    presentation_interval_p95_ms: float = 0.0
    target_interval_ms: float = 0.0
    cadence_misses: int = 0


class SourceRateEstimator:
    """Estimate a stable source cadence from monotonic capture timestamps."""

    def __init__(self, *, window_size: int = 30, minimum_samples: int = 5) -> None:
        if window_size < minimum_samples or minimum_samples < 2:
            raise ValueError("Source-rate estimator window is too small.")
        self._samples: deque[tuple[float, int | None]] = deque(maxlen=window_size + 1)
        self._minimum_samples = minimum_samples

    def add(self, timestamp: float, sequence_id: int | None = None) -> None:
        if self._samples and timestamp <= self._samples[-1][0]:
            return
        self._samples.append((float(timestamp), sequence_id))

    @property
    def fps(self) -> float:
        intervals = self._intervals()
        if not intervals:
            return 0.0
        median = statistics.median(intervals)
        return 1.0 / median if median > 0.0 else 0.0

    @property
    def reliable(self) -> bool:
        intervals = self._intervals()
        if len(intervals) < self._minimum_samples:
            return False
        mean = statistics.mean(intervals)
        if mean <= 0.0:
            return False
        # A deliberately conservative gate keeps irregular or bursty capture
        # uncapped instead of inventing a source rate.
        return statistics.pstdev(intervals) / mean <= 0.20

    def _intervals(self) -> list[float]:
        intervals: list[float] = []
        samples = list(self._samples)
        for (previous_at, previous_id), (current_at, current_id) in zip(
            samples, samples[1:]
        ):
            elapsed = current_at - previous_at
            if elapsed <= 0.0:
                continue
            sequence_distance = 1
            if (
                previous_id is not None
                and current_id is not None
                and current_id > previous_id
            ):
                sequence_distance = current_id - previous_id
            intervals.append(elapsed / sequence_distance)
        return intervals


class FramePacer:
    """Schedule newest presentation batches without busy waiting."""

    def __init__(
        self,
        *,
        mode: str = "auto",
        target_fps: float | None = None,
        max_frame_latency_ms: float = DEFAULT_MAX_FRAME_LATENCY_MS,
        clock: Callable[[], float] = time.perf_counter,
        shutdown_event: threading.Event | None = None,
        waiter: Callable[[float], bool] | None = None,
        keep_latest_real: bool = False,
    ) -> None:
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in PACING_MODES:
            raise ValueError(f"Frame pacing must be one of: {', '.join(PACING_MODES)}.")
        if target_fps is not None and (
            not math.isfinite(float(target_fps)) or float(target_fps) <= 0.0
        ):
            raise ValueError("Target FPS must be a positive finite value.")
        if not math.isfinite(float(max_frame_latency_ms)) or max_frame_latency_ms <= 0.0:
            raise ValueError("Maximum frame latency must be a positive finite value.")
        if normalized_mode == "fixed" and target_fps is None:
            raise ValueError("Fixed frame pacing requires --target-fps.")

        self.keep_latest_real = keep_latest_real
        self.mode = normalized_mode
        self.target_fps = None if target_fps is None else float(target_fps)
        self.max_frame_latency_ms = float(max_frame_latency_ms)
        self._max_latency_seconds = self.max_frame_latency_ms / 1000.0
        self._clock = clock
        self._shutdown_event = shutdown_event or threading.Event()
        self._waiter = waiter or self._shutdown_event.wait
        self._source_rate = SourceRateEstimator()
        self._last_scheduled: float | None = None
        self._presented_at: deque[float] = deque(maxlen=120)
        self._presentation_intervals: deque[float] = deque(maxlen=240)
        self._active_interval: float | None = None
        self._generated_cadence_started = False
        self._scheduled_at = 0.0
        self._actual_at = 0.0
        self._error_ms = 0.0
        self._late_frames = 0
        self._generated_drops = 0
        self._stale_generated_drops = 0
        self._real_drops = 0
        self._presented_frames = 0
        self._presented_real_frames = 0
        self._presented_generated_frames = 0
        self._cadence_misses = 0

    @property
    def shutdown_event(self) -> threading.Event:
        return self._shutdown_event

    def stop(self) -> None:
        self._shutdown_event.set()

    def _interval(self, *, batch_size: int) -> float | None:
        if self.mode == "off":
            return None
        if self.target_fps is not None:
            return 1.0 / self.target_fps
        if not self._source_rate.reliable:
            return None
        source_interval = 1.0 / self._source_rate.fps
        return source_interval / max(1, batch_size)

    def pace_batch(self, frames: Iterable[PresentationFrame]) -> list[PacingDecision]:
        """Collect decisions for callers that do not render live output."""
        return list(self.iter_pace_batch(frames))

    def iter_pace_batch(
        self,
        frames: Iterable[PresentationFrame],
    ) -> Iterable[PacingDecision]:
        """Yield each decision at its presentation slot; render before advancing.

        Pace one newest ordered batch, normally ``[midpoint, real]``."""

        batch = list(frames)
        for frame in batch:
            if frame.frame_kind != "generated":
                self._source_rate.add(frame.captured_at, frame.source_sequence_id)
        interval = self._interval(batch_size=len(batch))

        # Auto pacing starts uncapped while the source cadence is learned.
        # When generated output first becomes paceable, reserve exactly one
        # slot before the first generated frame.  Fixed pacing deliberately
        # keeps its historical first-frame-immediate behaviour.
        transitioning_from_unpaced = (
            self.mode == "auto"
            and not self._generated_cadence_started
            and interval is not None
            and any(frame.frame_kind == "generated" for frame in batch)
        )
        cadence_changed = (
            self.mode == "auto"
            and interval is not None
            and self._active_interval is not None
            and abs(interval - self._active_interval) > self._active_interval * 0.20
            and any(frame.frame_kind == "generated" for frame in batch)
        )
        if transitioning_from_unpaced or cadence_changed:
            self._last_scheduled = self._clock()
        if interval is not None and any(
            frame.frame_kind == "generated" for frame in batch
        ):
            self._generated_cadence_started = True
        self._active_interval = interval

        for index, frame in enumerate(batch):
            now = self._clock()
            if interval is None or self._last_scheduled is None:
                scheduled = now
            else:
                scheduled = self._last_scheduled + interval

            # Interpolation becomes available only after the following real
            # frame has been captured. If a newly ready pair narrowly missed
            # its historical slot, restart that pair from now when the real
            # frame still fits inside the latency budget. This preserves the
            # generated midpoint without emitting a catch-up burst.
            if (
                index == 0
                and frame.frame_kind == "generated"
                and interval is not None
            ):
                late_tolerance = max(0.0015, interval * 0.08)
                following_real = next(
                    (
                        candidate
                        for candidate in batch[index + 1 :]
                        if candidate.frame_kind != "generated"
                    ),
                    None,
                )
                real_deadline = (
                    frame.captured_at + self._max_latency_seconds
                    if following_real is None
                    else following_real.captured_at + self._max_latency_seconds
                )
                remaining_slots = max(1, len(batch) - index - 1)
                if (
                    now - scheduled > late_tolerance
                    and now + remaining_slots * interval < real_deadline
                ):
                    scheduled = now

            # A generated midpoint never gets to hold a ready real frame past
            # its own following presentation slot or latency budget.
            if frame.frame_kind == "generated":
                following_real = next(
                    (candidate for candidate in batch[index + 1 :] if candidate.frame_kind != "generated"),
                    None,
                )
                real_deadline = (
                    frame.captured_at + self._max_latency_seconds
                    if following_real is None
                    else following_real.captured_at + self._max_latency_seconds
                )
                next_real_slot = scheduled + (interval or 0.0)
                late_tolerance = max(0.0015, (interval or 0.0) * 0.08)
                timing_late = interval is not None and (
                    now - scheduled > late_tolerance
                    or now >= scheduled + interval
                )
                if timing_late or (
                    interval is not None and next_real_slot >= real_deadline
                ):
                    yield self._drop(frame, scheduled, now, "generated_late")
                    continue

            preserve_real = self.keep_latest_real and frame.frame_kind != "generated"
            age = now - frame.captured_at
            if age > self._max_latency_seconds and not preserve_real:
                reason = "stale_generated" if frame.frame_kind == "generated" else "maximum_latency"
                yield self._drop(frame, scheduled, now, reason)
                continue

            # Latency wins over cadence.  Present a real frame immediately when
            # waiting for its nominal slot would exceed the configured budget.
            wait_until = scheduled
            if frame.frame_kind != "generated" and scheduled - frame.captured_at > self._max_latency_seconds:
                wait_until = now
            delay = max(0.0, wait_until - now)
            if delay > 0.0 and self._waiter(delay):
                yield self._drop(frame, scheduled, self._clock(), "shutdown")
                continue

            actual = self._clock()
            if actual - frame.captured_at > self._max_latency_seconds and not preserve_real:
                reason = "stale_generated" if frame.frame_kind == "generated" else "maximum_latency"
                yield self._drop(frame, scheduled, actual, reason)
                continue
            error_ms = (actual - scheduled) * 1000.0
            if error_ms > 0.5:
                self._late_frames += 1
            self._record_presented(frame, scheduled, actual, error_ms)

            # Re-anchor after a seriously late real frame so the pacer never
            # emits a catch-up burst of stale presentation slots.
            if frame.frame_kind != "generated" and interval is not None and actual - scheduled > interval:
                self._last_scheduled = actual
            else:
                self._last_scheduled = scheduled

            yield PacingDecision(frame, True, scheduled, actual, error_ms)

    def _drop(
        self,
        frame: PresentationFrame,
        scheduled: float,
        actual: float,
        reason: str,
    ) -> PacingDecision:
        if frame.frame_kind == "generated":
            self._generated_drops += 1
            if reason == "stale_generated":
                self._stale_generated_drops += 1
        else:
            self._real_drops += 1
        return PacingDecision(
            frame=frame,
            present=False,
            scheduled_at=scheduled,
            actual_at=actual,
            pacing_error_ms=(actual - scheduled) * 1000.0,
            reason=reason,
        )

    def _record_presented(
        self,
        frame: PresentationFrame,
        scheduled: float,
        actual: float,
        error_ms: float,
    ) -> None:
        self._scheduled_at = scheduled
        self._actual_at = actual
        self._error_ms = error_ms
        self._presented_at.append(actual)
        if len(self._presented_at) >= 2:
            gap = self._presented_at[-1] - self._presented_at[-2]
            if gap > 0.0:
                self._presentation_intervals.append(gap)
                if self._active_interval is not None and abs(gap - self._active_interval) > max(
                    0.002, self._active_interval * 0.25
                ):
                    self._cadence_misses += 1
        self._presented_frames += 1
        if frame.frame_kind == "generated":
            self._presented_generated_frames += 1
        else:
            self._presented_real_frames += 1

    def snapshot(self) -> PacingSnapshot:
        fps = 0.0
        if len(self._presented_at) >= 2:
            elapsed = self._presented_at[-1] - self._presented_at[0]
            if elapsed > 0.0:
                fps = (len(self._presented_at) - 1) / elapsed
        interval_ms = (
            self._presentation_intervals[-1] * 1000.0
            if self._presentation_intervals
            else 0.0
        )
        interval_p95_ms = 0.0
        if self._presentation_intervals:
            ordered = sorted(self._presentation_intervals)
            interval_p95_ms = ordered[math.ceil(0.95 * len(ordered)) - 1] * 1000.0
        return PacingSnapshot(
            scheduled_presentation_timestamp=self._scheduled_at,
            actual_presentation_timestamp=self._actual_at,
            pacing_error_ms=self._error_ms,
            late_frames=self._late_frames,
            generated_frames_dropped_late=self._generated_drops,
            stale_generated_frames_dropped=self._stale_generated_drops,
            real_frames_dropped_late=self._real_drops,
            estimated_source_fps=self._source_rate.fps,
            presentation_fps=fps,
            presented_frames=self._presented_frames,
            presented_real_frames=self._presented_real_frames,
            presented_generated_frames=self._presented_generated_frames,
            presentation_interval_ms=interval_ms,
            presentation_interval_p95_ms=interval_p95_ms,
            target_interval_ms=(self._active_interval or 0.0) * 1000.0,
            cadence_misses=self._cadence_misses,
        )
