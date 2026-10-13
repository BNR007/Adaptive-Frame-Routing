# Buffered playout experiment status

## Implementation

The experimental `IRIS Buffered Playout 720p30` profile uses:

- one normal midpoint from the configured interpolator;
- one cheap 0.75 candidate made by blending that midpoint toward the following
  real endpoint;
- a three-frame startup reservoir with a four-frame hard bound;
- optional-reserve-first, midpoint-second, oldest-real-last eviction;
- complete queue clearing after capture or inference recovery;
- rebuffering after an underrun;
- candidate-specific method, provenance, content-time, buffer-depth, wait, and
  drop telemetry.

The reserve is bounded jitter concealment. It is not a second neural inference
and it does not reconstruct uniformly spaced motion when inserted.

## Automated validation

- 152 targeted pipeline, pacing, UI, shutdown, and research tests passed before
  the final telemetry additions.
- 78 directly affected pacing, pipeline, research, and CLI tests passed after
  the final additions.
- A deterministic 15-second source-jitter test had no empty 60 Hz software
  slots while maintaining the four-frame bound.

## Subnautica recording results

The five-second software scheduler ablation with real RIFE processing preserved
150/150 real endpoints, reported no output-queue replacements, and reached
59.91 software presentation calls/s. However p95 intervals remained about
30.6 ms, so this is not evidence of smooth visible 60 FPS.

With a 50 ms source-delivery stall, the buffer increased median frame age from
about 44 ms to 61 ms without improving the software-paced p95 gap. This is a
measured latency cost, not a positive smoothness result.

On 60 withheld Subnautica samples, the 0.75 reserve had median 32.01 dB PSNR and
0.975 SSIM. It beat a repeated midpoint by PSNR in 51.7% of samples and a
repeated endpoint in 56.7%. Its pixels were almost equivalent to a direct 0.75
linear blend in this low-resolution regional configuration. The reserve is
therefore plausible as an occasional concealment frame, but it has not shown a
quality advantage strong enough for continuous use.

## Decision gate

Do not merge or claim an improvement until a paired live 60 Hz trial compares
the existing `IRIS Research 720p30` profile with `IRIS Buffered Playout
720p30`, using the same scene and OBS marker analysis. Keep the buffered method
only if it materially reduces 33.3/50 ms visible gaps without unacceptable
latency, time-warp, ghosting, or HUD damage.

## 2026-10-13 live checkpoint

A 60 Hz Subnautica: Below Zero run using the buffered profile was subjectively
reported as "pretty smooth." The source remained near 30 FPS while the scheduler
released about 60 frames/s. The completed research summary reported 59.90 preview
calls/s, 59.93 exact-unique submissions/s, and 59.60 meaningfully different
submissions/s. Scheduler p95 gaps were approximately 17.5--17.8 ms after startup.

This run did not validate neural-frame quality: the neural worker submitted 1,528
jobs, but the confidence gate accepted and applied none. Smoothness therefore came
from buffered cadence and the spatial midpoint fallback. Median frame age was
approximately 88--92 ms and p95 age was approximately 102--113 ms.

The first PresentMon summary incorrectly reported 29.92 display events/s because
it averaged per-stream frame durations after PresentMon divided the same process's
events between a DXGI swap-chain category and an `Other` category. Reading the raw
event timestamps in chronological order gives 59.81 process display events/s,
with a 16.67 ms median event gap and a 17.14 ms p95 gap. The parser now reports
timestamp-derived process totals and retains both per-stream breakdowns so this
case remains auditable. This passes the provisional 50-event/s external telemetry
gate, although PresentMon remains process-level ETW evidence rather than optical
proof of unique visible frames.
