# IRIS research framing

## The problem

Frame interpolation can make a 30 FPS application look smoother, but an
external tool receives only completed pixels. It does not receive the motion
vectors, depth, exposure, or timing information available inside a game
engine. Capture frames also arrive irregularly. A uniform neural pass can
therefore corrupt text and uncertain motion, finish after its display deadline,
or report close to 60 presentation calls while viewers still see repeated
33--50 ms gaps.

This matters beyond one game. The same limitation applies to older software,
cloud gaming, remote desktops, recorded media, and other applications that
cannot be modified to integrate a vendor-specific renderer feature.

## Research question

Under a fixed latency and compute budget, can confidence-aware regional
routing plus bounded deadline-aware playout produce more evenly timed,
visually valid frames than uniform whole-frame interpolation in a
renderer-independent Windows pipeline?

## Hypothesis

Compared with uniform interpolation and the previous newest-batch scheduler,
the proposed system will:

1. reduce the frequency of displayed gaps above one refresh interval;
2. preserve real source endpoints and HUD/text regions more reliably;
3. keep added capture-to-presentation latency within a declared bound; and
4. expose which mechanism caused each change through ablations and frame
   provenance.

## What is original here

RIFE, IFRNet, OpenCV, Windows Graphics Capture, ONNX Runtime, and PresentMon
are dependencies, not claimed inventions. The research contribution is their
control system:

> A provenance-instrumented controller that jointly decides where
> interpolation is reliable, whether optional work can meet its deadline, and
> which real or generated candidate should occupy each bounded playout slot.

- each region is routed according to motion, detail, text/HUD likelihood,
  confidence, and deadline;
- uncertain regions fall back to stable real or spatially processed pixels;
- a bounded playout reservoir prioritizes newer real endpoints and may use a
  cheap late-interval reserve candidate for bounded jitter concealment;
- every output records its route, provenance, timing, buffer depth, and drop
  reason so components can be ablated separately.

## Scientific and practical value

The project studies a general systems question: how should limited compute be
allocated when useful visual work has different cost, confidence, and
deadlines? The resulting measurements can show when generated frames are
useful, when a stable fallback is safer, and when a high software FPS counter
does not imply smooth visible output. The practical result is an open,
renderer-independent reference implementation for applications that cannot
provide engine data or native frame-generation support.

## Required evidence

Use identical clips or controlled scenes and at least three repetitions per
condition. Report confidence intervals across scenes and hardware; do not pool
individual frames as if they were independent experiments.

The minimum ablation is:

| Condition | Regional routing | Deadline admission | Bounded playout | Reserve candidate |
|---|---:|---:|---:|---:|
| A: real-frame control | no | no | no | no |
| B: uniform interpolation | no | no | no | no |
| C: regional routing | yes | yes | no | no |
| D: complete proposed method | yes | yes | yes | yes |

Measure displayed interval p50/p95/p99, gaps above 1.5 refresh intervals,
real and generated endpoints observed, capture-to-submission latency, PSNR,
SSIM, temporal error, HUD corruption, route shares, buffer depth, and every
drop reason. PresentMon is supporting evidence; OBS marker decoding or an
external camera is required for claims about visible unique frames. Input-to-
photon latency requires a physical event and optical measurement and must not
be inferred from internal timestamps.

## Current result boundary

The existing 60 Hz OBS run demonstrated 56.6 distinct visible events per
second, but its p95 and p99 visible gaps were 33.3 ms. It is therefore valid
throughput evidence and a failed smoothness result. The buffered scheduler is
an experimental response to that measured failure; it must outperform the
previous scheduler in repeated visible-output trials before it is presented as
an improvement.

The reserve candidate is not a second neural reconstruction. It blends the
midpoint toward the following real endpoint. When used, it trades extra delay
and nonuniform motion timing for one additional distinct image during a short
source-delivery gap. Evaluation must therefore report motion-time error and
added latency as well as apparent smoothness.
