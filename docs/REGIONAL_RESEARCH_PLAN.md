# URFTS regional transformation research plan

## Clean starting point

Development starts from commit `76a864f` (`Save stable IFRNet 60 FPS
baseline`) on branch `codex/regional-transform-v1`. The GPU-resident, OBS,
QuickSR, and later presentation experiments remain preserved on the original
branches but are not part of this baseline.

Baseline verification on 10 October 2026: 330 tests passed and 3 were skipped.

The research implementation must not bundle independent mechanisms behind one
mode. Every mechanism has its own configuration switch, trace field, unit test,
and ablation case.

## Research question

Can a renderer-independent system use regional uncertainty and deadline-aware
processing to improve already-rendered frames while avoiding the cost and
temporal instability of full-frame transformation?

The paper must distinguish software presentation submission, display events,
visible unique frames, and input-to-photon latency. None of these terms may be
used as a substitute for another.

## Phase 0: measurement foundation

This is implemented before the new algorithm so the stable baseline and every
later phase use identical instrumentation.

### Per-frame identity and trace

Assign every captured real frame a monotonically increasing source ID. A
generated frame records both real parent IDs and its interpolation position.
For every output record:

- capture, queue, processing-start, processing-end, submission, and drop times;
- source and parent IDs;
- queue depth and exact drop/fallback reason;
- route-area percentages and predicted/actual processing cost;
- whether the output is real, generated, blended, repeated, or fallback.

All pipeline timestamps use one monotonic software clock. Raw traces are saved,
not only aggregates.

### Display-event measurement

Integrate PresentMon/ETW as a separate observer where supported. Record present
mode, application-present events, displayed events, dropped presents, and
display cadence. These measurements describe the Windows presentation path;
they do not prove photon emission.

### Visible unique-frame measurement

Add an optional measurement border containing a 16-bit Gray-code frame ID and
a checksum. Film the display at 240 fps or faster with fixed exposure. The
offline decoder reports:

- visible unique frames per second;
- repeated-frame rate;
- missing-ID rate;
- camera samples occupied by each frame;
- disagreement between submitted and visibly decoded IDs.

The marker is enabled only for measurement and is excluded from quality
metrics.

### Input-to-photon measurement

Use a USB microcontroller or equivalent rig that triggers an input and an LED
on the same event. A high-speed camera films the LED and a screen response patch
together. Prefer a photodiode for higher precision. Report median, p95, p99,
sample count, camera frame interval, and measurement uncertainty for:

1. source application without URFTS;
2. URFTS passthrough;
3. each regional algorithm phase.

Software timestamps may supplement this experiment but may not replace it.

### Hardware manifest and resource sampling

Every run records exact CPU, GPU adapters, dedicated/shared memory, RAM, Windows
build, display resolution and refresh, driver version, power mode, Python,
OpenCV, ONNX Runtime, provider, model SHA-256, input SHA-256, code revision, and
dirty-worktree state. During a run sample CPU, RAM, GPU-engine utilization,
dedicated/shared GPU memory, clocks, temperature, and power when exposed by the
hardware. Unsupported values are `unavailable`, never zero.

## Phase 1: per-region analysis and routing

Compute features at one-eighth resolution and aggregate them to configurable
64x64 or 128x128 output tiles:

- pixel change;
- edge density;
- motion magnitude and consistency;
- forward/backward occlusion indication;
- text/HUD likelihood;
- previous reconstruction error;
- temporal flicker;
- scene-cut probability.

The first implementation is diagnostic: it produces feature maps and route
labels while the presented output remains identical to the baseline. This
isolates analysis overhead and validates masks before routing changes pixels.

Independent switch: `regional_analysis`.

Exit criteria:

- deterministic maps for fixed frame pairs;
- no output-pixel change in diagnostic mode;
- per-feature and total analysis cost reported;
- labelled validation scenes for HUD, camera motion, particles, and cuts.

## Phase 2: confidence-controlled composition

Construct a stable spatial fallback `S`, candidate interpolation `I`, and
confidence map `C`:

`O = C * I + (1 - C) * S`.

Confidence uses motion agreement, occlusion indication, brightness/colour
agreement, edge stability, HUD probability, and recent error. Tile boundaries
use overlap and feathering. Hysteresis prevents rapid route changes.

Independent switch: `confidence_composition`.

Exit criteria:

- identity at confidence zero/one;
- no visible tile seams in synthetic boundary tests;
- lower temporal flicker than full-frame interpolation on the fixed dataset;
- separate results for HUD, particles, disocclusions, and scene cuts.

## Phase 3: deadline-aware regional allocation

Maintain rolling cost distributions for each route, device, and tile shape.
For every candidate region calculate:

`utility = expected_quality_gain * confidence / predicted_cost`.

Select work in descending utility order while the conservative predicted cost
fits the remaining deadline. Unselected work uses the stable fallback. A
generated region that cannot arrive in time is never computed merely to be
dropped later.

Independent switch: `deadline_allocation`.

Exit criteria:

- simulated-clock tests prove that selected work never exceeds the configured
  predicted budget;
- predicted and actual cost errors are traced;
- comparison with equal-cost random and motion-only selection;
- p95 frame age improves without hiding quality loss.

## Phase 4: temporal quality feedback

When the next real frame arrives, evaluate earlier generated regions against
that newly available evidence. Update bounded per-route/per-scene statistics,
not neural weights. Feedback adjusts confidence calibration, route priority,
cost estimates, and temporary interpolation suppression.

Independent switch: `temporal_feedback`.

Exit criteria:

- deterministic convergence in recorded replay;
- explicit reset on cuts and capture discontinuities;
- no adaptation leakage between independent benchmark clips;
- report initial, steady-state, and post-cut behaviour separately.

## Phase 5: frame provenance and paper evaluation

Store a compact route map with the following labels:

- real;
- spatially upscaled;
- interpolated;
- confidence blended;
- reused;
- deadline fallback.

Provide an optional colour overlay and export per-frame route-area percentages.
The overlay is a diagnostic view and never participates in quality scoring.

Independent switch: `provenance_overlay`; provenance recording itself remains
available in all research runs.

Exit criteria:

- route maps reproduce compositor decisions exactly;
- route percentages sum to 100 percent;
- measured overhead is reported;
- paper figures can trace visible failures back to their selected route.

## Causal ablation design

The stable commit already contains a coarse temporal-stabilization and
confidence-fallback heuristic. It is prior URFTS behaviour, not part of the new
regional contribution. Therefore the evaluation retains both the shipped
stable configuration and a bare-interpolation control with that legacy
heuristic disabled.

Use the same recorded inputs, model, resolution, provider, queue, and pacing for
all rows:

| Case | Legacy heuristic | Regional analysis | Confidence composition | Deadline allocation | Temporal feedback |
|---|---:|---:|---:|---:|---:|
| B0 shipped stable baseline | on | off | off | off | off |
| B1 bare interpolation | off | off | off | off | off |
| A1 analysis overhead | off | on, diagnostic | off | off | off |
| A2 regional routing | off | on | off | off | off |
| A3 confidence | off | on | on | off | off |
| A4 deadline | off | on | on | on | off |
| A5 feedback | off | on | on | on | on |

Additional controls compare A4 with random selection, motion-only selection,
and the same number of processed tiles without utility ranking. This identifies
whether gains come from doing less work or from the proposed allocation rule.

Report per-scene values and confidence intervals across independent scenes or
runs. Frames from one recording are not treated as independent replicates.

## Minimum paper evidence

- At least 24 independent scenes across three or more applications, including
  slow motion, fast motion, rotation, disocclusion, particles/water, HUD/text,
  faces/video calls, and scene cuts.
- At least three independent repeated runs for performance and five when run
  variance is material.
- PSNR, SSIM, LPIPS, temporal error, route coverage, processing cost, frame age,
  drops, resource usage, display events, visible unique FPS, and
  input-to-photon latency with measurement uncertainty.
- At least two hardware classes if the paper claims hardware generality.
- Raw traces, manifests, input hashes, model hashes, code revision, and analysis
  scripts released with the artifact.

## Claim boundary

Before physical measurement, the system may claim only software throughput and
presentation submission. A real-time or visible-FPS claim requires displayed
event evidence and visible-frame decoding. An input-latency claim requires the
external LED/camera or photodiode experiment.
