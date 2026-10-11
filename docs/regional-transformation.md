# Regional transformation research mode

Regional transformation is an opt-in experimental path for explaining and
controlling where frame interpolation is used. It is disabled by default so
the stable baseline remains directly comparable.

## Current pipeline

1. Downscale the two real endpoint frames to the model resolution.
2. Measure motion, motion consistency, edge density, and a conservative
   text/HUD proxy on a tile grid.
3. Route each tile to static detail, smooth motion, unreliable motion,
   HUD/text, low detail, or scene cut.
4. Rank eligible smooth-motion tiles by expected benefit, confidence, and the
   measured cost estimate.
5. Skip neural inference when eligible coverage is too sparse, confidence-weighted
   coverage is too small, or the learned end-to-end transform cost cannot fit
   the remaining deadline. Warmup excludes its first graph-compilation sample
   and seeds a conservative cost estimate so the first live frame cannot cause
   an avoidable neural stall. Otherwise run the interpolator once and accept its
   pixels only in confident smooth-motion regions.
6. When a neural pass is rejected, use one consistent midpoint blend for the
   frame; scene cuts use the newest real endpoint. Avoiding tile-by-tile endpoint
   replacement removes visible block boundaries and route-switch flicker. When
   the model is admitted, uncertain model pixels still use the stable fallback.
7. Attach per-tile provenance to the generated frame for telemetry and the
   optional colour-coded debug view. Midpoint blends, real endpoints,
   interpolated pixels, mixed blends, and deadline fallbacks remain distinct;
   `spatial` is reserved for a real spatial upscaler. Telemetry also records the
   admission reason (`model`, `no_eligible_tiles`, `sparse_coverage`,
   `low_confidence_coverage`, or `deadline`) so each speedup is attributable.

The current ONNX models still operate on a complete low-resolution image when
any tile is selected. Therefore the deadline allocator can avoid a whole model
call, but it does not yet reduce the cost of a model call by processing only
selected tiles. Region-cropped or feature-reuse inference is a later research
stage and must be evaluated separately.

## GUI controls

Select a RIFE-family frame-generation model, then enable **Regional
transformation**. **Regional deadline** sets the analysis and allocation budget
in milliseconds. **Provenance view** colours the output by origin and is for
debugging, not normal visual-quality tests.

## Reproducible offline evaluation

`benchmarks/evaluate_regional_video.py` withholds every middle frame from a
video and compares four outputs: endpoint midpoint, raw interpolation, the
legacy stabilizer, and regional transformation. It writes per-frame CSV data,
a JSON summary, and sample images. Report PSNR, SSIM, temporal error, processing
time, every route count, and every provenance count together.

`benchmarks/evaluate_live_regional.py` exercises the exact integrated runtime
path and reports end-to-end transform time, p95 time, estimated processing
throughput, model-use fraction, admission reason, reconstruction metrics, and
temporal error. On a difficult consecutive 300-sample section of the attached
1218x720 Subnautica: Below Zero recording, the 16 ms configuration measured
3.32 ms mean and 4.14 ms p95 after warmup. All neural passes were rejected: 286
frames reported `deadline`, 13 reported `no_eligible_tiles`, and one reported
`sparse_coverage`. The output matched the midpoint baseline within measurement
rounding and still showed midpoint motion doubling on fast foreground objects.
These results validate a stable fallback, not improved interpolation or display
scanout. A live capture/presentation test is still required.

## Two insertions per source interval

`benchmarks/evaluate_two_insertions.py` evaluates equally spaced insertions at
1/3 and 2/3 between real frames. The checked-in RIFE exports accept only two
endpoint images and cannot take a timestep, so they cannot directly produce
these two positions. The prototype uses bidirectional OpenCV Farneback flow
with the regional confidence gate; uncertain tiles fall back to a temporal
blend or the nearer real endpoint. It reports which route each insertion took,
latency, exact image hashes, and PSNR/SSIM against withheld frames. A hash only
proves the generated arrays are not byte-identical; it does not establish that
all pixels contain correct motion or that a display actually scanned them out.

On the 240-pair Subnautica: Below Zero clip run (59.87 fps source, sampled as
19.96 source intervals/s), 480 insertions were produced and all 480 differed
within their pairs. The regional flow path was used for 43.96% of inserts and
the stable fallback for 56.04%. Mean generated latency was 33.33 ms (p95
79.79 ms); mean pair latency was 66.69 ms (p95 152.36 ms). Maintaining the
sampled source cadence requires finishing each pair within 50.11 ms; only
56.1% of measured pairs met that deadline. The estimated serial output rate
was 44.98 frames/s, so this prototype did not demonstrate stable 60 fps.

Against withheld frames, mean PSNR was 34.44/33.98 dB at 1/3 and 2/3,
compared with 34.35/34.07 dB for a plain temporal blend. Mean SSIM was
0.9779/0.9757 compared with 0.9814/0.9796 for the blend. The small, mixed
differences do not establish a general quality improvement. This test is a
single lossy gameplay clip, and its offline rate does not measure interactive
presentation, unique scanout, or input latency. The machine-local JSON and CSV
results are retained under `output/two_insertions_subnautica_final/`; do not
compare runs taken under different system load as a controlled ablation.

This experiment measures offline reconstruction quality and processing time.
It does not measure display scanout, distinct displayed frames, or
input-to-photon latency, and its results must not be described as proof of
real-time presentation.

## Required ablations

Keep the same source clips, frame indices, model, model resolution, provider,
and hardware for each comparison. Report at least:

- stable baseline with regional mode disabled;
- analysis and provenance only;
- pre-inference eligibility gate;
- confidence composition without deadline allocation;
- full regional mode;
- full regional mode with provenance rendering enabled only as a diagnostic.

Use several lossless or high-bitrate clips covering static scenes, camera
motion, moving objects, particles, HUD/text, and scene cuts. Publish sample
counts, confidence intervals, random seeds where applicable, driver version,
GPU/CPU model, utilization, memory use, and power-measurement method.
