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
