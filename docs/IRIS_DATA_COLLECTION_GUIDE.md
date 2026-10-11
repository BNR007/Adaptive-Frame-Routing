# IRIS data-collection guide

This protocol is designed to fix the weaknesses in the rejected URFTS paper.
Do not start a run until its configuration and hardware manifest are saved.
Do not compare results collected from different commits.

## 1. Freeze the research version

Use repository commit:

`1c402e7dab7f471e525a0d0c017b00ce6fdec34d`

On every machine, clone the Adaptive Frame Routing repository and check out
that exact commit. Record `git status --porcelain`; the benchmark worktree must
be clean. User settings and logs should be copied into the run directory after
each live run rather than committed.

Current limitation: the checkpoint exposes the complete regional mode, but not
every causal mechanism as an independent live GUI switch. Live research runs
now launch PresentMon automatically when it is installed and record a cheap
pipeline-output uniqueness heuristic. They do not yet provide an optical
visible-frame Gray-code measurement or the full 2x2 FIFO/newest scheduler
experiment. Do not call the content-change heuristic physical display proof,
and do not call the bundled regional-versus-baseline comparison a complete
causal ablation.

## 2. Make one folder per machine and run

Use this naming scheme:

`data/raw/<machine_id>/<date>/<application>/<scene_id>/<condition>/run_<01-05>/`

Example:

`data/raw/laptop_low/2026-10-12/subnautica/fast_turn_01/regional/run_01/`

Each run folder should contain:

- `manifest.txt` or `manifest.json`;
- exact command or exported GUI profile;
- source clip name and SHA-256;
- model name and SHA-256;
- raw URFTS log;
- raw per-frame/per-pair CSV and JSON summary when generated;
- PresentMon/ETW trace when available;
- resource trace;
- notes on visible artifacts, interruptions, thermal throttling, or exclusions.

Never overwrite an earlier run. If a run fails, retain it and mark the failure
reason.

## 3. Record the hardware manifest

Save the following before the first run on each machine and again whenever the
driver, Windows build, display, or power mode changes:

- manufacturer and laptop/desktop model;
- CPU name and logical-core count;
- all GPU adapter names;
- dedicated and shared graphics memory;
- installed RAM;
- Windows edition, version, and build;
- GPU driver version and date;
- display resolution and refresh rate;
- AC/battery state and active Windows power mode;
- Python, OpenCV, ONNX Runtime, NumPy, and psutil versions;
- DirectML/CPU provider and numeric device selection;
- code commit and dirty-worktree state;
- model and input-file SHA-256 hashes;
- room/cooling notes and whether the machine was charging.

On Windows, `dxdiag /t dxdiag.txt` captures much of the hardware/driver record.
Also save the output of `powercfg /getactivescheme`, `git rev-parse HEAD`,
`git status --porcelain`, and `python -m pip freeze`. If NVIDIA tooling is
available, retain `nvidia-smi -q`; if it is not available, write
`unavailable`. Do not substitute a guessed value.

## 4. Build the scene dataset

Target at least 24 independent clips across at least three application classes.
Use clips you are legally able to evaluate and describe. Prefer lossless or
high-bitrate capture. Keep the original files unchanged.

Cover these categories with at least three independent scenes each where
possible:

1. static or nearly static view;
2. slow object motion;
3. fast object motion;
4. camera pan/rotation;
5. disocclusion or objects crossing edges;
6. particles, water, smoke, foliage, or transparency;
7. HUD, subtitles, and fine text;
8. abrupt scene cut or menu transition;
9. face/video-call material for the non-game use case;
10. compressed/cloud-video material for the upscaling use case.

For interpolation ground truth, record at 60 FPS or higher. For midpoint tests,
use frames `n` and `n+2` as endpoints and `n+1` as withheld truth. For the
one-third/two-thirds experiment, use `n` and `n+3` as endpoints and `n+1`,
`n+2` as withheld truth. Record clip duration, codec, bitrate, pixel format,
resolution, nominal/decoded frame rate, and SHA-256.

Do not describe hundreds of frames from one clip as hundreds of independent
samples. Aggregate frame metrics within each clip; the clip/scene is the unit
used across scenes.

## 5. Conditions to collect now

### Built-in live research profile

Select **IRIS Research 720p30** in the GUI. This profile freezes the current
tested live configuration: WGC capture, Bicubic spatial scaling, RIFE 4.25
Lite on CPU, one generated midpoint, queue depth 2, temporal stabilization,
and the 34 ms regional deadline. The **Research Run** card then changes only
the causal condition:

- **B Capture/presentation** disables frame generation and regional routing;
- **C Whole-frame FG** enables the same interpolator without regional routing;
- **D Adaptive routing** enables the same interpolator and regional routing.

Condition A is the native application and must be measured with AFR stopped.
Before Start, enter the machine ID, application, scene ID, and unused
repetition number. The recorder performs a 30-second warm-up followed by a
60-second measured interval and stops automatically. It refuses to overwrite
an existing run.

Record Condition A with the standalone observer while AFR is closed. With the
game already running at the scene start, use:

```powershell
& ".\.venv\Scripts\python.exe" .\benchmarks\record_native_baseline.py `
  --process-name "SubnauticaZero.exe" `
  --application "subnautica_below_zero" --scene "scene_01" --repetition 1
```

The observer starts the installed PresentMon console recorder, waits through
the same 30-second warm-up, measures for 60 seconds, samples target-process
CPU/RAM, and records NVIDIA adapter utilization, VRAM, temperature, and power
when those sensors are available. The manifest includes hardware and driver
details. Fill in the generated `game_settings.md` and `observations.md` after
every run. Do not run the AFR GUI or OBS during Condition A.

Each completed run writes `manifest.json`, `telemetry_frames.csv`,
`resources.csv`, `summary.json`, `raw.log`, and `observations.md` beneath
`data/raw/<machine>/<date>/<application>/<scene>/<condition>/run_XX/`. Fill in
`observations.md` immediately after the run. The output directory is ignored
by Git; archive it separately with the paper data.

When PresentMon is found, that same directory also contains `presentmon.csv`,
`presentmon.log`, `presentmon_command.txt`, and `presentmon_summary.json`.
The GUI log prints the folder and a final one-line comparison of preview calls,
content-changing frames, and PresentMon displayed-event rate. If the status is
`no_events`, retain the run but do not use it as display evidence.

Use the same clips and frame indices for every condition.

### Offline quality set

- plain endpoint blend;
- stable regional mode disabled;
- regional mode enabled at fixed budgets (for example 8, 16, and 33 ms);
- each selected interpolation model/provider that is genuinely active;
- the experimental one-third/two-thirds path, reported separately.

Run the offline scripts in `benchmarks/` and retain both their CSV and JSON
outputs. Use at least 120 evaluated intervals per clip when the clip permits it,
but compute final uncertainty across clips, not across frames.

Example PowerShell commands from the repository root (replace the paths and run
ID; keep them identical across paired conditions):

```powershell
$video = "C:\research-inputs\scene_01.mp4"
$model = ".\models\RIFE_v4.25_lite.onnx"
$out = ".\data\raw\machine_id\2026-10-12\application\scene_01\regional\run_01"

& ".\.venv\Scripts\python.exe" .\benchmarks\evaluate_regional_video.py `
  --video $video --model $model --provider cpu `
  --inference-width 160 --inference-height 90 `
  --max-samples 120 --start-frame 0 --sample-stride 1 `
  --output-directory "$out\offline_regional"

& ".\.venv\Scripts\python.exe" .\benchmarks\evaluate_live_regional.py `
  --video $video --model $model --provider cpu `
  --inference-width 160 --inference-height 90 --budget-ms 16 `
  --samples 120 --stride 1 --start-frame 0 `
  --output-directory "$out\integrated_replay"
```

Run the one-third/two-thirds experiment only as a separate future-work result:

```powershell
& ".\.venv\Scripts\python.exe" .\benchmarks\evaluate_two_insertions.py `
  --video $video --model $model --provider cpu `
  --inference-width 160 --inference-height 90 --budget-ms 16 `
  --samples 240 --start-frame 0 `
  --output-directory "$out\two_insertions"
```

Copy the exact command text into the run notes. A successful exit is not enough:
confirm that the expected CSV and JSON files exist and contain the requested
sample count.

### Live stable set

- passthrough/no frame generation;
- stable one-midpoint generation with regional mode off;
- regional mode on with the selected deadline;
- optional provenance overlay as a diagnostic run only, never as the primary
  performance result.

For every condition, perform three 60-second measured runs after a 30-second
warm-up. Use five repetitions if the coefficient of variation or obvious
thermal drift is high. Randomize condition order for each scene/machine. Keep
resolution, application graphics settings, route/camera movement, frame cap,
capture backend, model, provider, inference size, queue depth, and presentation
target fixed.

For gameplay, use a written route with timed actions. If exact replay is not
available, record the route and treat it as an application case study rather
than a tightly controlled causal comparison.

## 6. Metrics to retain

For every live run retain:

- observed source cadence;
- scheduler releases and preview submission calls (reported separately);
- PresentMon presentation/display events when available;
- exact sampled-output changes and meaningful-motion changes, explicitly
  labelled as pipeline-content heuristics rather than visible scanout;
- generated and real-frame counts;
- mean, median, p95, and p99 frame age/latency;
- capture, processing, interpolation, and composition times;
- dropped frames split by reason;
- queue depth/replacements;
- inference, bypass, midpoint-blend, endpoint, and deadline-fallback counts;
- route and provenance area percentages;
- predicted and actual transform cost;
- CPU, RAM, GPU-engine utilization, dedicated/shared GPU memory, clock,
  temperature, and power when available.

For offline quality retain:

- PSNR and SSIM;
- LPIPS when the same implementation and preprocessing are available for all
  conditions;
- temporal error/flicker metric;
- per-category visible failure counts;
- processing mean, median, p95, and p99;
- route/provenance coverage;
- exact sample indices and exclusions.

## 7. Measurements still needed before a “real-time” claim

### Displayed events

Use PresentMon/ETW against the presenter process after its integration is
validated. Retain the raw CSV and document PresentMon version, command, display
refresh, present mode, and whether events are application presents or displayed
events.

### Visibly unique frames

After the measurement marker is implemented, enable a 16-bit Gray-code frame ID
and checksum outside the quality-scored area. Film the monitor at 240 FPS or
higher with fixed exposure. Decode IDs to report visible unique FPS, repeats,
missing IDs, and disagreement with software submissions. Record camera model,
mode, exposure, monitor refresh, and decoder accuracy.

### Input-to-photon delay

For a defensible result, use one electrical event to trigger both an input and
an LED, then film the LED and response area together at high speed; a photodiode
is better when available. Collect at least 30 trials per condition and report
median, p95, p99, camera-frame uncertainty, and failures for source-only,
passthrough, stable midpoint, and regional mode.

Until these are complete, use “software presentation calls/s” and
“capture-to-submission latency,” not “display FPS” or “input lag.”

## 8. Causal experiments that require one more code milestone

Before claiming the scheduler or regional allocator caused an improvement, add
independent switches and run:

- threaded FIFO versus threaded newest-frame replacement;
- identical queue capacity and workload in both;
- regional analysis-only;
- routing-only;
- confidence composition added;
- deadline allocation added;
- temporal feedback added;
- deadline allocation versus equal-cost random, motion-only, and equal-count
  tile selection.

These experiments should use identical recorded inputs. Run at least three
repetitions in randomized order on both machines. This is the direct answer to
the reviewer’s causal-attribution criticism.

## 9. Friend’s lower-spec checklist

Send your friend:

1. the repository URL and exact commit;
2. setup instructions and model files/hashes;
3. the same evaluation clips or a shared manifest of clips;
4. exported profiles for each condition;
5. the route/timing sheet for live tests;
6. this folder naming convention;
7. a blank run-notes form;
8. instructions to run on AC power, close nonessential programs, disable
   overlays/recorders unless required, preserve failures, and never change
   settings between repetitions.

Ask the friend to begin with a short smoke test, then collect the hardware
manifest, offline replay data, and three repeated 60-second live runs per ready
condition. They should not tune settings to make their machine look better;
hardware generality requires the same predefined conditions and transparent
fallback behavior.

Use `docs/IRIS_RUN_RECORD_TEMPLATE.md` for every run.

## 10. Analysis rules

- Freeze exclusions before looking at which method wins.
- Aggregate frames into one value per scene/run before inferential analysis.
- Use paired comparisons because the same scenes feed each condition.
- Report effect size and 95% confidence interval, not only a p-value.
- Plot individual scene/run points in addition to means.
- Separate results by hardware; combine only with an explicit hierarchical or
  mixed-effects analysis.
- Publish negative results, fallback-heavy cases, and missing resource fields.
- Generate every paper table/figure from retained raw files with a checked-in
  script.

## Minimum evidence gate for drafting conclusions

Do not freeze the paper conclusion until the dataset contains:

- at least 24 independent scenes from three or more application classes;
- at least two hardware classes;
- three repeated performance runs per condition;
- a matched scheduler comparison that holds concurrency constant;
- the regional component ablations;
- resource manifests and raw traces;
- either displayed/unique-frame and input-latency measurements, or a paper title
  and claims that explicitly avoid “real-time” and display-FPS assertions.
