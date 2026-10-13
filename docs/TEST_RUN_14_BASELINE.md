# Test run 14 reference checkpoint

This checkpoint retains the buffered regional gameplay path selected by the
author after the October 13 manual tests, plus exact PresentMon attribution.
It excludes the later waitable-swap-chain experiment. No interpolation or
playout behavior is changed by this checkpoint.

The original run used HEAD 41310e3 with uncommitted measurement changes. No
complete source snapshot was saved with that run; this is a reconstruction of
its code path and retained configuration, not a byte-for-byte verified snapshot.

## Friend setup

Run `Setup GUI.cmd`, then `Start GUI.cmd`. Select **IRIS Buffered Playout
720p30**, select the actual game window, and check these settings:

| Setting | Run 14 reference |
| --- | --- |
| Source game | Subnautica: Below Zero, 1280x720, 30 FPS cap |
| Monitor | 60 Hz |
| Capture / processing / presenter | WGC / CPU / D3D11 |
| Spatial method | Shader mode, Bicubic, refinement off |
| Interpolator | RIFE 4.25 Lite, internal 160x90 |
| Provider | DirectML, select the friend's actual GPU |
| Generated amount / buffered playout | 2 / enabled (midpoint + cheap reserve) |
| Target / pacing | 60 / Fixed |
| Queue / max latency | 2 / 130 ms |
| Regional routing / budget | enabled / 34 ms |
| Temporal stabilization | enabled |
| Research warmup / measurement | 10 seconds / 30 seconds |
| Visual marker | enabled |

The GUI preset defaults to CPU inference. Change it to DirectML to match the
original provider. The original machine used device index 1; do not assume the
friend's device 1 is the same GPU. If CPU is required, record it as a different
hardware/provider condition rather than an identical reproduction.

Use a new machine ID, scene ID and repetition number for each test; existing
run folders are deliberately not overwritten. PresentMon needs administrator
or Performance Log Users access. Check the final log for exact swap-chain
attribution, not just the software overlay. Prefer a 60 FPS OBS recording with
the marker if recording overhead is acceptable; record whether OBS was running.

## Reference results and limits

Run 14: source 30.09 FPS, software submissions 59.57/s, internally changing
frames 59.50/s, matched DXGI display events 33.92/s. Display interval median
33.32 ms, p95 50.00 ms and p99 50.02 ms. Median/p95 content age was
110.98/121.38 ms. Zero neural refinements were applied. This does not establish
smooth 60 FPS, successful neural generation, unique physical screen frames or
input-to-photon latency. Do not add the unexplained Other/0x0 stream to DXGI.

Later run 15 and standalone presenter tests are developmental evidence, not
substitutes for these baseline numbers. Friend results may differ. Save the
manifest, summaries, raw logs, PresentMon CSV, resource trace and observations.
