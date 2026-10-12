<div align="center">

# Adaptive Frame Routing (FrameRoute)

### Confidence- and deadline-aware regional frame transformation

**Research software for renderer-independent capture, regional enhancement, frame interpolation, pacing, and presentation.**

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Platform](https://img.shields.io/badge/Platform-Windows-0078D6)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active%20Development-orange)

</div>

## Overview

Adaptive Frame Routing captures a selected Windows application, classifies
regions by motion and reliability, applies the most useful transformation that
fits the current deadline, and presents a provenance-traceable output. It is a
research platform for testing when external frame generation helps and when a
stable real-frame fallback is safer.

The central research question and the boundary between original work and
third-party components are documented in
[`docs/IRIS_RESEARCH_FRAMING.md`](docs/IRIS_RESEARCH_FRAMING.md).

The clickable GUI provides target-window selection, WGC capture, CPU or experimental D3D11 rendering, spatial and ONNX AI processing, DirectML support, RIFE frame generation, runtime tuning, profiles, and performance counters.

**Naming:** Adaptive Frame Routing is the research project and repository.
`UniversalUpscaler` remains only as the legacy executable/configuration name
while compatibility is preserved.

## Install and launch from source

Use Python 3.11 or newer on Windows:

```powershell
git clone https://github.com/BNR007/Adaptive-Frame-Routing.git
cd Adaptive-Frame-Routing
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-runtime.txt
python -m pip install -r requirements-gui.txt
python src/ui/app.py
```

For development, testing, and packaging, install the complete build set:

```powershell
python -m pip install -r requirements-build.txt
```

The dependency files include `pywin32`, `mss`, `dxcam`, `onnx`, `PySide6`, `psutil`, OpenCV, ONNX Runtime, and DirectML support. Torch is optional and is used by model export and provenance tests.

## Frame generation

The generated amount is the number of generated frames inserted for every real frame interval:

| Setting | Real FPS | Generated FPS | Displayed FPS |
|---|---:|---:|---:|
| 1x | 30 | 30 | 60 |
| 2x | 30 | 60 | 90 |
| 3x | 30 | 90 | 120 |
| 4x | 30 | 120 | 150 |

The Start button counts down from five before capture begins, giving you time to switch to the target. The processed preview then opens fullscreen. The optional preview overlay contains only `real / generated`; for example, `30 / 120` means 30 real frames and 120 generated frames were actually presented. Detailed timing and dropped-frame diagnostics remain available in the expandable Status area.

The first real frame establishes interpolation history, so a short finite session has one fewer generated interval than its real-frame count. Selecting 60 FPS is a pacing target, not a guarantee that the hardware can calculate 60 frames per second.

The **IRIS Buffered Playout 720p30** research profile is a separate ablation.
Its `2` setting means one normal midpoint plus one inexpensive 0.75 endpoint
blend held as a jitter-concealment candidate. It does not perform two neural
inferences and must not be reported as true 3x temporal reconstruction.

Temporal stabilization is enabled by default in the GUI. Effectively duplicate
frame pairs bypass model inference, nearly stationary regions retain stable real
pixels, and low-motion generated values are constrained by their two real
endpoints. Moving regions remain model-generated. The control can be disabled
under Rendering for direct A/B comparisons.

On the first packaged-app run, UniversalUpscaler benchmarks the local CPU and OpenCV configuration before starting the countdown. The result is saved in the user's local application-data directory, so subsequent starts skip that one-time tuning.

Frame generation limits its internal inference resolution according to the selected preset: Performance 160x90, Fast Quality 240x135, Balanced 320x180, and Quality 640x360. Generated frames are restored to the presentation size. RIFE v3.6 is the default; IFRNet-S Vimeo90K and RIFE 4.25 Lite are experimental GUI options. Higher generation amounts use recursive interpolation and require additional inference work.

AI upscaling and frame generation have independent DirectML device controls. On
dual-GPU laptops, this permits the game and optional AI upscaler to remain on the
discrete GPU while interpolation runs on the integrated GPU. The CLI equivalent
is `--ai-device-id 1 --rife-device-id 0`; actual adapter ordering is system-specific.

## Feature status

- Confirmed working: WGC capture, CPU spatial upscaling, ONNX AI processing, RIFE interpolation, and the shader runtime tested with AssaultCube and Portal 2.
- Supported CPU combinations: spatial or AI upscaling with optional RIFE 1x-4x generation.
- D3D11 remains experimental. It requires a selected WGC window and `nearest`, `bilinear`, `lanczos`, or `fsr1_like` spatial processing.
- AI processing and frame generation use the CPU-frame pipeline and cannot currently be combined with D3D11.
- D3D11 presentation requires an interactive Windows graphics session and cannot use `--no-preview`.
- The bundled AI model is SRVGGNetCompact x2. Custom ONNX models must match the selected tensor layout, color order, and scale.

## CLI

```powershell
python src/main.py --list-windows
python src/main.py --window-title "AssaultCube" --upscaler-method fsr1_like
python src/main.py --window-title "PORTAL 2" --frame-generation rife --generated-frames 1 --warmup-seconds 5
```

CLI warm-up defaults to zero to preserve scripting behavior. The GUI uses five seconds.

## Architecture

```text
Application Window
        |
        v
Windows Graphics Capture
        |
        v
Newest Frame Queue
        |
        +-------------> Spatial Processing
        |
        +-------------> AI Processing (ONNX)
                              |
                              v
                     Optional RIFE Generation
                              |
                              v
                     Frame Pacing & Scheduling
                              |
                              v
                       Enhanced Presentation
```

## Tests

```powershell
python -m pip install -r requirements-test.txt
python -m pytest
```

Torch-backed export and provenance tests skip automatically when Torch is unavailable. Install `torch` to include them. Test discovery is restricted to the project's `tests` directory by `pytest.ini`, so backup repositories are not collected.

## Build the clickable application

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_windows.ps1
powershell -ExecutionPolicy Bypass -File scripts/test_packaged_app.ps1
```

The executable is produced at `dist\UniversalUpscaler\UniversalUpscaler.exe`. The package includes the GUI, ONNX models, Qt platform plugin, ONNX Runtime and DirectML libraries, application icon, and runtime configuration. Logs and user settings are written to the user's local application-data directory.

## Troubleshooting

- Refresh the window list after opening or closing a target application.
- Use WGC for selected-window capture. Exclusive-fullscreen applications may need borderless or windowed mode.
- Select the Performance preset, lower the RIFE generation amount, or lower the target FPS when generated frames are being dropped.
- Select CPU if DirectML initialization fails, or enable explicit provider fallback.
- Install current GPU drivers before using DirectML or D3D11.

Hardware, drivers, window permissions, source cadence, and capture-backend availability affect achievable output FPS.

## Contributing

Contributions covering bug fixes, performance improvements, model integration, documentation, benchmarks, and compatibility testing are welcome. Performance reports should include the CPU, GPU, Windows version, resolution, processing mode, model, latency, and measured FPS.

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

## Disclaimer

URFTS is an independent open-source research project. It is not affiliated with or endorsed by Microsoft, AMD, NVIDIA, Intel, OpenAI, or the developers of any AI model or game. All trademarks belong to their respective owners.
