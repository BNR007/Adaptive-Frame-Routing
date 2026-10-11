"""Record an AFR-free native baseline for the IRIS evaluation.

PresentMon supplies display/presentation timing while this process records
resource and environment metadata.  It intentionally does not import or start
the AFR capture/presentation pipeline.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import statistics
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_run import ProcessResourceProbe, ResearchRunSpec  # noqa: E402


RESOURCE_FIELDS = (
    "elapsed_seconds",
    "phase",
    "game_cpu_percent",
    "game_ram_mb",
    "gpu_usage_percent",
    "dedicated_vram_mb",
    "gpu_temperature_c",
    "gpu_power_w",
    "gpu_source",
)


def _creation_flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def _run_text(command: list[str], *, timeout: float = 30.0) -> str | None:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            creationflags=_creation_flags(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _powershell_json(script: str) -> Any:
    output = _run_text(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            f"{script} | ConvertTo-Json -Depth 4 -Compress",
        ]
    )
    if not output:
        return None
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return None


def hardware_manifest() -> dict[str, Any]:
    return {
        "captured_at": datetime.now().astimezone().isoformat(),
        "machine": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": _powershell_json(
            "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors"
        ),
        "gpus": _powershell_json(
            "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,DriverDate,AdapterRAM"
        ),
        "memory": _powershell_json(
            "Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer,Model,TotalPhysicalMemory"
        ),
        "windows": _powershell_json(
            "Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber"
        ),
        "display": _powershell_json(
            "Get-CimInstance Win32_VideoController | Select-Object Name,CurrentHorizontalResolution,CurrentVerticalResolution,CurrentRefreshRate"
        ),
        "power_scheme": _run_text(["powercfg.exe", "/getactivescheme"]),
        "nvidia": _run_text(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
    }


def find_process(process_name: str):
    try:
        import psutil
    except ImportError as error:
        raise RuntimeError("psutil is required for native-baseline recording") from error
    wanted = process_name.casefold().removesuffix(".exe")
    matches = []
    for process in psutil.process_iter(("pid", "name", "exe")):
        try:
            name = str(process.info.get("name") or "").casefold().removesuffix(".exe")
            if name == wanted:
                matches.append(process)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    if not matches:
        raise RuntimeError(f"Process is not running: {process_name}")
    if len(matches) > 1:
        raise RuntimeError(
            f"More than one {process_name} process is running; pass --process-id instead."
        )
    return matches[0]


def process_from_id(process_id: int):
    import psutil

    try:
        return psutil.Process(process_id)
    except psutil.Error as error:
        raise RuntimeError(f"Process ID is not running: {process_id}") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--process-name", help="Game executable, for example SubnauticaZero.exe")
    target.add_argument("--process-id", type=int)
    parser.add_argument(
        "--presentmon",
        type=Path,
        default=Path.home() / "Downloads" / "PresentMon-2.3.1-x64.exe",
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--machine-id", default=platform.node() or "machine")
    parser.add_argument("--application", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--warmup-seconds", type=float, default=30.0)
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    parser.add_argument("--resource-interval", type=float, default=0.5)
    return parser


def _numbers(rows: list[dict[str, str]], field: str) -> list[float]:
    result = []
    for row in rows:
        try:
            result.append(float(row[field]))
        except (KeyError, TypeError, ValueError):
            continue
    return result


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"samples": 0, "mean": None, "median": None, "p95": None, "p99": None}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "samples": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def write_summary(run_directory: Path) -> dict[str, Any]:
    with (run_directory / "presentmon.csv").open(
        newline="", encoding="utf-8-sig"
    ) as source:
        present_rows = list(csv.DictReader(source))
    with (run_directory / "resources.csv").open(
        newline="", encoding="utf-8-sig"
    ) as source:
        resource_rows = [
            row for row in csv.DictReader(source) if row.get("phase") == "measuring"
        ]
    frame_times = _numbers(present_rows, "FrameTime")
    median_frame_time = statistics.median(frame_times) if frame_times else 0.0
    summary = {
        "metric_name": "PresentMon-observed native presentation events",
        "presentmon_rows": len(present_rows),
        "presentation_rate_per_second": (
            1000.0 / statistics.fmean(frame_times) if frame_times else 0.0
        ),
        "frame_time_ms": _distribution(frame_times),
        "gpu_time_ms": _distribution(_numbers(present_rows, "GPUTime")),
        "display_latency_ms": _distribution(
            _numbers(present_rows, "DisplayLatency")
        ),
        "presentmon_all_input_to_photon_ms": _distribution(
            _numbers(present_rows, "AllInputToPhotonLatency")
        ),
        "frame_time_spikes": {
            "over_50_ms": sum(value > 50.0 for value in frame_times),
            "over_100_ms": sum(value > 100.0 for value in frame_times),
            "over_1_5x_median": sum(
                value > 1.5 * median_frame_time for value in frame_times
            ),
        },
        "resources": {
            field: _distribution(_numbers(resource_rows, field))
            for field in (
                "game_cpu_percent",
                "game_ram_mb",
                "gpu_usage_percent",
                "dedicated_vram_mb",
                "gpu_temperature_c",
                "gpu_power_w",
            )
        },
        "notes": [
            "CPU percent is target-process usage and can exceed 100 on multicore systems.",
            "GPU, VRAM, temperature, and power are whole-adapter NVIDIA readings.",
            "PresentMon latency is not a substitute for an external input-to-photon measurement.",
        ],
    }
    (run_directory / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.resource_interval <= 0:
        raise SystemExit("--resource-interval must be positive")
    if not args.presentmon.is_file():
        raise SystemExit(f"PresentMon executable not found: {args.presentmon}")

    process = (
        process_from_id(args.process_id)
        if args.process_id is not None
        else find_process(args.process_name)
    )
    spec = ResearchRunSpec(
        root=args.root,
        machine_id=args.machine_id,
        application=args.application,
        scene_id=args.scene,
        condition="a_native",
        repetition=args.repetition,
        warmup_seconds=args.warmup_seconds,
        duration_seconds=args.duration_seconds,
    )
    run_directory = spec.directory()
    if run_directory.exists():
        raise SystemExit(f"Run already exists and will not be overwritten: {run_directory}")
    run_directory.mkdir(parents=True)

    process_name = process.name()
    manifest = {
        "status": "recording",
        "run": {
            "machine_id": args.machine_id,
            "application": args.application,
            "scene": args.scene,
            "condition": "a_native",
            "repetition": args.repetition,
            "warmup_seconds": args.warmup_seconds,
            "duration_seconds": args.duration_seconds,
        },
        "target": {"pid": process.pid, "process_name": process_name, "exe": process.exe()},
        "presentmon": str(args.presentmon),
        "hardware": hardware_manifest(),
        "metric_scope": {
            "cpu_ram": "target process",
            "gpu_vram_temperature_power": "whole NVIDIA adapter when nvidia-smi is available",
            "display_timing": "PresentMon events",
        },
    }
    manifest_path = run_directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_directory / "game_settings.md").write_text(
        "# Game settings\n\n"
        "- Resolution: 1280x720\n"
        "- Display mode:\n"
        "- Frame cap: 30 FPS\n"
        "- VSync:\n"
        "- Graphics preset:\n"
        "- Per-setting overrides:\n"
        "- Display resolution and refresh rate:\n"
        "- Scene route and timed actions:\n",
        encoding="utf-8",
    )
    (run_directory / "observations.md").write_text(
        "# Native-run observations\n\n"
        "- Visible stutter (0-3):\n"
        "- Freeze count and timestamps:\n"
        "- Thermal throttling observed:\n"
        "- Route deviations or interruptions:\n"
        "- Notes:\n",
        encoding="utf-8",
    )

    presentmon_csv = run_directory / "presentmon.csv"
    presentmon_log = (run_directory / "presentmon.log").open("w", encoding="utf-8")
    command = [
        str(args.presentmon),
        "--process_id",
        str(process.pid),
        "--restart_as_admin",
        "--delay",
        str(max(0, round(args.warmup_seconds))),
        "--timed",
        str(max(1, round(args.duration_seconds))),
        "--terminate_after_timed",
        "--v2_metrics",
        "--output_file",
        str(presentmon_csv),
    ]
    (run_directory / "presentmon_command.txt").write_text(
        subprocess.list2cmdline(command), encoding="utf-8"
    )
    presentmon = subprocess.Popen(
        command,
        stdout=presentmon_log,
        stderr=subprocess.STDOUT,
        creationflags=_creation_flags(),
    )

    probe = ProcessResourceProbe(game_pid=process.pid)
    resource_path = run_directory / "resources.csv"
    started_at = time.perf_counter()
    measurement_at = started_at + args.warmup_seconds
    finish_at = measurement_at + args.duration_seconds
    try:
        with resource_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=RESOURCE_FIELDS)
            writer.writeheader()
            next_sample = started_at
            while time.perf_counter() < finish_at:
                now = time.perf_counter()
                if now < next_sample:
                    time.sleep(min(0.05, next_sample - now))
                    continue
                sample = probe.sample()
                writer.writerow(
                    {
                        "elapsed_seconds": now - started_at,
                        "phase": "warmup" if now < measurement_at else "measuring",
                        "game_cpu_percent": sample.get("game_cpu_percent"),
                        "game_ram_mb": sample.get("game_ram_mb"),
                        "gpu_usage_percent": sample.get("gpu_usage_percent"),
                        "dedicated_vram_mb": sample.get("dedicated_vram_mb"),
                        "gpu_temperature_c": sample.get("gpu_temperature_c"),
                        "gpu_power_w": sample.get("gpu_power_w"),
                        "gpu_source": sample.get("gpu_source"),
                    }
                )
                next_sample += args.resource_interval
    finally:
        try:
            presentmon.wait(timeout=15)
        except subprocess.TimeoutExpired:
            presentmon.terminate()
            presentmon.wait(timeout=5)
        presentmon_log.close()

    manifest["status"] = "complete" if presentmon.returncode == 0 else "presentmon_failed"
    manifest["presentmon_return_code"] = presentmon.returncode
    manifest["completed_at"] = datetime.now().astimezone().isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Condition A artifacts: {run_directory}")
    if not presentmon_csv.exists() or presentmon_csv.stat().st_size == 0:
        print("PresentMon did not produce frame events; inspect presentmon.log.", file=sys.stderr)
        return 2
    write_summary(run_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
