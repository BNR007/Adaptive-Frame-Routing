"""Reproducible live-run recording for the IRIS evaluation.

The normal five-second console telemetry is intended for operators.  This
module records every submitted output during the measured interval so that
run-wide distributions and regional provenance can be reconstructed without
sampling the console log.
"""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import threading
import time
from typing import Any, Callable, Mapping

import numpy as np


FRAME_FIELDS = (
    "elapsed_seconds",
    "source_sequence_id",
    "output_frame_id",
    "frame_kind",
    "generation_method",
    "generation_position",
    "active_provider",
    "interpolation_provider",
    "frame_generation",
    "captured_at",
    "processing_started_at",
    "processed_at",
    "scheduled_at",
    "submitted_at",
    "capture_ms",
    "processing_ms",
    "interpolation_ms",
    "frame_age_ms",
    "capture_to_submission_ms",
    "pacing_error_ms",
    "estimated_source_fps",
    "software_presentation_fps",
    "presentation_interval_ms",
    "target_interval_ms",
    "cadence_misses",
    "exact_unique",
    "meaningful_motion",
    "motion_mad",
    "source_frames_submitted",
    "source_frames_processed",
    "input_queue_replacements",
    "output_queue_replacements",
    "stale_reserved_batches",
    "pipeline_generated_drops",
    "generated_deadline_drops",
    "stale_generated_drops",
    "real_deadline_drops",
    "late_frames",
    "runtime_state",
    "recovery_attempts",
    "successful_recoveries",
    "failed_recoveries",
    "provider_fallbacks",
    "regional_admission",
    "route_static_detail",
    "route_smooth_motion",
    "route_unreliable_motion",
    "route_hud_text",
    "route_low_detail",
    "route_scene_cut",
    "provenance_real",
    "provenance_upscaled",
    "provenance_interpolated",
    "provenance_blended",
    "provenance_reused",
    "provenance_deadline_fallback",
    "provenance_admission_fallback",
)

RESOURCE_FIELDS = (
    "elapsed_seconds",
    "phase",
    "afr_cpu_percent",
    "afr_ram_mb",
    "game_cpu_percent",
    "game_ram_mb",
    "gpu_usage_percent",
    "dedicated_vram_mb",
    "gpu_temperature_c",
    "gpu_power_w",
    "gpu_source",
)

_ROUTE_FIELDS = {
    "static_detail": "route_static_detail",
    "smooth_motion": "route_smooth_motion",
    "unreliable_motion": "route_unreliable_motion",
    "hud_text": "route_hud_text",
    "low_detail": "route_low_detail",
    "scene_cut": "route_scene_cut",
}

_PROVENANCE_FIELDS = {
    "real": "provenance_real",
    "upscaled": "provenance_upscaled",
    "interpolated": "provenance_interpolated",
    "blended": "provenance_blended",
    "reused": "provenance_reused",
    "deadline_fallback": "provenance_deadline_fallback",
    "admission_fallback": "provenance_admission_fallback",
}


def _slug(value: str, *, fallback: str) -> str:
    cleaned = "".join(
        character.lower() if character.isalnum() else "_"
        for character in str(value).strip()
    )
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or fallback


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _hash_file(path_value: Any) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "samples": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
    }


def _safe_git(root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            creationflags=(
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                if os.name == "nt"
                else 0
            ),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _window_process_id(hwnd: int) -> int | None:
    if os.name != "nt" or hwnd <= 0:
        return None
    try:
        import ctypes

        process_id = ctypes.c_ulong()
        ctypes.windll.user32.GetWindowThreadProcessId(
            ctypes.c_void_p(hwnd), ctypes.byref(process_id)
        )
        return int(process_id.value) or None
    except (AttributeError, OSError, ValueError):
        return None


def find_presentmon(explicit: str | Path | None = None) -> Path | None:
    """Locate PresentMon without making research runs depend on one install path."""

    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    environment_path = os.environ.get("PRESENTMON_PATH")
    if environment_path:
        candidates.append(Path(environment_path).expanduser())
    discovered = shutil.which("PresentMon.exe") or shutil.which("PresentMon")
    if discovered:
        candidates.append(Path(discovered))
    downloads = Path.home() / "Downloads"
    if downloads.is_dir():
        candidates.extend(sorted(downloads.glob("PresentMon*.exe"), reverse=True))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


class OutputFrameAnalyzer:
    """Cheap output uniqueness and cadence measurements for live runs."""

    def __init__(self, *, motion_threshold: float = 1.0) -> None:
        self.motion_threshold = float(motion_threshold)
        self._previous_luma: np.ndarray | None = None
        self._previous_hash: bytes | None = None
        self._previous_at: float | None = None
        self.calls = 0
        self.unique = 0
        self.meaningful = 0
        self.started_at: float | None = None
        self.last = {
            "presentation_interval_ms": 0.0,
            "exact_unique": 0,
            "meaningful_motion": 0,
            "motion_mad": 0.0,
        }

    @staticmethod
    def _thumbnail(frame: Any) -> np.ndarray:
        image = np.asarray(frame)
        if image.ndim not in (2, 3):
            raise ValueError("Presented frame must be a 2D or 3D image array.")
        y_step = max(1, image.shape[0] // 36)
        x_step = max(1, image.shape[1] // 64)
        # Slice first. Converting the full 720p/1080p image to float merely to
        # keep 2,304 samples added measurable work to every preview submission.
        sample = image[::y_step, ::x_step][:36, :64]
        if sample.ndim == 3:
            sample = sample[..., :3].astype(np.uint16).sum(axis=2) // 3
        return np.ascontiguousarray(sample, dtype=np.uint8)

    def observe(self, frame: Any, presented_at: float) -> dict[str, float | int]:
        luma = self._thumbnail(frame)
        digest = hashlib.blake2b(luma.tobytes(), digest_size=8).digest()
        interval_ms = (
            (float(presented_at) - self._previous_at) * 1000.0
            if self._previous_at is not None
            else 0.0
        )
        exact_unique = int(self._previous_hash is None or digest != self._previous_hash)
        motion_mad = (
            float(np.mean(np.abs(luma.astype(np.int16) - self._previous_luma.astype(np.int16))))
            if self._previous_luma is not None and self._previous_luma.shape == luma.shape
            else 0.0
        )
        meaningful = int(self._previous_luma is None or motion_mad >= self.motion_threshold)
        if self.started_at is None:
            self.started_at = float(presented_at)
        self.calls += 1
        self.unique += exact_unique
        self.meaningful += meaningful
        self._previous_luma = luma
        self._previous_hash = digest
        self._previous_at = float(presented_at)
        self.last = {
            "presentation_interval_ms": interval_ms,
            "exact_unique": exact_unique,
            "meaningful_motion": meaningful,
            "motion_mad": motion_mad,
        }
        return dict(self.last)

    def snapshot(self, now: float | None = None) -> dict[str, float | int]:
        current = float(now if now is not None else (self._previous_at or 0.0))
        elapsed = current - self.started_at if self.started_at is not None else 0.0
        return {
            **self.last,
            "calls_per_second": self.calls / elapsed if elapsed > 0.0 else 0.0,
            "unique_per_second": self.unique / elapsed if elapsed > 0.0 else 0.0,
            "meaningful_per_second": self.meaningful / elapsed if elapsed > 0.0 else 0.0,
            "calls": self.calls,
            "unique": self.unique,
            "meaningful": self.meaningful,
        }
@dataclass(frozen=True, slots=True)
class ResearchRunSpec:
    root: Path
    machine_id: str
    application: str
    scene_id: str
    condition: str
    repetition: int
    warmup_seconds: float = 30.0
    duration_seconds: float = 60.0

    def validate(self) -> None:
        for value, name in (
            (self.machine_id, "machine ID"),
            (self.application, "application"),
            (self.scene_id, "scene ID"),
            (self.condition, "condition"),
        ):
            if not str(value).strip():
                raise ValueError(f"Research {name} cannot be empty.")
        if self.repetition <= 0:
            raise ValueError("Research repetition must be positive.")
        if self.warmup_seconds < 0.0:
            raise ValueError("Research warm-up cannot be negative.")
        if self.duration_seconds <= 0.0:
            raise ValueError("Research duration must be positive.")

    def directory(self, *, date: str | None = None) -> Path:
        self.validate()
        date = date or datetime.now().astimezone().date().isoformat()
        return (
            Path(self.root)
            / _slug(self.machine_id, fallback="machine")
            / date
            / _slug(self.application, fallback="application")
            / _slug(self.scene_id, fallback="scene")
            / _slug(self.condition, fallback="condition")
            / f"run_{self.repetition:02d}"
        )


class ProcessResourceProbe:
    """Best-effort process and NVIDIA sensor sampler.

    NVIDIA values are whole-adapter measurements and are labelled as such.
    Missing sensors remain empty rather than being inferred.
    """

    def __init__(self, *, game_pid: int | None, afr_pid: int | None = None) -> None:
        self._afr_pid = int(afr_pid or os.getpid())
        self._game_pid = game_pid
        self._nvidia_smi = shutil.which("nvidia-smi")
        self._afr_process = None
        self._game_process = None
        self._last_gpu_sample_at = float("-inf")
        self._gpu_cache = (None, None, None, None, "unavailable")
        try:
            import psutil

            self._afr_process = psutil.Process(self._afr_pid)
            self._afr_process.cpu_percent(None)
            if game_pid:
                self._game_process = psutil.Process(game_pid)
                self._game_process.cpu_percent(None)
        except (ImportError, OSError):
            self._afr_process = None
            self._game_process = None

    @staticmethod
    def _process_values(process) -> tuple[float | None, float | None]:
        if process is None:
            return None, None
        try:
            return float(process.cpu_percent(None)), process.memory_info().rss / 1048576.0
        except Exception:
            return None, None

    def _gpu_values(self) -> tuple[float | None, float | None, float | None, float | None, str]:
        # nvidia-smi starts a separate process. Query it at most once per second
        # so the research instrumentation does not become the workload.
        now = time.monotonic()
        if now - self._last_gpu_sample_at < 1.0:
            return self._gpu_cache
        self._last_gpu_sample_at = now
        if not self._nvidia_smi:
            self._gpu_cache = (None, None, None, None, "unavailable")
            return self._gpu_cache
        command = [
            self._nvidia_smi,
            "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            first = completed.stdout.splitlines()[0]
            values = [float(part.strip()) for part in first.split(",")]
            if completed.returncode == 0 and len(values) == 4:
                self._gpu_cache = (*values, "nvidia-smi whole adapter")
                return self._gpu_cache
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            pass
        self._gpu_cache = (None, None, None, None, "unavailable")
        return self._gpu_cache

    def sample(self) -> dict[str, Any]:
        afr_cpu, afr_ram = self._process_values(self._afr_process)
        game_cpu, game_ram = self._process_values(self._game_process)
        gpu, vram, temperature, power, source = self._gpu_values()
        return {
            "afr_cpu_percent": afr_cpu,
            "afr_ram_mb": afr_ram,
            "game_cpu_percent": game_cpu,
            "game_ram_mb": game_ram,
            "gpu_usage_percent": gpu,
            "dedicated_vram_mb": vram,
            "gpu_temperature_c": temperature,
            "gpu_power_w": power,
            "gpu_source": source,
        }


class ResearchRunRecorder:
    """Own one immutable run directory and its raw research artifacts."""

    def __init__(
        self,
        spec: ResearchRunSpec,
        *,
        project_root: Path,
        hwnd: int = 0,
        window_title: str | None = None,
        configuration: Mapping[str, Any] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        resource_probe: Any | None = None,
        resource_interval_seconds: float = 0.5,
        presentmon_path: str | Path | None = None,
    ) -> None:
        spec.validate()
        if resource_interval_seconds <= 0.0:
            raise ValueError("Resource sampling interval must be positive.")
        self.spec = spec
        self.project_root = Path(project_root)
        self.hwnd = int(hwnd)
        self.window_title = window_title
        self.configuration = dict(configuration or {})
        self.clock = clock
        self.resource_interval_seconds = float(resource_interval_seconds)
        self.run_directory = spec.directory()
        self._resource_probe = resource_probe or ProcessResourceProbe(
            game_pid=_window_process_id(self.hwnd)
        )
        self._prepared_at = self.clock()
        self._timing_started_at: float | None = None
        self._measurement_started_at: float | None = None
        self._finished_at: float | None = None
        self._output_id = 0
        self._frame_rows: list[dict[str, Any]] = []
        self._resource_rows: list[dict[str, Any]] = []
        self._route_totals: Counter[str] = Counter()
        self._provenance_totals: Counter[str] = Counter()
        self._counter_baseline: dict[str, int] | None = None
        self._stop_resources = threading.Event()
        self._resource_thread: threading.Thread | None = None
        self._frame_file = None
        self._resource_file = None
        self._frame_writer = None
        self._resource_writer = None
        self._log_handler: logging.Handler | None = None
        self._closed = False
        self._presentmon_path = find_presentmon(presentmon_path) if presentmon_path else None
        self._presentmon_process: subprocess.Popen | None = None
        self._presentmon_log = None
        self._presentmon_summary: dict[str, Any] = {}

    @property
    def measurement_started_at(self) -> float | None:
        return self._measurement_started_at

    def prepare(self) -> Path:
        if self.run_directory.exists():
            raise FileExistsError(
                f"Research run already exists and will not be overwritten: {self.run_directory}"
            )
        self.run_directory.mkdir(parents=True)
        self._frame_file = (self.run_directory / "telemetry_frames.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._resource_file = (self.run_directory / "resources.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._frame_writer = csv.DictWriter(self._frame_file, fieldnames=FRAME_FIELDS)
        self._resource_writer = csv.DictWriter(
            self._resource_file, fieldnames=RESOURCE_FIELDS
        )
        self._frame_writer.writeheader()
        self._resource_writer.writeheader()
        self._log_handler = logging.FileHandler(
            self.run_directory / "raw.log", encoding="utf-8"
        )
        self._log_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logging.getLogger().addHandler(self._log_handler)
        (self.run_directory / "observations.md").write_text(
            "# Run observations\n\n"
            "- Flicker (0-3):\n"
            "- Ghosting (0-3):\n"
            "- HUD/text corruption (0-3):\n"
            "- Visible stutter (0-3):\n"
            "- Freezes (count):\n"
            "- Capture recoveries observed:\n"
            "- Interruptions or exclusions:\n"
            "- Notes:\n",
            encoding="utf-8",
        )
        self._write_manifest(status="prepared")
        return self.run_directory

    def begin_timing(self, *, sample_resources: bool = True) -> None:
        if self._frame_writer is None:
            raise RuntimeError("Research recorder must be prepared before timing begins.")
        if self._timing_started_at is not None:
            return
        self._timing_started_at = self.clock()
        self._measurement_started_at = (
            self._timing_started_at + self.spec.warmup_seconds
        )
        self._write_manifest(status="warming_up")
        self._start_presentmon()
        if sample_resources:
            self._resource_thread = threading.Thread(
                target=self._sample_resources,
                name="research-resource-sampler",
                daemon=True,
            )
            self._resource_thread.start()

    def phase(self, now: float | None = None) -> str:
        if self._timing_started_at is None:
            return "prepared"
        current = self.clock() if now is None else now
        assert self._measurement_started_at is not None
        if current < self._measurement_started_at:
            return "warmup"
        if current < self._measurement_started_at + self.spec.duration_seconds:
            return "measuring"
        return "complete"

    def remaining_seconds(self, now: float | None = None) -> float:
        if self._timing_started_at is None:
            return self.spec.warmup_seconds + self.spec.duration_seconds
        current = self.clock() if now is None else now
        if self.phase(current) == "warmup":
            assert self._measurement_started_at is not None
            return max(0.0, self._measurement_started_at - current)
        assert self._measurement_started_at is not None
        return max(
            0.0,
            self._measurement_started_at + self.spec.duration_seconds - current,
        )

    def record_frame(
        self,
        *,
        processed_frame: Any,
        decision: Any,
        pacing_snapshot: Any,
        recovery_snapshot: Any,
        pipeline: Any,
        output_analysis: Mapping[str, Any] | None = None,
        submitted_at: float | None = None,
    ) -> bool:
        # decision.actual_at is the scheduler release time.  In preview mode
        # the caller supplies the timestamp taken immediately after imshow so
        # capture-to-submission does not silently omit refinement/render work.
        now = float(decision.actual_at if submitted_at is None else submitted_at)
        if self.phase(now) != "measuring":
            return False
        assert self._measurement_started_at is not None
        self._output_id += 1
        current_counters = {
            "source_frames_submitted": int(getattr(pipeline, "submitted_frames", 0)),
            "source_frames_processed": int(getattr(pipeline, "processed_frames", 0)),
            "input_queue_replacements": int(pipeline.input_replacements),
            "output_queue_replacements": int(pipeline.result_replacements),
            "stale_reserved_batches": int(
                getattr(pipeline, "stale_reserved_batches", 0)
            ),
            "pipeline_generated_drops": int(pipeline.dropped_generated_frames),
            "generated_deadline_drops": int(
                pacing_snapshot.generated_frames_dropped_late
            ),
            "stale_generated_drops": int(
                getattr(pacing_snapshot, "stale_generated_frames_dropped", 0)
            ),
            "real_deadline_drops": int(pacing_snapshot.real_frames_dropped_late),
            "late_frames": int(pacing_snapshot.late_frames),
            "recovery_attempts": int(recovery_snapshot.retry_attempts),
            "successful_recoveries": int(recovery_snapshot.successful_recoveries),
            "failed_recoveries": int(recovery_snapshot.failed_recoveries),
            "provider_fallbacks": int(recovery_snapshot.fallback_activations),
        }
        if self._counter_baseline is None:
            self._counter_baseline = dict(current_counters)
        measured_counters = {
            key: max(0, value - self._counter_baseline[key])
            for key, value in current_counters.items()
        }
        routes = dict(processed_frame.regional_route_counts or {})
        provenance = dict(processed_frame.regional_provenance_counts or {})
        self._route_totals.update({str(key): int(value) for key, value in routes.items()})
        self._provenance_totals.update(
            {str(key): int(value) for key, value in provenance.items()}
        )
        row = {field: "" for field in FRAME_FIELDS}
        row.update(
            {
                "elapsed_seconds": now - self._measurement_started_at,
                "source_sequence_id": processed_frame.sequence_id,
                "output_frame_id": self._output_id,
                "frame_kind": processed_frame.frame_kind,
                "generation_method": processed_frame.generation_method,
                "generation_position": processed_frame.generation_position,
                "active_provider": processed_frame.active_provider,
                "interpolation_provider": processed_frame.interpolation_provider,
                "frame_generation": processed_frame.frame_generation,
                "captured_at": processed_frame.captured_at,
                "processing_started_at": processed_frame.processing_started_at,
                "processed_at": processed_frame.processed_at,
                "scheduled_at": decision.scheduled_at,
                "submitted_at": now,
                "capture_ms": processed_frame.capture_ms,
                "processing_ms": processed_frame.processing_ms,
                "interpolation_ms": processed_frame.interpolation_ms,
                "frame_age_ms": (now - processed_frame.captured_at) * 1000.0,
                "capture_to_submission_ms": (now - processed_frame.captured_at) * 1000.0,
                "pacing_error_ms": decision.pacing_error_ms,
                "estimated_source_fps": pacing_snapshot.estimated_source_fps,
                "software_presentation_fps": pacing_snapshot.presentation_fps,
                "presentation_interval_ms": (output_analysis or {}).get(
                    "presentation_interval_ms", getattr(pacing_snapshot, "presentation_interval_ms", 0.0)
                ),
                "target_interval_ms": getattr(pacing_snapshot, "target_interval_ms", 0.0),
                "cadence_misses": int(getattr(pacing_snapshot, "cadence_misses", 0)),
                "exact_unique": int((output_analysis or {}).get("exact_unique", 0)),
                "meaningful_motion": int((output_analysis or {}).get("meaningful_motion", 0)),
                "motion_mad": float((output_analysis or {}).get("motion_mad", 0.0)),
                "source_frames_submitted": measured_counters["source_frames_submitted"],
                "source_frames_processed": measured_counters["source_frames_processed"],
                "input_queue_replacements": measured_counters["input_queue_replacements"],
                "output_queue_replacements": measured_counters["output_queue_replacements"],
                "stale_reserved_batches": measured_counters["stale_reserved_batches"],
                "pipeline_generated_drops": measured_counters["pipeline_generated_drops"],
                "generated_deadline_drops": measured_counters["generated_deadline_drops"],
                "stale_generated_drops": measured_counters["stale_generated_drops"],
                "real_deadline_drops": measured_counters["real_deadline_drops"],
                "late_frames": measured_counters["late_frames"],
                "runtime_state": str(recovery_snapshot.state),
                "recovery_attempts": measured_counters["recovery_attempts"],
                "successful_recoveries": measured_counters["successful_recoveries"],
                "failed_recoveries": measured_counters["failed_recoveries"],
                "provider_fallbacks": measured_counters["provider_fallbacks"],
                "regional_admission": processed_frame.regional_admission_reason or "",
            }
        )
        for source, destination in _ROUTE_FIELDS.items():
            row[destination] = int(routes.get(source, 0))
        for source, destination in _PROVENANCE_FIELDS.items():
            row[destination] = int(provenance.get(source, 0))
        self._frame_writer.writerow(row)
        # Keep per-frame recording cheap; close() flushes the final partial block.
        if self._output_id % 120 == 0:
            self._frame_file.flush()
        self._frame_rows.append(row)
        return True

    def _start_presentmon(self) -> None:
        if self._presentmon_path is None:
            self._presentmon_summary = {
                "status": "unavailable",
                "rows": 0,
                "scope": "PresentMon executable was not found; URFTS telemetry remains available",
            }
            logging.getLogger(__name__).warning(
                "PresentMon was not found; this run will contain URFTS telemetry only."
            )
            return
        if self._presentmon_process is not None:
            return
        output = self.run_directory / "presentmon.csv"
        self._presentmon_log = (self.run_directory / "presentmon.log").open(
            "w", encoding="utf-8"
        )
        command = [
            str(self._presentmon_path),
            "--process_id", str(os.getpid()),
            "--session_name", f"URFTS_{os.getpid()}",
            "--restart_as_admin",
            "--delay", str(max(0, round(self.spec.warmup_seconds))),
            "--timed", str(max(1, round(self.spec.duration_seconds))),
            "--terminate_after_timed",
            "--no_console_stats",
            "--v2_metrics",
            "--output_file", str(output),
        ]
        (self.run_directory / "presentmon_command.txt").write_text(
            subprocess.list2cmdline(command), encoding="utf-8"
        )
        try:
            self._presentmon_process = subprocess.Popen(
                command,
                stdout=self._presentmon_log,
                stderr=subprocess.STDOUT,
                creationflags=(getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0),
            )
            logging.getLogger(__name__).info(
                "PresentMon research capture started for URFTS PID %s; output %s",
                os.getpid(), output,
            )
        except OSError as error:
            logging.getLogger(__name__).warning("PresentMon could not start: %s", error)
            self._presentmon_summary = {
                "status": "start_failed",
                "rows": 0,
                "error": str(error),
            }
            self._presentmon_process = None

    def _finish_presentmon(self) -> None:
        if self._presentmon_path is None:
            if not self._presentmon_summary:
                self._presentmon_summary = {"status": "unavailable", "rows": 0}
            (self.run_directory / "presentmon_summary.json").write_text(
                json.dumps(self._presentmon_summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            return
        process = self._presentmon_process
        if process is not None:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            self._presentmon_summary["return_code"] = process.returncode
        if self._presentmon_log is not None:
            self._presentmon_log.close()
            self._presentmon_log = None
        csv_path = self.run_directory / "presentmon.csv"
        if not csv_path.is_file() or csv_path.stat().st_size == 0:
            self._presentmon_summary.update({"status": "no_events", "rows": 0})
            (self.run_directory / "presentmon_summary.json").write_text(
                json.dumps(self._presentmon_summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            return
        try:
            with csv_path.open(newline="", encoding="utf-8-sig") as source:
                rows = list(csv.DictReader(source))
            columns = list(rows[0]) if rows else []

            def numeric_column(*names: str) -> list[float]:
                values: list[float] = []
                for row in rows:
                    for name in names:
                        raw = row.get(name)
                        if raw in (None, ""):
                            continue
                        try:
                            values.append(float(raw))
                        except (TypeError, ValueError):
                            pass
                        break
                return values

            frame_times = numeric_column("FrameTime", "MsBetweenPresents")
            displayed_times = [
                value for value in numeric_column("DisplayedTime", "MsBetweenDisplayChange")
                if value > 0.0
            ]
            display_latency = numeric_column("DisplayLatency", "MsUntilDisplayed")
            frame_types = Counter(
                str(row.get("FrameType", "")).strip()
                for row in rows
                if str(row.get("FrameType", "")).strip()
            )
            self._presentmon_summary.update({
                "status": "complete" if rows else "no_events",
                "rows": len(rows),
                "columns": columns,
                "observed_presentation_events_per_second": (
                    1000.0 / statistics.fmean(frame_times) if frame_times else 0.0
                ),
                "frame_time_ms": _distribution(frame_times),
                "observed_displayed_events_per_second": (
                    1000.0 / statistics.fmean(displayed_times) if displayed_times else None
                ),
                "displayed_time_ms": _distribution(displayed_times),
                "display_latency_ms": _distribution(display_latency),
                "frame_types": dict(sorted(frame_types.items())),
                "scope": (
                    "PresentMon events attributed to the URFTS process. DisplayedTime, "
                    "when supplied by PresentMon, is stronger evidence than URFTS calls "
                    "but is still not an optical input-to-photon measurement."
                ),
            })
            (self.run_directory / "presentmon_summary.json").write_text(
                json.dumps(self._presentmon_summary, indent=2, sort_keys=True), encoding="utf-8"
            )
        except (OSError, csv.Error) as error:
            self._presentmon_summary.update({"status": "parse_failed", "error": str(error)})

    def _sample_resources(self) -> None:
        while not self._stop_resources.wait(self.resource_interval_seconds):
            if self._timing_started_at is None:
                continue
            current = self.clock()
            phase = self.phase(current)
            if phase == "complete":
                break
            values = dict(self._resource_probe.sample())
            row = {field: "" for field in RESOURCE_FIELDS}
            row.update(values)
            row["elapsed_seconds"] = current - self._timing_started_at
            row["phase"] = phase
            self._resource_writer.writerow(row)
            self._resource_file.flush()
            if phase == "measuring":
                self._resource_rows.append(row)

    @staticmethod
    def _numeric(rows: list[dict[str, Any]], field: str) -> list[float]:
        values: list[float] = []
        for row in rows:
            value = row.get(field)
            if value in (None, ""):
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
        return values

    @staticmethod
    def _last_int(rows: list[dict[str, Any]], field: str) -> int:
        return int(float(rows[-1].get(field, 0))) if rows else 0

    def _summary(self, status: str) -> dict[str, Any]:
        submitted = self._numeric(self._frame_rows, "submitted_at")
        elapsed = submitted[-1] - submitted[0] if len(submitted) >= 2 else 0.0
        provenance_total = sum(self._provenance_totals.values())
        route_total = sum(self._route_totals.values())
        frame_kinds = Counter(str(row["frame_kind"]) for row in self._frame_rows)
        presentation_intervals = self._numeric(self._frame_rows, "presentation_interval_ms")
        presentation_intervals = [value for value in presentation_intervals if value > 0.0]
        unique_count = sum(self._numeric(self._frame_rows, "exact_unique"))
        meaningful_count = sum(self._numeric(self._frame_rows, "meaningful_motion"))
        return {
            "status": status,
            "measured_frames": len(self._frame_rows),
            "software_presentation_calls_per_second": (
                (len(submitted) - 1) / elapsed if elapsed > 0.0 else 0.0
            ),
            "observed_source_fps": _distribution(
                self._numeric(self._frame_rows, "estimated_source_fps")
            ),
            "presented_frame_counts": dict(sorted(frame_kinds.items())),
            "effective_output": {
                "exact_unique_frames": int(unique_count),
                "meaningful_motion_frames": int(meaningful_count),
                "exact_unique_frames_per_second": unique_count / elapsed if elapsed > 0.0 else 0.0,
                "meaningful_motion_frames_per_second": meaningful_count / elapsed if elapsed > 0.0 else 0.0,
                "presentation_interval_ms": _distribution(presentation_intervals),
                "motion_mad": _distribution(self._numeric(self._frame_rows, "motion_mad")),
                "cadence_misses": self._last_int(self._frame_rows, "cadence_misses"),
            },
            "presentmon": dict(self._presentmon_summary),
            "source_frame_counts": {
                "submitted": self._last_int(
                    self._frame_rows, "source_frames_submitted"
                ),
                "processed": self._last_int(
                    self._frame_rows, "source_frames_processed"
                ),
            },
            "capture_to_submission_ms": _distribution(
                self._numeric(self._frame_rows, "capture_to_submission_ms")
            ),
            "frame_age_ms": _distribution(
                self._numeric(self._frame_rows, "frame_age_ms")
            ),
            "capture_ms": _distribution(self._numeric(self._frame_rows, "capture_ms")),
            "processing_ms": _distribution(
                self._numeric(self._frame_rows, "processing_ms")
            ),
            "interpolation_ms": _distribution(
                self._numeric(self._frame_rows, "interpolation_ms")
            ),
            "drops": {
                "input_queue_replacements": self._last_int(
                    self._frame_rows, "input_queue_replacements"
                ),
                "output_queue_replacements": self._last_int(
                    self._frame_rows, "output_queue_replacements"
                ),
                "stale_reserved_batches": self._last_int(
                    self._frame_rows, "stale_reserved_batches"
                ),
                "pipeline_generated": self._last_int(
                    self._frame_rows, "pipeline_generated_drops"
                ),
                "generated_deadline": self._last_int(
                    self._frame_rows, "generated_deadline_drops"
                ),
                "stale_generated": self._last_int(
                    self._frame_rows, "stale_generated_drops"
                ),
                "real_deadline": self._last_int(
                    self._frame_rows, "real_deadline_drops"
                ),
            },
            "recoveries": {
                "attempts": self._last_int(self._frame_rows, "recovery_attempts"),
                "successful": self._last_int(
                    self._frame_rows, "successful_recoveries"
                ),
                "failed": self._last_int(self._frame_rows, "failed_recoveries"),
                "provider_fallbacks": self._last_int(
                    self._frame_rows, "provider_fallbacks"
                ),
            },
            "regional_route_counts": dict(sorted(self._route_totals.items())),
            "regional_route_percent": {
                key: value * 100.0 / route_total
                for key, value in sorted(self._route_totals.items())
            } if route_total else {},
            "regional_provenance_counts": dict(
                sorted(self._provenance_totals.items())
            ),
            "regional_provenance_percent": {
                key: value * 100.0 / provenance_total
                for key, value in sorted(self._provenance_totals.items())
            } if provenance_total else {},
            "resources": {
                field: _distribution(self._numeric(self._resource_rows, field))
                for field in RESOURCE_FIELDS
                if field not in {"elapsed_seconds", "phase", "gpu_source"}
            },
            "resource_sources": sorted(
                {
                    str(row.get("gpu_source"))
                    for row in self._resource_rows
                    if row.get("gpu_source")
                }
            ),
        }

    def _manifest_payload(self, status: str) -> dict[str, Any]:
        commit = _safe_git(self.project_root, "rev-parse", "HEAD")
        dirty = _safe_git(self.project_root, "status", "--porcelain")
        package_versions = {}
        for package in ("numpy", "opencv-python", "opencv-contrib-python", "onnxruntime-directml", "psutil"):
            try:
                package_versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                package_versions[package] = None
        configuration = _jsonable(self.configuration)
        return {
            "status": status,
            "created_at": datetime.now().astimezone().isoformat(),
            "run": _jsonable(asdict(self.spec)),
            "window": {"hwnd": self.hwnd, "title": self.window_title},
            "configuration": configuration,
            "model_sha256": {
                "rife": _hash_file(configuration.get("rife_model")),
                "ai": _hash_file(configuration.get("model")),
            },
            "repository": {
                "root": str(self.project_root),
                "commit": commit,
                "dirty": bool(dirty),
                "status_porcelain": dirty or "",
            },
            "system": {
                "machine": platform.node(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "processor": platform.processor(),
                "packages": package_versions,
            },
            "measurement": {
                "warmup_seconds": self.spec.warmup_seconds,
                "duration_seconds": self.spec.duration_seconds,
                "resource_interval_seconds": self.resource_interval_seconds,
                "resource_gpu_scope": "whole adapter when nvidia-smi is available",
            },
        }

    def _write_manifest(self, *, status: str) -> None:
        (self.run_directory / "manifest.json").write_text(
            json.dumps(self._manifest_payload(status), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def close(self, *, status: str = "complete") -> dict[str, Any]:
        if self._closed:
            return self._summary(status)
        self._closed = True
        self._finished_at = self.clock()
        self._stop_resources.set()
        if self._resource_thread is not None:
            self._resource_thread.join(timeout=3.0)
        self._finish_presentmon()
        summary = self._summary(status)
        (self.run_directory / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        self._write_manifest(status=status)
        if self._frame_file is not None:
            self._frame_file.close()
        if self._resource_file is not None:
            self._resource_file.close()
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler.close()
        return summary


__all__ = [
    "FRAME_FIELDS",
    "RESOURCE_FIELDS",
    "ProcessResourceProbe",
    "ResearchRunRecorder",
    "ResearchRunSpec",
    "OutputFrameAnalyzer",
    "find_presentmon",
]
