from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from research_run import OutputFrameAnalyzer, ResearchRunRecorder, ResearchRunSpec


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class FakeProbe:
    def sample(self):
        return {
            "afr_cpu_percent": 12.0,
            "afr_ram_mb": 200.0,
            "game_cpu_percent": 30.0,
            "game_ram_mb": 500.0,
            "gpu_usage_percent": None,
            "dedicated_vram_mb": None,
            "gpu_temperature_c": None,
            "gpu_power_w": None,
            "gpu_source": "unavailable",
        }


def frame(sequence_id: int, *, kind: str = "generated"):
    return SimpleNamespace(
        sequence_id=sequence_id,
        frame_kind=kind,
        generation_method="rife" if kind == "generated" else "real",
        generation_position=0.5 if kind == "generated" else 1.0,
        active_provider="CPUExecutionProvider",
        interpolation_provider="CPUExecutionProvider",
        frame_generation="rife",
        captured_at=float(sequence_id),
        processing_started_at=float(sequence_id) + 0.002,
        processed_at=float(sequence_id) + 0.012,
        capture_ms=2.0,
        processing_ms=10.0,
        interpolation_ms=8.0 if kind == "generated" else None,
        regional_route_counts=(
            {"smooth_motion": 6, "hud_text": 2} if kind == "generated" else None
        ),
        regional_provenance_counts=(
            {"interpolated": 5, "blended": 1, "real": 2}
            if kind == "generated"
            else None
        ),
        regional_admission_reason=("within_budget" if kind == "generated" else None),
    )


def pacing(actual: float, *, generated_drops: int = 0):
    return SimpleNamespace(
        estimated_source_fps=30.0,
        presentation_fps=60.0,
        generated_frames_dropped_late=generated_drops,
        real_frames_dropped_late=0,
        late_frames=1,
    )


def recovery():
    return SimpleNamespace(
        state="running",
        retry_attempts=0,
        successful_recoveries=0,
        failed_recoveries=0,
        fallback_activations=0,
    )


def pipeline(*, input_replacements: int):
    return SimpleNamespace(
        input_replacements=input_replacements,
        result_replacements=1,
        dropped_generated_frames=0,
    )


def decision(actual: float):
    return SimpleNamespace(
        actual_at=actual,
        scheduled_at=actual - 0.001,
        pacing_error_ms=1.0,
    )


def test_research_run_records_only_measurement_and_summarizes(tmp_path: Path) -> None:
    clock = FakeClock()
    spec = ResearchRunSpec(
        root=tmp_path,
        machine_id="Laptop A",
        application="Portal",
        scene_id="fast_turn",
        condition="adaptive_routing",
        repetition=1,
        warmup_seconds=2.0,
        duration_seconds=5.0,
    )
    recorder = ResearchRunRecorder(
        spec,
        project_root=tmp_path,
        hwnd=0,
        configuration={"rife_model": None},
        clock=clock,
        resource_probe=FakeProbe(),
    )
    run_directory = recorder.prepare()
    recorder.begin_timing(sample_resources=False)

    assert not recorder.record_frame(
        processed_frame=frame(0),
        decision=decision(1.0),
        pacing_snapshot=pacing(1.0),
        recovery_snapshot=recovery(),
        pipeline=pipeline(input_replacements=4),
    )
    assert recorder.record_frame(
        processed_frame=frame(1),
        decision=decision(3.0),
        pacing_snapshot=pacing(3.0, generated_drops=2),
        recovery_snapshot=recovery(),
        pipeline=pipeline(input_replacements=5),
        visual_marker_id=7,
    )
    assert recorder.record_frame(
        processed_frame=frame(2, kind="real"),
        decision=decision(4.0),
        pacing_snapshot=pacing(4.0, generated_drops=3),
        recovery_snapshot=recovery(),
        pipeline=pipeline(input_replacements=7),
    )
    summary = recorder.close()

    assert summary["measured_frames"] == 2
    assert summary["presented_frame_counts"] == {"generated": 1, "real": 1}
    assert summary["drops"]["input_queue_replacements"] == 2
    assert summary["drops"]["generated_deadline"] == 1
    assert summary["capture_to_submission_ms"]["p99"] is not None
    assert summary["regional_provenance_counts"]["interpolated"] == 5
    assert (run_directory / "manifest.json").is_file()
    assert (run_directory / "summary.json").is_file()
    assert (run_directory / "raw.log").is_file()
    assert (run_directory / "observations.md").is_file()
    with (run_directory / "telemetry_frames.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 2
    assert rows[0]["active_provider"] == "CPUExecutionProvider"
    assert rows[0]["visual_marker_id"] == "7"
    assert json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))["status"] == "complete"


def test_research_run_never_overwrites_existing_directory(tmp_path: Path) -> None:
    spec = ResearchRunSpec(
        root=tmp_path,
        machine_id="machine",
        application="game",
        scene_id="scene",
        condition="condition",
        repetition=1,
    )
    first = ResearchRunRecorder(spec, project_root=tmp_path, resource_probe=FakeProbe())
    first.prepare()
    first.close(status="interrupted")
    second = ResearchRunRecorder(spec, project_root=tmp_path, resource_probe=FakeProbe())
    with pytest.raises(FileExistsError, match="will not be overwritten"):
        second.prepare()


def test_presentmon_can_be_deferred_until_presenter_exists(
    tmp_path: Path, monkeypatch
) -> None:
    clock = FakeClock()
    spec = ResearchRunSpec(
        root=tmp_path,
        machine_id="m",
        application="a",
        scene_id="s",
        condition="c",
        repetition=1,
    )
    recorder = ResearchRunRecorder(
        spec, project_root=tmp_path, clock=clock, resource_probe=FakeProbe()
    )
    recorder.prepare()
    starts = []
    monkeypatch.setattr(recorder, "_start_presentmon", lambda: starts.append(True))
    recorder.begin_timing(sample_resources=False, start_presentmon=False)
    assert starts == []
    recorder.start_presentmon()
    assert starts == [True]
    recorder.close(status="interrupted")


def test_output_analyzer_separates_calls_unique_and_meaningful_motion() -> None:
    analyzer = OutputFrameAnalyzer(motion_threshold=1.0)
    black = np.zeros((72, 128, 3), dtype=np.uint8)
    tiny_change = black.copy()
    tiny_change[0, 0] = 1
    moving = np.full((72, 128, 3), 20, dtype=np.uint8)

    first = analyzer.observe(black, 1.0)
    repeated = analyzer.observe(black, 1.016)
    analyzer.observe(tiny_change, 1.032)
    changed = analyzer.observe(moving, 1.048)
    snapshot = analyzer.snapshot(1.048)

    assert first["exact_unique"] == 1
    assert repeated["exact_unique"] == 0
    assert repeated["meaningful_motion"] == 0
    assert changed["meaningful_motion"] == 1
    assert snapshot["calls"] == 4
    assert snapshot["meaningful"] == 2
