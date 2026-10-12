"""Small JSON-backed GUI preference store."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path
import platform

from resource_paths import resource_path, user_data_dir


IRIS_RESEARCH_PROFILE = "IRIS Research 720p30"
IRIS_BUFFERED_PROFILE = "IRIS Buffered Playout 720p30"


@dataclass(slots=True)
class GuiSettings:
    theme: str = "light"
    profiles: list[str] = field(default_factory=lambda: ["Default"])
    selected_profile: str = "Default"
    profile_settings: dict[str, dict] = field(default_factory=dict)
    upscaling_mode: str = "shader"
    upscaling_method: str = "fsr1_like"
    frame_generation: str = "off"
    generated_frames: int = 1
    buffered_playout: bool = False
    rife_model_path: str = ""
    performance_preset: str = "balanced"
    target_fps: str = "auto"
    custom_fps: float = 60.0
    advanced_visible: bool = False
    provider: str = "directml"
    device_id: int = 0
    rife_device_id: int | None = None
    ai_tile: str = "auto"
    ai_tile_overlap: int = 16
    ai_input_width: int = 320
    ai_input_height: int = 180
    frame_pacing: str = "auto"
    max_frame_latency_ms: float = 100.0
    queue_depth: int = 2
    allow_provider_fallback: bool = False
    show_performance_overlay: bool = True
    output_refinement: float = 0.0
    temporal_stabilization: bool = True
    regional_transform: bool = False
    regional_budget_ms: float = 16.0
    regional_debug_overlay: bool = False
    ai_model_path: str = ""
    ai_scale: str = "2"
    ai_input_layout: str = "nchw"
    ai_output_layout: str = "nchw"
    ai_color_order: str = "rgb"
    capture_backend: str = "wgc"
    pipeline: str = "cpu"
    fsr1_like_sharpening: bool = True
    fsr1_like_sharpening_strength: float = 0.2
    fsr1_like_edge_strength: float = 0.35
    status_details_visible: bool = False
    research_enabled: bool = False
    research_root: str = ""
    research_machine_id: str = ""
    research_application: str = "application"
    research_scene: str = "scene_01"
    research_condition: str = "adaptive_routing"
    research_repetition: int = 1
    research_warmup_seconds: float = 30.0
    research_duration_seconds: float = 60.0


def _ensure_iris_profile(settings: GuiSettings) -> GuiSettings:
    """Add the reproducible research preset and enforce its cadence contract."""

    if IRIS_RESEARCH_PROFILE not in settings.profiles:
        settings.profiles.append(IRIS_RESEARCH_PROFILE)
    if IRIS_RESEARCH_PROFILE not in settings.profile_settings:
        settings.profile_settings[IRIS_RESEARCH_PROFILE] = {
            "upscaling_mode": "shader",
            "upscaling_method": "bicubic",
            "frame_generation": "rife",
            "generated_frames": 1,
            "rife_model_path": str(resource_path("models", "RIFE_v4.25_lite.onnx")),
            "performance_preset": "performance",
            "target_fps": "60",
            "custom_fps": 60.0,
            "provider": "cpu",
            "device_id": 0,
            "rife_device_id": 0,
            "frame_pacing": "fixed",
            "max_frame_latency_ms": 94.0,
            "queue_depth": 2,
            "allow_provider_fallback": False,
            "show_performance_overlay": True,
            "output_refinement": 0.0,
            "temporal_stabilization": True,
            "regional_transform": True,
            "regional_budget_ms": 34.0,
            "regional_debug_overlay": False,
            "capture_backend": "wgc",
            "pipeline": "cpu",
            "research_enabled": True,
            "research_root": str(resource_path("data", "raw")),
            "research_machine_id": platform.node() or "machine",
            "research_application": "application",
            "research_scene": "scene_01",
            "research_condition": "adaptive_routing",
            "research_repetition": 1,
            "research_warmup_seconds": 30.0,
            "research_duration_seconds": 60.0,
        }
    else:
        # Research comparisons require the same explicit output cadence on
        # every machine. Earlier copies used Auto, which let queue replacement
        # feed back into the estimated source rate and collapse output cadence.
        settings.profile_settings[IRIS_RESEARCH_PROFILE].update(
            {"target_fps": "60", "custom_fps": 60.0, "frame_pacing": "fixed"}
        )
    if IRIS_BUFFERED_PROFILE not in settings.profiles:
        settings.profiles.append(IRIS_BUFFERED_PROFILE)
    buffered = dict(settings.profile_settings[IRIS_RESEARCH_PROFILE])
    buffered.update(
        {
            "generated_frames": 2,
            "buffered_playout": True,
            "max_frame_latency_ms": 130.0,
            "research_condition": "adaptive_routing_buffered",
        }
    )
    settings.profile_settings[IRIS_BUFFERED_PROFILE] = buffered
    return settings


class SettingsStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = (
            Path(path)
            if path is not None
            else user_data_dir() / "gui_settings.json"
        )

    def load(self) -> GuiSettings:
        if not self.path.exists():
            return _ensure_iris_profile(GuiSettings())

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return _ensure_iris_profile(GuiSettings())

        allowed = {field.name for field in fields(GuiSettings)}
        values = {
            key: value
            for key, value in payload.items()
            if key in allowed
        }

        try:
            settings = GuiSettings(**values)
        except TypeError:
            return _ensure_iris_profile(GuiSettings())

        if settings.theme not in {"light", "dark"}:
            settings.theme = "light"

        if not isinstance(settings.profiles, list):
            settings.profiles = ["Default"]
        settings.profiles = [
            str(name).strip()
            for name in settings.profiles
            if str(name).strip()
        ]
        if "Default" not in settings.profiles:
            settings.profiles.insert(0, "Default")
        settings.profiles = list(dict.fromkeys(settings.profiles))
        if settings.selected_profile not in settings.profiles:
            settings.selected_profile = "Default"
        if not isinstance(settings.profile_settings, dict):
            settings.profile_settings = {}
        settings.profile_settings = {
            str(name): values
            for name, values in settings.profile_settings.items()
            if str(name) in settings.profiles and isinstance(values, dict)
        }

        return _ensure_iris_profile(settings)

    def save(self, settings: GuiSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(asdict(settings), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self.path)
