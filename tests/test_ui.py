from __future__ import annotations

import os
from pathlib import Path
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QThread, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
import pytest

from ui.controller import GuiController, PRESETS, RIFE_MODEL, SRVGG_MODEL, RuntimeConfiguration
from ui.main_window import MainWindow
from ui.settings_store import (
    IRIS_BUFFERED_PROFILE,
    IRIS_RESEARCH_PROFILE,
    GuiSettings,
    SettingsStore,
)
from window_capture import WindowInfo


@pytest.fixture(scope="module")
def qt_app():
    application = QApplication.instance() or QApplication([])
    yield application


def wait_until(qt_app, predicate, timeout_ms: int = 3000) -> None:
    deadline = time.perf_counter() + timeout_ms / 1000.0
    while not predicate() and time.perf_counter() < deadline:
        qt_app.processEvents()
        QTest.qWait(5)
    assert predicate()


def target_windows():
    return [WindowInfo(101, "Example Game"), WindowInfo(202, "Second Window")]


def make_window(qt_app, tmp_path: Path, **kwargs) -> MainWindow:
    window = MainWindow(
        settings_store=SettingsStore(tmp_path / "gui.json"),
        window_provider=kwargs.pop("window_provider", target_windows),
        **kwargs,
    )
    window.show()
    qt_app.processEvents()
    return window


def test_gui_creation_does_not_start_engine(qt_app, tmp_path: Path) -> None:
    controller = GuiController(engine_runner=lambda *args, **kwargs: None)
    window = make_window(qt_app, tmp_path, controller=controller)
    assert window.windowTitle() == "UniversalUpscaler"
    assert window.status_label.text() == "Ready"
    assert controller.running is False
    window.close()


def test_large_resizable_window_and_responsive_workspace(qt_app, tmp_path: Path) -> None:
    window = make_window(qt_app, tmp_path)
    assert window.minimumWidth() == 1050
    assert window.minimumHeight() == 650
    assert window.width() >= 1050
    assert window.height() >= 650
    window._apply_responsive_layout(900)
    assert window.workspace_column_count == 2
    window._apply_responsive_layout(700)
    assert window.workspace_column_count == 1
    window._apply_responsive_layout(900)
    assert window.workspace_column_count == 2
    window.close()


def test_light_theme_is_default(qt_app, tmp_path: Path) -> None:
    window = make_window(qt_app, tmp_path)
    assert window.settings.theme == "light"
    assert window.theme_button.text() == "Dark"
    window.close()


def test_theme_switching_and_persistence(qt_app, tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / "theme.json")
    store.save(GuiSettings(theme="dark"))
    window = MainWindow(settings_store=store, window_provider=target_windows)
    assert window.settings.theme == "dark"
    window.toggle_theme()
    assert window.settings.theme == "light"
    window.close()
    assert store.load().theme == "light"


def test_profile_selection_and_local_persistence(qt_app, tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / "profiles.json")
    window = MainWindow(settings_store=store, window_provider=target_windows)
    window.show()
    window._add_profile("Cinema")
    assert window.settings.selected_profile == "Cinema"
    assert '"Cinema"' in window.profile_heading_label.text()
    window.upscaling_control.set_value("shader")
    window.method_combo.setCurrentIndex(window.method_combo.findData("bicubic"))
    window._select_profile("Default")
    assert window.method_combo.currentData() == "fsr1_like"
    window._select_profile("Cinema")
    assert window.method_combo.currentData() == "bicubic"
    window._rename_profile("Game")
    assert window.settings.selected_profile == "Game"
    assert store.load().profiles == [
        "Default", IRIS_RESEARCH_PROFILE, IRIS_BUFFERED_PROFILE, "Game"
    ]
    window._delete_profile()
    assert window.settings.profiles == [
        "Default", IRIS_RESEARCH_PROFILE, IRIS_BUFFERED_PROFILE
    ]
    assert window.settings.selected_profile == "Default"
    window.close()
    loaded = store.load()
    assert loaded.profiles == [
        "Default", IRIS_RESEARCH_PROFILE, IRIS_BUFFERED_PROFILE
    ]
    assert loaded.selected_profile == "Default"


def test_model_selection_and_ai_adapter_configuration(qt_app, tmp_path: Path) -> None:
    window = make_window(qt_app, tmp_path)
    window.upscaling_control.set_value("ai")
    assert window.model_combo.currentData() == str(SRVGG_MODEL)
    assert window._configuration().model_path == SRVGG_MODEL

    custom_model = tmp_path / "custom_x4.onnx"
    custom_model.touch()
    window._custom_model_path = str(custom_model)
    window.model_combo.setCurrentIndex(window.model_combo.findData("custom"))
    window.ai_scale_combo.setCurrentIndex(window.ai_scale_combo.findData("4"))
    window.ai_input_layout_combo.setCurrentIndex(window.ai_input_layout_combo.findData("nhwc"))
    window.ai_output_layout_combo.setCurrentIndex(window.ai_output_layout_combo.findData("nhwc"))
    window.ai_color_order_combo.setCurrentIndex(window.ai_color_order_combo.findData("bgr"))
    configuration = window._configuration()
    assert configuration.model_path == custom_model
    assert configuration.ai_scale == "4"
    assert configuration.ai_input_layout == "nhwc"
    assert configuration.ai_output_layout == "nhwc"
    assert configuration.ai_color_order == "bgr"
    args = configuration.to_engine_args()
    assert args.model == custom_model
    assert args.ai_scale == "4"
    window.close()


def test_iris_research_profile_and_condition_routing(qt_app, tmp_path: Path) -> None:
    window = make_window(qt_app, tmp_path)
    window._select_profile(IRIS_RESEARCH_PROFILE)
    configuration = window._configuration()
    assert configuration.research_enabled is True
    assert configuration.capture_backend == "wgc"
    assert configuration.upscaling_method == "bicubic"
    assert configuration.frame_generation == "rife"
    assert configuration.generated_frames == 1
    assert configuration.provider == "cpu"
    assert configuration.preset == "performance"
    assert configuration.target_fps == pytest.approx(60.0)
    assert configuration.frame_pacing == "fixed"
    assert configuration.regional_transform is True
    assert configuration.regional_budget_ms == pytest.approx(34.0)
    assert configuration.research_warmup_seconds == pytest.approx(30.0)
    assert configuration.research_duration_seconds == pytest.approx(60.0)

    window.research_condition.setCurrentIndex(
        window.research_condition.findData("capture_presentation")
    )
    assert window._configuration().frame_generation == "off"
    assert window._configuration().regional_transform is False
    window.research_condition.setCurrentIndex(
        window.research_condition.findData("whole_frame_fg")
    )
    assert window._configuration().frame_generation == "rife"
    assert window._configuration().regional_transform is False
    window.research_condition.setCurrentIndex(
        window.research_condition.findData("adaptive_routing")
    )
    assert window._configuration().regional_transform is True
    window.close()


def test_iris_buffered_profile_enables_bounded_extra_candidates(
    qt_app, tmp_path: Path
) -> None:
    window = make_window(qt_app, tmp_path)
    window._select_profile(IRIS_BUFFERED_PROFILE)
    configuration = window._configuration()

    assert configuration.research_enabled is True
    assert configuration.research_condition == "adaptive_routing_buffered"
    assert configuration.generated_frames == 2
    assert configuration.buffered_playout is True
    assert configuration.max_frame_latency_ms == pytest.approx(130.0)
    assert configuration.provider == "directml"
    assert configuration.frame_pacing == "fixed"
    assert configuration.ai_input_width == 160
    assert configuration.ai_input_height == 90
    assert configuration.research_warmup_seconds == pytest.approx(10.0)
    assert configuration.research_duration_seconds == pytest.approx(30.0)
    args = configuration.to_engine_args()
    assert args.buffered_playout is True
    window.close()


def test_buffered_profile_refresh_preserves_local_adapter(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / "buffered.json")
    settings = store.load()
    settings.profile_settings[IRIS_BUFFERED_PROFILE].update(
        {"provider": "cpu", "rife_device_id": 1, "frame_pacing": "auto"}
    )
    store.save(settings)
    buffered = store.load().profile_settings[IRIS_BUFFERED_PROFILE]
    assert buffered["provider"] == "directml"
    assert buffered["rife_device_id"] == 1
    assert buffered["frame_pacing"] == "fixed"


def test_workspace_cards_keep_content_height_in_two_column_layout(
    qt_app, tmp_path: Path
) -> None:
    window = make_window(qt_app, tmp_path)
    window.resize(1450, 900)
    window.show()
    qt_app.processEvents()
    window._select_profile(IRIS_RESEARCH_PROFILE)
    window._apply_responsive_layout(1100)
    qt_app.processEvents()

    assert window.workspace_column_count == 2
    for card in (
        window.frame_card,
        window.rendering_card,
        window.research_card,
    ):
        assert card.height() >= card.sizeHint().height()

    assert window.research_card.height() > window.status_card.height()
    window.close()


def test_full_shader_fsr_capture_and_pipeline_controls(qt_app, tmp_path: Path) -> None:
    window = make_window(qt_app, tmp_path)
    window.upscaling_control.set_value("shader")
    methods = [window.method_combo.itemData(index) for index in range(window.method_combo.count())]
    assert methods == ["nearest", "bilinear", "bicubic", "lanczos", "fsr1_like"]
    window.method_combo.setCurrentIndex(window.method_combo.findData("fsr1_like"))
    assert window.fsr_sharpening_row.isVisible()
    assert window.fsr_edge_strength_row.isVisible()
    window.fsr_sharpening.setChecked(False)
    assert not window.fsr_sharpening_strength_row.isVisible()
    window.capture_backend_combo.setCurrentIndex(window.capture_backend_combo.findData("wgc"))
    window.pipeline_combo.setCurrentIndex(window.pipeline_combo.findData("d3d11"))
    configuration = window._configuration()
    configuration.validate()
    args = configuration.to_engine_args()
    assert args.pipeline == "d3d11"
    assert args.capture_backend == "wgc"
    assert args.fsr1_like_sharpening is False
    assert args.no_preview is False
    assert args.preview_fullscreen is True
    assert args.draw_preview_overlay is True
    assert args.persist_runtime_profile is True
    assert args.keep_latest_real is True
    window.close()


def test_pipeline_compatibility_is_explained_inline(qt_app, tmp_path: Path) -> None:
    window = make_window(qt_app, tmp_path)
    window.upscaling_control.set_value("ai")
    window.pipeline_combo.setCurrentIndex(window.pipeline_combo.findData("d3d11"))
    qt_app.processEvents()
    assert "requires Shader upscaling" in window.compatibility_label.text()
    assert not window.start_button.isEnabled()
    window.close()


def test_rife_model_and_detailed_telemetry(qt_app, tmp_path: Path) -> None:
    window = make_window(qt_app, tmp_path)
    window.frame_generation_control.set_value("rife")
    assert window.rife_model_combo.currentData() == str(RIFE_MODEL)
    window.generated_frames_combo.setCurrentIndex(
        window.generated_frames_combo.findData(4)
    )
    assert window._configuration().generated_frames == 4
    args = window._configuration().to_engine_args()
    assert (args.rife_input_width, args.rife_input_height) == (320, 180)
    assert args.max_frame_latency_ms == pytest.approx(100.0)
    window.status_details_button.setChecked(True)
    window._telemetry_changed(
        {
            "presentation_fps": 59.9,
            "total_frame_ms": 14.0,
            "capture_ms": 2.0,
            "inference_ms": 7.0,
            "interpolation_ms": 5.0,
            "p95_frame_ms": 18.0,
            "pacing_error_ms": 0.4,
            "estimated_source_fps": 30.0,
            "active_provider": "DmlExecutionProvider",
            "dropped_frames": 1,
            "ai_input_dimensions": (320, 180),
            "ai_output_dimensions": (640, 360),
            "tile_mode": "auto (256px)",
            "successful_recoveries": 2,
            "fallback_activations": 1,
            "presented_real_frames": 30,
            "presented_generated_frames": 120,
            "presented_frames": 150,
        }
    )
    assert window.status_details_panel.isVisible()
    assert window.inference_metric.value() == "7.0 ms"
    assert window.dimensions_metric.value() == "320×180 → 640×360"
    assert window.recovery_metric.value() == "2 / 1"
    assert window.frame_count_metric.value() == "30 / 120"
    window.close()


def test_restored_rife_profile_shows_effective_latency(qt_app, tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / "rife-latency.json")
    store.save(
        GuiSettings(
            frame_generation="rife",
            performance_preset="performance",
            max_frame_latency_ms=100.0,
        )
    )

    window = MainWindow(settings_store=store, window_provider=target_windows)

    assert window.max_latency.value() == pytest.approx(100.0)
    assert window._configuration().to_engine_args().max_frame_latency_ms == pytest.approx(
        100.0
    )
    window.close()


def test_warmup_countdown_is_shown_on_start_button(qt_app, tmp_path: Path) -> None:
    window = make_window(qt_app, tmp_path)
    window.controller._running = True

    window._state_changed("warming_up_5")
    assert window.status_label.text() == "Warming up · 5s"
    assert window.start_button.text() == "5"

    window._state_changed("running")
    assert window.start_button.text() == "Stop · Esc"
    window.controller._running = False
    window.close()


def test_escape_shortcut_stops_a_running_session(qt_app, tmp_path: Path, monkeypatch) -> None:
    window = make_window(qt_app, tmp_path)
    stop_calls = []
    window.controller._running = True
    monkeypatch.setattr(window.controller, "stop", lambda: stop_calls.append(True) or True)

    QTest.keyClick(window, Qt.Key_Escape)
    qt_app.processEvents()

    assert stop_calls == [True]
    window.controller._running = False
    window.close()


def test_settings_persist_without_hwnd(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    values = GuiSettings(
        upscaling_mode="ai",
        upscaling_method="srvgg",
        frame_generation="rife",
        generated_frames=4,
        performance_preset="quality",
        target_fps="144",
        advanced_visible=True,
    )
    store.save(values)
    loaded = store.load()
    assert loaded.upscaling_mode == "ai"
    assert loaded.frame_generation == "rife"
    assert loaded.generated_frames == 4
    assert loaded.performance_preset == "quality"
    assert "hwnd" not in store.path.read_text(encoding="utf-8").lower()


def test_window_refresh_and_disappearing_selection(qt_app, tmp_path: Path) -> None:
    available = target_windows()
    window = make_window(qt_app, tmp_path, window_provider=lambda: list(available))
    window.window_combo.setCurrentIndex(1)
    assert window.window_combo.currentData() == 202
    available.pop()
    window.refresh_windows()
    assert window.window_combo.currentIndex() == -1
    assert "no longer available" in window.status_label.text()
    assert window.start_button.isEnabled() is False
    window.close()


def test_conditional_controls_and_preset_mapping(qt_app, tmp_path: Path) -> None:
    window = make_window(qt_app, tmp_path)
    window.upscaling_control.set_value("ai")
    qt_app.processEvents()
    assert window.ai_card.isVisible()
    assert window.ai_warning.isVisible()
    assert window.method_combo.currentData() == "srvgg"
    window.preset_combo.setCurrentIndex(window.preset_combo.findData("quality"))
    assert window.ai_width.value() == PRESETS["quality"]["width"]
    assert window.ai_height.value() == PRESETS["quality"]["height"]
    assert window.max_latency.value() == PRESETS["quality"]["latency"]
    window.upscaling_control.set_value("shader")
    assert not window.ai_card.isVisible()
    assert not window.ai_warning.isVisible()
    window.frame_generation_control.set_value("rife")
    qt_app.processEvents()
    assert window.rife_type_row.isVisible()
    assert window.rife_provider_row.isVisible()
    window.frame_generation_control.set_value("off")
    assert not window.rife_type_row.isVisible()
    assert not window.rife_provider_row.isVisible()
    window.close()


@pytest.mark.parametrize(
    "configuration",
    [
        RuntimeConfiguration(hwnd=0),
        RuntimeConfiguration(hwnd=1, target_fps=0),
        RuntimeConfiguration(hwnd=1, queue_depth=0),
        RuntimeConfiguration(hwnd=1, ai_input_width=0),
        RuntimeConfiguration(hwnd=1, rife_device_id=-1),
    ],
)
def test_invalid_configuration_is_rejected(configuration) -> None:
    with pytest.raises(ValueError):
        configuration.validate()


def test_start_enabled_only_with_valid_target(qt_app, tmp_path: Path) -> None:
    window = make_window(qt_app, tmp_path, window_provider=lambda: [])
    assert window.start_button.isEnabled() is False
    window.window_provider = target_windows
    window.refresh_windows()
    assert window.start_button.isEnabled() is True
    window.close()


def test_duplicate_start_stop_state_telemetry_and_gui_thread_delivery(qt_app) -> None:
    started = threading.Event()
    stopped = threading.Event()

    def engine(args, *, shutdown_event, state_callback, telemetry_callback):
        del args
        state_callback("running")
        telemetry_callback({
            "presentation_fps": 60.0,
            "total_frame_ms": 12.5,
            "active_provider": "CPUExecutionProvider",
            "dropped_frames": 2,
        })
        started.set()
        shutdown_event.wait(2.0)
        state_callback("stopped")
        stopped.set()

    controller = GuiController(engine_runner=engine)
    states = []
    telemetry = []
    delivery_threads = []
    controller.state_changed.connect(states.append)
    controller.telemetry_changed.connect(lambda value: (telemetry.append(value), delivery_threads.append(QThread.currentThread())))
    configuration = RuntimeConfiguration(hwnd=101)
    assert controller.start(configuration) is True
    assert controller.start(configuration) is False
    wait_until(qt_app, started.is_set)
    wait_until(qt_app, lambda: bool(telemetry))
    assert delivery_threads == [qt_app.thread()]
    assert controller.stop() is True
    wait_until(qt_app, stopped.is_set)
    wait_until(qt_app, lambda: not controller.running)
    assert "running" in states and "stopping" in states and "stopped" in states
    assert telemetry[0]["presentation_fps"] == 60.0


def test_user_facing_error_and_clean_close_while_running(qt_app, tmp_path: Path) -> None:
    started = threading.Event()

    def engine(args, *, shutdown_event, state_callback, telemetry_callback):
        del args, telemetry_callback
        state_callback("running"); started.set(); shutdown_event.wait(2.0)

    controller = GuiController(engine_runner=engine)
    window = make_window(qt_app, tmp_path, controller=controller)
    window._start_stop()
    wait_until(qt_app, started.is_set)
    controller.error_occurred.emit("Selected window closed.")
    qt_app.processEvents()
    assert "Selected window closed" in window.error_label.text()
    window.close()
    wait_until(qt_app, lambda: not controller.running)
    wait_until(qt_app, lambda: not window.isVisible())


def test_engine_exception_is_reported_and_runtime_finishes(qt_app) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("provider initialization failed")

    controller = GuiController(engine_runner=fail)
    errors = []
    states = []
    controller.error_occurred.connect(errors.append)
    controller.state_changed.connect(states.append)
    assert controller.start(RuntimeConfiguration(hwnd=101))
    wait_until(qt_app, lambda: not controller.running)
    assert any("provider initialization failed" in error for error in errors)
    assert "failed" in states


def test_auto_target_preserves_pacing_off():
    args = RuntimeConfiguration(hwnd=101, frame_pacing="off").to_engine_args()
    assert args.frame_pacing == "off"


def test_fixed_pacing_requires_target():
    with pytest.raises(ValueError, match="requires a target FPS"):
        RuntimeConfiguration(hwnd=101, frame_pacing="fixed").validate()


def test_rife_respects_explicit_latency():
    args = RuntimeConfiguration(hwnd=101, frame_generation="rife",
                                max_frame_latency_ms=75).to_engine_args()
    assert args.max_frame_latency_ms == 75


def test_frame_generation_device_can_be_split_from_ai_device(qt_app, tmp_path):
    window = make_window(qt_app, tmp_path)
    window.device_id.setValue(1)
    window.rife_device_id.setValue(0)

    args = window._configuration().to_engine_args()

    assert args.ai_device_id == 1
    assert args.rife_device_id == 0
    window.close()
    restored = SettingsStore(tmp_path / "gui.json").load()
    assert restored.device_id == 1
    assert restored.rife_device_id == 0


def test_output_refinement_is_saved_and_reaches_runtime(qt_app,tmp_path):
    window=make_window(qt_app,tmp_path)
    window.output_refinement_combo.setCurrentIndex(window.output_refinement_combo.findData(.12))
    assert window._configuration().to_engine_args().output_refinement==.12
    window.close()
    assert SettingsStore(tmp_path/"gui.json").load().output_refinement==.12


def test_temporal_stabilization_is_saved_and_reaches_runtime(qt_app, tmp_path):
    window = make_window(qt_app, tmp_path)
    assert window.temporal_stabilization.isChecked()
    assert window._configuration().to_engine_args().temporal_stabilization is True
    window.temporal_stabilization.setChecked(False)
    assert window._configuration().to_engine_args().temporal_stabilization is False
    window.close()
    assert SettingsStore(tmp_path / "gui.json").load().temporal_stabilization is False


def test_regional_transform_is_saved_and_reaches_runtime(qt_app, tmp_path):
    window = make_window(qt_app, tmp_path)
    window.frame_generation_control.set_value("rife")
    window.regional_transform.setChecked(True)
    window.regional_budget.setValue(12.5)
    window.regional_debug_overlay.setChecked(True)
    args = window._configuration().to_engine_args()
    assert args.regional_transform is True
    assert args.regional_budget_ms == pytest.approx(12.5)
    assert args.regional_debug_overlay is True
    window._save_settings()
    loaded = SettingsStore(tmp_path / "gui.json").load()
    assert loaded.regional_transform is True
    assert loaded.regional_budget_ms == pytest.approx(12.5)
    assert loaded.regional_debug_overlay is True
    window.close()


def test_rife_lite_selection_persists_and_reaches_runtime(qt_app,tmp_path):
    from ui.controller import RIFE_LITE_MODEL
    window=make_window(qt_app,tmp_path)
    window.frame_generation_control.set_value("rife")
    window.rife_model_combo.setCurrentIndex(window.rife_model_combo.findData(str(RIFE_LITE_MODEL)))
    assert window._configuration().to_engine_args().rife_model==RIFE_LITE_MODEL
    window.close()
    assert SettingsStore(tmp_path/"gui.json").load().rife_model_path==str(RIFE_LITE_MODEL)


def test_ifrnet_selection_persists_and_reaches_runtime(qt_app,tmp_path):
    from ui.controller import IFRNET_MODEL
    window=make_window(qt_app,tmp_path)
    window.frame_generation_control.set_value("rife")
    window.rife_model_combo.setCurrentIndex(window.rife_model_combo.findData(str(IFRNET_MODEL)))
    assert window._configuration().to_engine_args().rife_model==IFRNET_MODEL
    window.close()
    assert SettingsStore(tmp_path/"gui.json").load().rife_model_path==str(IFRNET_MODEL)


def test_fast_quality_preset_uses_240_by_135_interpolation(qt_app, tmp_path: Path) -> None:
    window = make_window(qt_app, tmp_path)
    window.frame_generation_control.set_value("rife")
    index = window.preset_combo.findData("fast_quality")
    assert index >= 0
    window.preset_combo.setCurrentIndex(index)
    configuration = window._configuration()
    assert configuration.preset == "fast_quality"
    args = configuration.to_engine_args()
    assert (args.rife_input_width, args.rife_input_height) == (240, 135)
    assert window.ai_width.value() == 240
    assert window.ai_height.value() == 135
    window.close()
