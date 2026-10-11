"""Responsive native PySide6 interface for UniversalUpscaler."""

from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path

from PySide6.QtCore import QTimer, QUrl, Qt
from PySide6.QtGui import QCloseEvent, QDesktopServices, QKeySequence, QResizeEvent, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from window_capture import WindowCaptureError, get_client_capture_region, list_visible_windows
from ui.controller import (
    GuiController,
    PRESETS,
    PROJECT_ROOT,
    RIFE_MODEL,
    RIFE_LITE_MODEL,
    IFRNET_MODEL,
    SRVGG_MODEL,
    RuntimeConfiguration,
)
from ui.settings_store import GuiSettings, SettingsStore
from ui.theme import apply_theme
from ui.widgets import (
    AccentButton,
    CompactStatusRow,
    ModeBinding,
    SettingsCard,
    SettingsRow,
    SidebarItem,
    Switch,
)
from resource_paths import user_data_dir


def _combo(items: tuple[tuple[str, str], ...], accessible_name: str = "") -> QComboBox:
    combo = QComboBox()
    for value, label in items:
        combo.addItem(label, value)
    if accessible_name:
        combo.setAccessibleName(accessible_name)
    return combo


class MainWindow(QMainWindow):
    def __init__(
        self,
        *,
        settings_store: SettingsStore | None = None,
        controller: GuiController | None = None,
        window_provider=list_visible_windows,
    ) -> None:
        super().__init__()
        self.settings_store = settings_store or SettingsStore()
        self.settings = self.settings_store.load()
        self.controller = controller or GuiController(parent=self)
        self.window_provider = window_provider
        self._loading = True
        self._logs: list[str] = []
        self._profile_buttons: list[SidebarItem] = []
        self._custom_model_path = self.settings.ai_model_path.strip()
        self.workspace_column_count = 2
        self._workspace_layout_signature: tuple[int, ...] = ()

        self.setWindowTitle("UniversalUpscaler")
        self.setMinimumSize(1050, 650)
        self.resize(1400, 850)
        self._build_ui()
        self._connect()
        self._restore_settings()
        self._loading = False
        self.refresh_windows()
        self._update_visibility()
        self._update_start_enabled()
        apply_theme(QApplication.instance(), self.settings.theme)
        QTimer.singleShot(0, self._apply_responsive_layout)

    # ---- shell ---------------------------------------------------------
    def _build_ui(self) -> None:
        shell = QWidget()
        shell.setObjectName("appShell")
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)
        shell_layout.addWidget(self._build_top_bar())

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        body_layout.addWidget(self._build_sidebar())
        self.view_stack = QStackedWidget()
        self.workspace_page = self._build_workspace_page()
        self.general_page = self._build_general_page()
        self.logs_page = self._build_logs_page()
        self.about_page = self._build_about_page()
        for page in (self.workspace_page, self.general_page, self.logs_page, self.about_page):
            self.view_stack.addWidget(page)
        body_layout.addWidget(self.view_stack, 1)
        shell_layout.addWidget(body, 1)
        self.setCentralWidget(shell)

    def _build_top_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("topBar")
        bar.setFixedHeight(58)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(9)
        brand = QLabel("UniversalUpscaler")
        brand.setObjectName("brandLabel")
        version = QLabel("v0.1")
        version.setObjectName("versionLabel")
        self.theme_button = QPushButton("Dark")
        self.theme_button.setObjectName("quietButton")
        self.theme_button.setAccessibleName("Toggle light and dark theme")
        self.start_button = AccentButton("Start")
        self.start_button.setDefault(True)
        self.start_button.setAccessibleName("Start UniversalUpscaler")
        layout.addWidget(brand)
        layout.addWidget(version)
        layout.addStretch(1)
        layout.addWidget(self.theme_button)
        layout.addWidget(self.start_button)
        return bar

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(196)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(14, 18, 14, 16)
        layout.setSpacing(7)
        caption = QLabel("Profiles")
        caption.setObjectName("sidebarCaption")
        layout.addWidget(caption)
        self.profile_list_layout = QVBoxLayout()
        self.profile_list_layout.setContentsMargins(0, 4, 0, 0)
        self.profile_list_layout.setSpacing(3)
        layout.addLayout(self.profile_list_layout)
        layout.addStretch(1)

        actions = QHBoxLayout()
        actions.setSpacing(5)
        self.add_profile_button = QPushButton("+ New")
        self.rename_profile_button = QPushButton("Rename")
        self.delete_profile_button = QPushButton("Delete")
        for button in (self.add_profile_button, self.rename_profile_button, self.delete_profile_button):
            button.setObjectName("sidebarAction")
            actions.addWidget(button)
        layout.addLayout(actions)
        layout.addSpacing(10)

        self.general_nav = SidebarItem("General")
        self.logs_nav = SidebarItem("Logs")
        self.about_nav = SidebarItem("About")
        for button in (self.general_nav, self.logs_nav, self.about_nav):
            layout.addWidget(button)
        self._rebuild_profile_buttons()
        return sidebar

    # ---- workspace -----------------------------------------------------
    def _build_workspace_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("workspacePage")
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)

        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(28, 20, 28, 12)
        header_layout.setSpacing(3)
        self.profile_heading_label = QLabel('Profile: "Default"')
        self.profile_heading_label.setObjectName("pageTitle")
        description = QLabel("Configure capture, enhancement, and presentation.")
        description.setObjectName("pageDescription")
        self.compatibility_label = QLabel("")
        self.compatibility_label.setObjectName("compatibilityLabel")
        self.compatibility_label.setWordWrap(True)
        header_layout.addWidget(self.profile_heading_label)
        header_layout.addWidget(description)
        header_layout.addWidget(self.compatibility_label)
        page_layout.addWidget(header)

        self.settings_scroll = QScrollArea()
        self.settings_scroll.setObjectName("settingsScroll")
        self.settings_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.settings_scroll.setWidgetResizable(True)
        self.settings_canvas = QWidget()
        canvas_layout = QVBoxLayout(self.settings_canvas)
        canvas_layout.setContentsMargins(28, 4, 28, 28)
        canvas_layout.setSpacing(0)
        self.workspace_grid = QGridLayout()
        self.workspace_grid.setHorizontalSpacing(10)
        self.workspace_grid.setVerticalSpacing(10)
        self.workspace_grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        canvas_layout.addLayout(self.workspace_grid)
        canvas_layout.addStretch(1)
        self.settings_scroll.setWidget(self.settings_canvas)
        page_layout.addWidget(self.settings_scroll, 1)

        self.frame_card = self._build_frame_generation_card()
        self.upscaling_card = self._build_upscaling_card()
        self.capture_card = self._build_capture_card()
        self.rendering_card = self._build_rendering_card()
        self.ai_card = self._build_ai_card()
        self.status_card = self._build_status_card()
        self._workspace_cards = (
            self.frame_card,
            self.upscaling_card,
            self.capture_card,
            self.rendering_card,
            self.ai_card,
            self.status_card,
        )
        self._apply_responsive_layout(1000)
        return page

    def _build_frame_generation_card(self) -> SettingsCard:
        card = SettingsCard("Frame Generation")
        self.frame_generation_enabled = Switch()
        self.frame_generation_enabled.setAccessibleName("Enable frame generation")
        card.add_row(SettingsRow("Enabled", self.frame_generation_enabled))
        self.frame_type_combo = _combo((("rife", "ONNX interpolation"),), "Frame generation type")
        self.rife_type_row = card.add_row(SettingsRow("Type", self.frame_type_combo))
        self.rife_model_combo = _combo(
            ((str(RIFE_MODEL), "RIFE v3.6 (recommended)"),
             (str(RIFE_LITE_MODEL), "RIFE 4.25 Lite (experimental)"),
             (str(IFRNET_MODEL), "IFRNet-S Vimeo90K (experimental)")),
            "Frame generation model"
        )
        self.rife_model_row = card.add_row(SettingsRow("Model", self.rife_model_combo))
        self.rife_provider_combo = _combo(
            (("directml", "DirectML"), ("cpu", "CPU")), "Frame generation provider"
        )
        self.rife_provider_row = card.add_row(SettingsRow("Provider", self.rife_provider_combo))
        self.rife_device_id = QSpinBox()
        self.rife_device_id.setRange(0, 64)
        self.rife_device_id.setToolTip(
            "DirectML adapter used only for frame generation. On dual-GPU laptops, "
            "try the iGPU here to leave the game GPU free."
        )
        self.rife_device_id.setAccessibleName("Frame generation device")
        self.rife_device_row = card.add_row(SettingsRow("Device", self.rife_device_id))
        self.generated_frames_combo = _combo(
            (
                (1, "1× (1 generated per real)"),
                (2, "2× (2 generated per real)"),
                (3, "3× (3 generated per real)"),
                (4, "4× (4 generated per real)"),
            ),
            "Generated frames per real frame",
        )
        self.generated_frames_row = card.add_row(
            SettingsRow("Generated amount", self.generated_frames_combo)
        )
        self.rife_fps_combo = self._build_fps_combo()
        card.add_row(SettingsRow("Target FPS", self.rife_fps_combo))
        self.rife_pacing_combo = self._build_pacing_combo()
        card.add_row(SettingsRow("Frame pacing", self.rife_pacing_combo))
        self.frame_generation_control = ModeBinding(
            self.frame_generation_enabled, self.frame_type_combo, off="off", parent=self
        )
        return card

    def _build_upscaling_card(self) -> SettingsCard:
        card = SettingsCard("Upscaling")
        self.upscaling_enabled = Switch()
        self.upscaling_enabled.setAccessibleName("Enable upscaling")
        card.add_row(SettingsRow("Enabled", self.upscaling_enabled))
        self.upscaling_type_combo = _combo(
            (("shader", "Shader"), ("ai", "AI")), "Upscaling type"
        )
        self.upscaling_type_row = card.add_row(SettingsRow("Type", self.upscaling_type_combo))
        self.method_combo = QComboBox()
        self.method_combo.setAccessibleName("Upscaling method")
        self.method_row = card.add_row(SettingsRow("Method", self.method_combo))
        self.preset_combo = _combo(
            (("quality", "Quality"), ("balanced", "Balanced"),
             ("fast_quality", "Fast Quality"), ("performance", "Performance")),
            "Performance preset",
        )
        self.preset_row = card.add_row(SettingsRow("Preset", self.preset_combo))
        self.fsr_sharpening = Switch()
        self.fsr_sharpening.setAccessibleName("Enable FSR1-like sharpening")
        self.fsr_sharpening_row = card.add_row(SettingsRow("Sharpening", self.fsr_sharpening))
        self.fsr_sharpening_strength = QDoubleSpinBox()
        self.fsr_sharpening_strength.setRange(0.0, 1.0)
        self.fsr_sharpening_strength.setSingleStep(0.05)
        self.fsr_sharpening_strength.setDecimals(2)
        self.fsr_sharpening_strength_row = card.add_row(
            SettingsRow("Sharpen strength", self.fsr_sharpening_strength)
        )
        self.fsr_edge_strength = QDoubleSpinBox()
        self.fsr_edge_strength.setRange(0.0, 1.0)
        self.fsr_edge_strength.setSingleStep(0.05)
        self.fsr_edge_strength.setDecimals(2)
        self.fsr_edge_strength_row = card.add_row(SettingsRow("Edge strength", self.fsr_edge_strength))
        self.upscaling_control = ModeBinding(
            self.upscaling_enabled, self.upscaling_type_combo, off="off", parent=self
        )
        return card

    def _build_capture_card(self) -> SettingsCard:
        card = SettingsCard("Capture")
        self.window_combo = QComboBox()
        self.window_combo.setAccessibleName("Target window")
        self.window_combo.setMinimumWidth(225)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setAccessibleName("Refresh open windows")
        target = QWidget()
        target_layout = QHBoxLayout(target)
        target_layout.setContentsMargins(0, 0, 0, 0)
        target_layout.setSpacing(6)
        target_layout.addWidget(self.window_combo, 1)
        target_layout.addWidget(self.refresh_button)
        card.add_row(SettingsRow("Target window", target))
        self.capture_backend_combo = _combo(
            (("auto", "Auto"), ("wgc", "WGC"), ("dxcam", "DXCam"), ("mss", "MSS")),
            "Capture API",
        )
        runtime_mode = QLineEdit("Asynchronous")
        runtime_mode.setReadOnly(True)
        runtime_mode.setEnabled(False)
        card.add_row(SettingsRow("Capture API", self.capture_backend_combo))
        card.add_row(SettingsRow("Runtime mode", runtime_mode))
        self.queue_depth = QSpinBox()
        self.queue_depth.setRange(1, 16)
        self.queue_depth.setAccessibleName("Capture queue depth")
        card.add_row(SettingsRow("Queue depth", self.queue_depth))
        return card

    def _build_ai_card(self) -> SettingsCard:
        card = SettingsCard("AI Settings")
        self.model_combo = _combo(
            ((str(SRVGG_MODEL), "SRVGGNetCompact ×2"), ("custom", "Custom ONNX model")),
            "AI model",
        )
        self.browse_model_button = QPushButton("Browse…")
        self.browse_model_button.setAccessibleName("Browse for an ONNX model")
        model_controls = QWidget()
        model_layout = QHBoxLayout(model_controls)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(5)
        model_layout.addWidget(self.model_combo, 1)
        model_layout.addWidget(self.browse_model_button)
        self.model_row = card.add_row(SettingsRow("Model", model_controls))
        self.model_path_label = QLabel("")
        self.model_path_label.setObjectName("rowDescription")
        self.model_path_label.setWordWrap(True)
        card.add_widget(self.model_path_label)
        self.provider_combo = _combo(
            (("directml", "DirectML"), ("cpu", "CPU")), "AI provider"
        )
        self.provider_row = card.add_row(SettingsRow("Provider", self.provider_combo))
        self.device_id = QSpinBox()
        self.device_id.setRange(0, 64)
        self.device_id.setToolTip(
            "DirectML adapter used for AI upscaling; independent from the "
            "frame-generation device."
        )
        self.device_row = card.add_row(SettingsRow("Device", self.device_id))
        self.ai_scale_combo = _combo(
            (("auto", "Auto"), ("1", "×1"), ("2", "×2"), ("3", "×3"), ("4", "×4")),
            "AI model scale",
        )
        card.add_row(SettingsRow("Output scale", self.ai_scale_combo))
        self.ai_input_layout_combo = _combo((("nchw", "NCHW"), ("nhwc", "NHWC")), "AI input layout")
        self.ai_output_layout_combo = _combo((("nchw", "NCHW"), ("nhwc", "NHWC")), "AI output layout")
        self.ai_color_order_combo = _combo((("rgb", "RGB"), ("bgr", "BGR")), "AI color order")
        card.add_row(SettingsRow("Input layout", self.ai_input_layout_combo))
        card.add_row(SettingsRow("Output layout", self.ai_output_layout_combo))
        card.add_row(SettingsRow("Color order", self.ai_color_order_combo))
        self.ai_width = QSpinBox()
        self.ai_width.setRange(16, 4096)
        self.ai_height = QSpinBox()
        self.ai_height.setRange(16, 4096)
        self.ai_dims = QWidget()
        dimensions = QHBoxLayout(self.ai_dims)
        dimensions.setContentsMargins(0, 0, 0, 0)
        dimensions.setSpacing(5)
        dimensions.addWidget(self.ai_width)
        multiply = QLabel("×")
        multiply.setObjectName("mutedLabel")
        dimensions.addWidget(multiply)
        dimensions.addWidget(self.ai_height)
        self.ai_dims_row = card.add_row(SettingsRow("Internal size", self.ai_dims))
        self.tile_combo = _combo(
            (("auto", "Auto"), ("off", "Off"), ("128", "128 px"), ("256", "256 px"), ("512", "512 px")),
            "AI tiling",
        )
        self.tile_row = card.add_row(SettingsRow("Tiling", self.tile_combo))
        self.tile_overlap = QSpinBox()
        self.tile_overlap.setRange(0, 128)
        self.tile_overlap.setSuffix(" px")
        self.tile_overlap_row = card.add_row(SettingsRow("Tile overlap", self.tile_overlap))
        self.provider_fallback = Switch()
        self.provider_fallback.setAccessibleName("Allow CPU provider fallback")
        card.add_row(SettingsRow("CPU fallback", self.provider_fallback))
        self.ai_warning = QLabel("AI upscaling can require substantially more processing time at large internal sizes.")
        self.ai_warning.setObjectName("rowDescription")
        self.ai_warning.setWordWrap(True)
        card.add_widget(self.ai_warning)
        return card

    def _build_rendering_card(self) -> SettingsCard:
        card = SettingsCard("Rendering")
        self.pipeline_combo = _combo(
            (("cpu", "CPU-frame pipeline"), ("d3d11", "D3D11 GPU pipeline")),
            "Runtime pipeline",
        )
        card.add_row(SettingsRow("Pipeline", self.pipeline_combo))
        self.fps_combo = self._build_fps_combo(include_custom=True)
        self.custom_fps = QDoubleSpinBox()
        self.custom_fps.setRange(1.0, 1000.0)
        self.custom_fps.setDecimals(2)
        self.custom_fps.setSuffix(" FPS")
        fps_container = QWidget()
        fps_layout = QHBoxLayout(fps_container)
        fps_layout.setContentsMargins(0, 0, 0, 0)
        fps_layout.setSpacing(5)
        fps_layout.addWidget(self.fps_combo)
        fps_layout.addWidget(self.custom_fps)
        card.add_row(SettingsRow("Target FPS", fps_container))
        self.max_latency = QDoubleSpinBox()
        self.max_latency.setRange(1, 60000)
        self.max_latency.setSuffix(" ms")
        card.add_row(SettingsRow("Maximum latency", self.max_latency))
        self.overlay = Switch()
        self.overlay.setAccessibleName("Show output FPS counter")
        card.add_row(SettingsRow("FPS counter overlay", self.overlay))
        self.output_refinement_combo = _combo(
            ((0.0, "Off"), (0.12, "Subtle"), (0.22, "Crisp")), "Output refinement")
        self.output_refinement_combo.setToolTip(
            "CPU-frame preview only. Sharpens real and generated frames equally; cannot recover missing detail.")
        card.add_row(SettingsRow("Output refinement", self.output_refinement_combo))
        self.temporal_stabilization = Switch()
        self.temporal_stabilization.setAccessibleName(
            "Stabilize stationary and low-motion generated regions"
        )
        self.temporal_stabilization.setToolTip(
            "Suppresses interpolation shimmer in stationary and low-motion regions; "
            "clear motion remains model-generated."
        )
        card.add_row(SettingsRow("Temporal stabilization", self.temporal_stabilization))
        self.regional_transform = Switch()
        self.regional_transform.setAccessibleName(
            "Use experimental regional frame transformation"
        )
        self.regional_transform.setToolTip(
            "Routes tiles by motion and reliability, blends uncertain generated "
            "regions, and records provenance for research evaluation."
        )
        card.add_row(SettingsRow("Regional transformation", self.regional_transform))
        self.regional_budget = QDoubleSpinBox()
        self.regional_budget.setRange(1.0, 100.0)
        self.regional_budget.setDecimals(1)
        self.regional_budget.setSuffix(" ms")
        self.regional_budget_row = SettingsRow(
            "Regional deadline", self.regional_budget
        )
        card.add_row(self.regional_budget_row)
        self.regional_debug_overlay = Switch()
        self.regional_debug_overlay.setAccessibleName(
            "Show regional provenance colours"
        )
        self.regional_debug_overlay_row = SettingsRow(
            "Provenance view", self.regional_debug_overlay
        )
        card.add_row(self.regional_debug_overlay_row)

        self.pacing_combo = self._build_pacing_combo()
        card.add_row(SettingsRow("Frame pacing", self.pacing_combo))
        return card

    def _build_status_card(self) -> SettingsCard:
        card = SettingsCard("Status")
        self.state_status = CompactStatusRow("State", "Ready")
        self.fps_metric = CompactStatusRow("Presentation")
        self.latency_metric = CompactStatusRow("Latency")
        self.provider_metric = CompactStatusRow("Provider")
        self.dropped_metric = CompactStatusRow("Dropped frames")
        self.frame_count_metric = CompactStatusRow("Real / generated", "0 / 0")
        for row in (
            self.state_status,
            self.fps_metric,
            self.latency_metric,
            self.provider_metric,
            self.dropped_metric,
            self.frame_count_metric,
        ):
            card.add_widget(row)
        self.status_details_button = QPushButton("Show details")
        self.status_details_button.setObjectName("quietButton")
        self.status_details_button.setCheckable(True)
        self.status_details_button.setAccessibleName("Show detailed runtime telemetry")
        card.add_widget(self.status_details_button)
        self.status_details_panel = QWidget()
        detail_layout = QVBoxLayout(self.status_details_panel)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(0)
        self.capture_metric = CompactStatusRow("Capture")
        self.inference_metric = CompactStatusRow("AI inference")
        self.interpolation_metric = CompactStatusRow("RIFE interpolation")
        self.p95_metric = CompactStatusRow("P95 frame time")
        self.pacing_error_metric = CompactStatusRow("Pacing error")
        self.source_fps_metric = CompactStatusRow("Processed real FPS")
        self.dimensions_metric = CompactStatusRow("AI dimensions")
        self.tile_metric = CompactStatusRow("Tiling")
        self.recovery_metric = CompactStatusRow("Recoveries / fallbacks")
        for row in (
            self.capture_metric,
            self.inference_metric,
            self.interpolation_metric,
            self.p95_metric,
            self.pacing_error_metric,
            self.source_fps_metric,
            self.dimensions_metric,
            self.tile_metric,
            self.recovery_metric,
        ):
            detail_layout.addWidget(row)
        card.add_widget(self.status_details_panel)
        self.status_label = self.state_status.value_label
        self.status_label.setAccessibleName("Runtime status")
        self.error_label = QLabel("")
        self.error_label.setObjectName("errorLabel")
        self.error_label.setWordWrap(True)
        card.add_widget(self.error_label)
        buttons = QWidget()
        controls = QHBoxLayout(buttons)
        controls.setContentsMargins(0, 4, 0, 0)
        controls.setSpacing(6)
        self.view_logs_button = QPushButton("View logs")
        self.open_logs_button = QPushButton("Open log folder")
        controls.addWidget(self.view_logs_button)
        controls.addWidget(self.open_logs_button)
        controls.addStretch(1)
        card.add_widget(buttons)
        return card

    @staticmethod
    def _build_fps_combo(include_custom: bool = False) -> QComboBox:
        items = [("auto", "Auto"), ("30", "30 FPS"), ("60", "60 FPS"), ("120", "120 FPS"), ("144", "144 FPS")]
        if include_custom:
            items.append(("custom", "Custom"))
        return _combo(tuple(items), "Target FPS")

    @staticmethod
    def _build_pacing_combo() -> QComboBox:
        return _combo((("auto", "Auto"), ("fixed", "Fixed"), ("off", "Off")), "Frame pacing")

    # ---- secondary views ----------------------------------------------
    def _secondary_page(self, object_name: str, title: str, description: str) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        page.setObjectName(object_name)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(32, 26, 32, 28)
        layout.setSpacing(8)
        heading = QLabel(title)
        heading.setObjectName("pageTitle")
        detail = QLabel(description)
        detail.setObjectName("pageDescription")
        layout.addWidget(heading)
        layout.addWidget(detail)
        layout.addSpacing(10)
        return page, layout

    def _build_general_page(self) -> QWidget:
        page, layout = self._secondary_page("generalPage", "General", "GUI preferences only. Runtime behavior is unchanged.")
        card = SettingsCard("Appearance and defaults")
        self.general_theme_combo = _combo((("light", "Light"), ("dark", "Dark")), "Theme")
        card.add_row(SettingsRow("Theme", self.general_theme_combo))
        self.default_preset_combo = _combo(
            (("quality", "Quality"), ("balanced", "Balanced"),
             ("fast_quality", "Fast Quality"), ("performance", "Performance")),
            "Default preset",
        )
        card.add_row(SettingsRow("Default preset", self.default_preset_combo))
        self.default_provider_combo = _combo((("directml", "DirectML"), ("cpu", "CPU")), "Default provider")
        card.add_row(SettingsRow("Default provider", self.default_provider_combo))
        self.reset_settings_button = QPushButton("Reset GUI settings")
        card.add_widget(self.reset_settings_button)
        card.setMaximumWidth(650)
        layout.addWidget(card)
        layout.addStretch(1)
        return page

    def _build_logs_page(self) -> QWidget:
        page, layout = self._secondary_page("logsPage", "Logs", "Runtime messages from the current application session.")
        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("logView")
        self.log_view.setReadOnly(True)
        self.log_view.setPlainText("No runtime events yet.")
        layout.addWidget(self.log_view, 1)
        return page

    def _build_about_page(self) -> QWidget:
        page, layout = self._secondary_page("aboutPage", "About", "UniversalUpscaler v0.1")
        text = QLabel("A lightweight Windows utility for capture, upscaling, frame generation, and paced presentation.")
        text.setWordWrap(True)
        layout.addWidget(text)
        layout.addStretch(1)
        return page

    # ---- connections and persistence ---------------------------------
    def _connect(self) -> None:
        self.theme_button.clicked.connect(self.toggle_theme)
        self.start_button.clicked.connect(self._start_stop)
        self.refresh_button.clicked.connect(self.refresh_windows)
        self.window_combo.currentIndexChanged.connect(self._update_start_enabled)
        self.upscaling_control.value_changed.connect(self._mode_changed)
        self.frame_generation_control.value_changed.connect(self._frame_generation_changed)
        self.rife_model_combo.currentIndexChanged.connect(self._settings_changed)
        self.output_refinement_combo.currentIndexChanged.connect(self._settings_changed)
        self.temporal_stabilization.toggled.connect(
            lambda _checked: self._settings_changed()
        )
        self.regional_transform.toggled.connect(
            lambda _checked: self._settings_changed()
        )
        self.regional_transform.toggled.connect(
            lambda _checked: self._update_visibility()
        )
        self.regional_budget.valueChanged.connect(
            lambda _value: self._settings_changed()
        )
        self.regional_debug_overlay.toggled.connect(
            lambda _checked: self._settings_changed()
        )
        self.generated_frames_combo.currentIndexChanged.connect(
            lambda _index: self._settings_changed()
        )
        self.method_combo.currentIndexChanged.connect(self._method_changed)
        self.preset_combo.currentIndexChanged.connect(self._preset_changed)
        self.fps_combo.currentIndexChanged.connect(lambda _index: self._fps_changed(self.fps_combo))
        self.rife_fps_combo.currentIndexChanged.connect(lambda _index: self._fps_changed(self.rife_fps_combo))
        self.custom_fps.valueChanged.connect(lambda _value: self._settings_changed())
        self.pacing_combo.currentIndexChanged.connect(lambda _index: self._pacing_changed(self.pacing_combo))
        self.rife_pacing_combo.currentIndexChanged.connect(lambda _index: self._pacing_changed(self.rife_pacing_combo))
        self.provider_combo.currentIndexChanged.connect(lambda _index: self._provider_changed(self.provider_combo))
        self.rife_provider_combo.currentIndexChanged.connect(lambda _index: self._provider_changed(self.rife_provider_combo))
        self.device_id.valueChanged.connect(self._device_changed)
        self.rife_device_id.valueChanged.connect(self._device_changed)
        self.model_combo.currentIndexChanged.connect(self._model_changed)
        self.browse_model_button.clicked.connect(self._browse_model)
        for widget in (
            self.tile_combo,
            self.ai_scale_combo,
            self.ai_input_layout_combo,
            self.ai_output_layout_combo,
            self.ai_color_order_combo,
            self.capture_backend_combo,
            self.pipeline_combo,
        ):
            widget.currentIndexChanged.connect(lambda _index: self._settings_changed())
        for widget in (self.tile_overlap, self.ai_width, self.ai_height, self.queue_depth):
            widget.valueChanged.connect(lambda _value: self._settings_changed())
        self.max_latency.valueChanged.connect(lambda _value: self._settings_changed())
        self.provider_fallback.toggled.connect(lambda _checked: self._settings_changed())
        self.overlay.toggled.connect(lambda _checked: self._settings_changed())
        self.fsr_sharpening.toggled.connect(lambda _checked: self._settings_changed())
        self.fsr_sharpening_strength.valueChanged.connect(lambda _value: self._settings_changed())
        self.fsr_edge_strength.valueChanged.connect(lambda _value: self._settings_changed())
        self.status_details_button.toggled.connect(self._status_details_toggled)

        self.add_profile_button.clicked.connect(lambda: self._add_profile())
        self.rename_profile_button.clicked.connect(lambda: self._rename_profile())
        self.delete_profile_button.clicked.connect(self._delete_profile)
        self.general_nav.clicked.connect(lambda: self._show_view(self.general_page))
        self.logs_nav.clicked.connect(self._show_logs)
        self.about_nav.clicked.connect(lambda: self._show_view(self.about_page))
        self.general_theme_combo.currentIndexChanged.connect(self._general_theme_changed)
        self.default_preset_combo.currentIndexChanged.connect(self._general_preset_changed)
        self.default_provider_combo.currentIndexChanged.connect(self._general_provider_changed)
        self.reset_settings_button.clicked.connect(self._reset_gui_settings)

        self.controller.state_changed.connect(self._state_changed)
        self.controller.telemetry_changed.connect(self._telemetry_changed)
        self.controller.error_occurred.connect(self._show_error)
        self.controller.log_message.connect(self._append_log)
        self.controller.running_changed.connect(self._running_changed)
        self.view_logs_button.clicked.connect(self._show_logs)
        self.open_logs_button.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(user_data_dir("logs"))))
        )
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self._start_stop)
        self.stop_shortcut = QShortcut(QKeySequence("Escape"), self)
        self.stop_shortcut.setContext(Qt.ApplicationShortcut)
        self.stop_shortcut.activated.connect(self._stop_if_running)

    def _restore_settings(self, *, restore_profiles: bool = True) -> None:
        if restore_profiles:
            self._apply_profile_payload(self.settings.selected_profile)
        self.upscaling_control.set_value(self.settings.upscaling_mode)
        self.frame_generation_control.set_value(self.settings.frame_generation)
        self._set_combo(self.rife_model_combo, self.settings.rife_model_path or str(RIFE_MODEL))
        self._set_combo(self.generated_frames_combo, self.settings.generated_frames)
        self._set_combo(self.preset_combo, self.settings.performance_preset)
        self._set_combo(self.default_preset_combo, self.settings.performance_preset)
        self._set_combo(self.fps_combo, self.settings.target_fps)
        self._set_combo(self.rife_fps_combo, self.settings.target_fps if self.settings.target_fps != "custom" else "auto")
        self.custom_fps.setValue(self.settings.custom_fps)
        self._set_combo(self.provider_combo, self.settings.provider)
        self._set_combo(self.rife_provider_combo, self.settings.provider)
        self._set_combo(self.default_provider_combo, self.settings.provider)
        self.device_id.setValue(self.settings.device_id)
        self.rife_device_id.setValue(
            self.settings.device_id
            if self.settings.rife_device_id is None
            else self.settings.rife_device_id
        )
        self._restore_model_selection()
        self._set_combo(self.ai_scale_combo, self.settings.ai_scale)
        self._set_combo(self.ai_input_layout_combo, self.settings.ai_input_layout)
        self._set_combo(self.ai_output_layout_combo, self.settings.ai_output_layout)
        self._set_combo(self.ai_color_order_combo, self.settings.ai_color_order)
        self._set_combo(self.capture_backend_combo, self.settings.capture_backend)
        self._set_combo(self.pipeline_combo, self.settings.pipeline)
        self._set_combo(self.tile_combo, self.settings.ai_tile)
        self.tile_overlap.setValue(self.settings.ai_tile_overlap)
        self.ai_width.setValue(self.settings.ai_input_width)
        self.ai_height.setValue(self.settings.ai_input_height)
        self._set_combo(self.pacing_combo, self.settings.frame_pacing)
        self._set_combo(self.rife_pacing_combo, self.settings.frame_pacing)
        self.max_latency.setValue(self.settings.max_frame_latency_ms)
        self.queue_depth.setValue(self.settings.queue_depth)
        self.provider_fallback.setChecked(self.settings.allow_provider_fallback)
        self.overlay.setChecked(self.settings.show_performance_overlay)
        self._set_combo(self.output_refinement_combo, self.settings.output_refinement)
        self.temporal_stabilization.setChecked(self.settings.temporal_stabilization)
        self.regional_transform.setChecked(self.settings.regional_transform)
        self.regional_budget.setValue(self.settings.regional_budget_ms)
        self.regional_debug_overlay.setChecked(
            self.settings.regional_debug_overlay
        )
        self.fsr_sharpening.setChecked(self.settings.fsr1_like_sharpening)
        self.fsr_sharpening_strength.setValue(self.settings.fsr1_like_sharpening_strength)
        self.fsr_edge_strength.setValue(self.settings.fsr1_like_edge_strength)
        self.status_details_button.setChecked(self.settings.status_details_visible)
        self.status_details_panel.setVisible(self.settings.status_details_visible)
        self.status_details_button.setText(
            "Hide details" if self.settings.status_details_visible else "Show details"
        )
        self._rebuild_methods(self.settings.upscaling_method)
        self._set_combo(self.general_theme_combo, self.settings.theme)
        self.theme_button.setText("Dark" if self.settings.theme == "light" else "Light")
        if restore_profiles:
            self._rebuild_profile_buttons()
            self._select_profile(self.settings.selected_profile, save=False, load=False)

        # Former disclosure attributes are retained without reintroducing the old layout.
        if not hasattr(self, "advanced_toggle"):
            self.advanced_toggle = QPushButton()
            self.advanced_toggle.setCheckable(True)
            self.advanced_toggle.hide()
        self.advanced_toggle.setChecked(self.settings.advanced_visible)
        self.advanced_panel = self.ai_card

    def _apply_profile_payload(self, name: str) -> None:
        payload = self.settings.profile_settings.get(name, {})
        allowed = {field.name for field in fields(GuiSettings)} - {
            "theme",
            "profiles",
            "selected_profile",
            "profile_settings",
            "status_details_visible",
        }
        for key, value in payload.items():
            if key in allowed:
                setattr(self.settings, key, value)

    @staticmethod
    def _set_combo(combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _restore_model_selection(self) -> None:
        custom_path = self.settings.ai_model_path.strip()
        self._custom_model_path = custom_path
        self.model_combo.setCurrentIndex(1 if custom_path else 0)
        if custom_path:
            self.model_combo.setItemText(1, f"Custom: {Path(custom_path).name}")
            self.model_path_label.setText(custom_path)
        else:
            self.model_combo.setItemText(1, "Custom ONNX model")
            self.model_path_label.setText(str(SRVGG_MODEL))
        self.browse_model_button.setVisible(bool(custom_path) or self.model_combo.currentData() == "custom")

    def _save_settings(self) -> None:
        if self._loading:
            return
        self.settings.upscaling_mode = self.upscaling_control.value()
        self.settings.upscaling_method = str(self.method_combo.currentData() or "bilinear")
        self.settings.frame_generation = self.frame_generation_control.value()
        self.settings.rife_model_path = str(self.rife_model_combo.currentData() or RIFE_MODEL)
        self.settings.generated_frames = int(self.generated_frames_combo.currentData() or 1)
        self.settings.performance_preset = str(self.preset_combo.currentData() or "balanced")
        self.settings.target_fps = str(self.fps_combo.currentData() or "auto")
        self.settings.custom_fps = self.custom_fps.value()
        self.settings.provider = str(self.provider_combo.currentData() or "directml")
        self.settings.device_id = self.device_id.value()
        self.settings.rife_device_id = self.rife_device_id.value()
        self.settings.ai_tile = str(self.tile_combo.currentData() or "auto")
        self.settings.ai_tile_overlap = self.tile_overlap.value()
        self.settings.ai_input_width = self.ai_width.value()
        self.settings.ai_input_height = self.ai_height.value()
        self.settings.frame_pacing = str(self.pacing_combo.currentData() or "auto")
        self.settings.max_frame_latency_ms = self.max_latency.value()
        self.settings.queue_depth = self.queue_depth.value()
        self.settings.allow_provider_fallback = self.provider_fallback.isChecked()
        self.settings.show_performance_overlay = self.overlay.isChecked()
        self.settings.output_refinement = float(self.output_refinement_combo.currentData() or 0.0)
        self.settings.temporal_stabilization = self.temporal_stabilization.isChecked()
        self.settings.regional_transform = self.regional_transform.isChecked()
        self.settings.regional_budget_ms = self.regional_budget.value()
        self.settings.regional_debug_overlay = (
            self.regional_debug_overlay.isChecked()
        )
        self.settings.ai_model_path = (
            self._custom_model_path if self.model_combo.currentData() == "custom" else ""
        )
        self.settings.ai_scale = str(self.ai_scale_combo.currentData() or "auto")
        self.settings.ai_input_layout = str(self.ai_input_layout_combo.currentData() or "nchw")
        self.settings.ai_output_layout = str(self.ai_output_layout_combo.currentData() or "nchw")
        self.settings.ai_color_order = str(self.ai_color_order_combo.currentData() or "rgb")
        self.settings.capture_backend = str(self.capture_backend_combo.currentData() or "wgc")
        self.settings.pipeline = str(self.pipeline_combo.currentData() or "cpu")
        self.settings.fsr1_like_sharpening = self.fsr_sharpening.isChecked()
        self.settings.fsr1_like_sharpening_strength = self.fsr_sharpening_strength.value()
        self.settings.fsr1_like_edge_strength = self.fsr_edge_strength.value()
        self.settings.status_details_visible = self.status_details_button.isChecked()
        profile_payload = asdict(self.settings)
        for key in ("theme", "profiles", "selected_profile", "profile_settings", "status_details_visible"):
            profile_payload.pop(key, None)
        self.settings.profile_settings[self.settings.selected_profile] = profile_payload
        try:
            self.settings_store.save(self.settings)
        except OSError as error:
            self._show_error(f"Unable to save settings: {error}")

    def _settings_changed(self) -> None:
        if self._loading:
            return
        self._update_visibility()
        self._update_start_enabled()
        self._save_settings()

    # ---- profiles and views -------------------------------------------
    def _rebuild_profile_buttons(self) -> None:
        if not hasattr(self, "profile_list_layout"):
            return
        while self.profile_list_layout.count():
            item = self.profile_list_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._profile_buttons = []
        for name in self.settings.profiles:
            button = SidebarItem(name)
            button.setProperty("profileName", name)
            button.clicked.connect(lambda _checked=False, selected=name: self._select_profile(selected))
            self.profile_list_layout.addWidget(button)
            self._profile_buttons.append(button)

    def _select_profile(self, name: str, *, save: bool = True, load: bool = True) -> None:
        if name not in self.settings.profiles:
            return
        previous = self.settings.selected_profile
        if save and name != previous and previous in self.settings.profiles:
            self._save_settings()
        self.settings.selected_profile = name
        if load and name != previous:
            self._apply_profile_payload(name)
            was_loading = self._loading
            self._loading = True
            self._restore_settings(restore_profiles=False)
            self._loading = was_loading
            self._update_visibility()
        self.profile_heading_label.setText(f'Profile: "{name}"')
        for button in self._profile_buttons:
            button.setChecked(button.property("profileName") == name)
        self.rename_profile_button.setEnabled(name != "Default")
        self.delete_profile_button.setEnabled(name != "Default")
        self._show_view(self.workspace_page)
        if save:
            self._save_settings()

    def _add_profile(self, name: str | None = None) -> None:
        if name is None:
            name, accepted = QInputDialog.getText(self, "New profile", "Profile name:")
            if not accepted:
                return
        name = name.strip()
        if not name or name in self.settings.profiles:
            return
        self._save_settings()
        source_profile = self.settings.selected_profile
        self.settings.profiles.append(name)
        self.settings.profile_settings[name] = dict(
            self.settings.profile_settings.get(source_profile, {})
        )
        self._rebuild_profile_buttons()
        self._select_profile(name)

    def _rename_profile(self, name: str | None = None) -> None:
        current = self.settings.selected_profile
        if current == "Default":
            return
        if name is None:
            name, accepted = QInputDialog.getText(self, "Rename profile", "Profile name:", text=current)
            if not accepted:
                return
        name = name.strip()
        if not name or name in self.settings.profiles:
            return
        index = self.settings.profiles.index(current)
        self.settings.profiles[index] = name
        self.settings.profile_settings[name] = self.settings.profile_settings.pop(current, {})
        self.settings.selected_profile = name
        self._rebuild_profile_buttons()
        self._select_profile(name)

    def _delete_profile(self) -> None:
        current = self.settings.selected_profile
        if current == "Default":
            return
        self.settings.profiles.remove(current)
        self.settings.profile_settings.pop(current, None)
        self._rebuild_profile_buttons()
        self._select_profile("Default")

    def _show_view(self, page: QWidget) -> None:
        self.view_stack.setCurrentWidget(page)
        if page is not self.workspace_page:
            for button in self._profile_buttons:
                button.setChecked(False)

    def _show_logs(self) -> None:
        self.log_view.setPlainText("\n".join(self._logs) or "No runtime events yet.")
        self._show_view(self.logs_page)
        self.logs_nav.setChecked(True)

    def _general_theme_changed(self) -> None:
        if self._loading:
            return
        theme = str(self.general_theme_combo.currentData())
        if theme != self.settings.theme:
            self.settings.theme = theme
            apply_theme(QApplication.instance(), theme)
            self.theme_button.setText("Dark" if theme == "light" else "Light")
            self._save_settings()

    def _general_preset_changed(self) -> None:
        if self._loading:
            return
        self._set_combo(self.preset_combo, str(self.default_preset_combo.currentData()))

    def _general_provider_changed(self) -> None:
        if self._loading:
            return
        self._set_combo(self.provider_combo, str(self.default_provider_combo.currentData()))

    def _reset_gui_settings(self) -> None:
        self._loading = True
        self.settings = GuiSettings()
        self._restore_settings()
        self._loading = False
        self._update_visibility()
        apply_theme(QApplication.instance(), "light")
        self._save_settings()

    # ---- settings behavior --------------------------------------------
    def toggle_theme(self) -> None:
        self.settings.theme = "light" if self.settings.theme == "dark" else "dark"
        apply_theme(QApplication.instance(), self.settings.theme)
        self.theme_button.setText("Dark" if self.settings.theme == "light" else "Light")
        self._set_combo(self.general_theme_combo, self.settings.theme)
        self._save_settings()

    def _mode_changed(self, _value: str) -> None:
        self._rebuild_methods(self.settings.upscaling_method)
        self._update_visibility()
        self._settings_changed()

    def _frame_generation_changed(self, _value: str) -> None:
        if not self._loading:
            self._apply_preset_latency()
        self._update_visibility()
        self._settings_changed()

    def _method_changed(self, _index: int) -> None:
        self._update_visibility()
        self._settings_changed()

    def _device_changed(self, _value: int) -> None:
        if self._loading:
            return
        self._settings_changed()

    def _model_changed(self, _index: int) -> None:
        custom = self.model_combo.currentData() == "custom"
        self.browse_model_button.setVisible(custom)
        if custom:
            self.model_path_label.setText(self._custom_model_path or "No custom ONNX model selected.")
        else:
            self.model_path_label.setText(str(SRVGG_MODEL))
        self._settings_changed()

    def _browse_model(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Select ONNX upscaling model",
            self._custom_model_path or str(PROJECT_ROOT / "models"),
            "ONNX models (*.onnx)",
        )
        if not path:
            return
        self._custom_model_path = str(Path(path).resolve())
        self.model_combo.setItemText(1, f"Custom: {Path(path).name}")
        self.model_combo.setCurrentIndex(1)
        self.model_path_label.setText(self._custom_model_path)
        self._settings_changed()

    def _status_details_toggled(self, checked: bool) -> None:
        self.status_details_panel.setVisible(checked)
        self.status_details_button.setText("Hide details" if checked else "Show details")
        self._settings_changed()

    def _rebuild_methods(self, preferred: str | None = None) -> None:
        mode = self.upscaling_control.value() or self.settings.upscaling_mode
        self.method_combo.blockSignals(True)
        self.method_combo.clear()
        if mode == "ai":
            items = (("srvgg", "SRVGGNetCompact ×2"),)
        elif mode == "shader":
            items = (
                ("nearest", "Nearest"),
                ("bilinear", "Bilinear"),
                ("bicubic", "Bicubic"),
                ("lanczos", "Lanczos2"),
                ("fsr1_like", "FSR1-like"),
            )
        else:
            items = (("bilinear", "Bilinear"),)
        for value, label in items:
            self.method_combo.addItem(label, value)
        index = self.method_combo.findData(preferred)
        self.method_combo.setCurrentIndex(index if index >= 0 else 0)
        self.method_combo.blockSignals(False)

    def _preset_changed(self) -> None:
        if self._loading:
            return
        value = str(self.preset_combo.currentData())
        preset = PRESETS[value]
        self.ai_width.setValue(preset["width"])
        self.ai_height.setValue(preset["height"])
        self._set_combo(self.tile_combo, preset["tile"])
        self._apply_preset_latency()
        self.queue_depth.setValue(preset["queue"])
        self._set_combo(self.default_preset_combo, value)
        self._settings_changed()

    def _apply_preset_latency(self) -> None:
        value = str(self.preset_combo.currentData() or "balanced")
        preset = PRESETS[value]
        latency_key = (
            "rife_latency"
            if self.frame_generation_control.value() == "rife"
            else "latency"
        )
        self.max_latency.setValue(float(preset[latency_key]))

    def _fps_changed(self, source: QComboBox) -> None:
        if self._loading:
            return
        value = str(source.currentData())
        target = self.fps_combo if source is self.rife_fps_combo else self.rife_fps_combo
        if value == "custom" and target is self.rife_fps_combo:
            value = "auto"
        target.blockSignals(True)
        self._set_combo(target, value)
        target.blockSignals(False)
        self.custom_fps.setVisible(self.fps_combo.currentData() == "custom")
        self._settings_changed()

    def _pacing_changed(self, source: QComboBox) -> None:
        if self._loading:
            return
        target = self.pacing_combo if source is self.rife_pacing_combo else self.rife_pacing_combo
        target.blockSignals(True)
        self._set_combo(target, str(source.currentData()))
        target.blockSignals(False)
        self._settings_changed()

    def _provider_changed(self, source: QComboBox) -> None:
        if self._loading:
            return
        target = self.provider_combo if source is self.rife_provider_combo else self.rife_provider_combo
        value = str(source.currentData())
        target.blockSignals(True)
        self._set_combo(target, value)
        target.blockSignals(False)
        self.default_provider_combo.blockSignals(True)
        self._set_combo(self.default_provider_combo, value)
        self.default_provider_combo.blockSignals(False)
        self._update_visibility()
        self._settings_changed()

    def _update_visibility(self) -> None:
        ai_enabled = self.upscaling_control.value() == "ai"
        upscaling_enabled = self.upscaling_control.value() != "off"
        rife_enabled = self.frame_generation_control.value() == "rife"
        self.upscaling_type_row.setVisible(upscaling_enabled)
        self.method_row.setVisible(upscaling_enabled)
        self.preset_row.setVisible(upscaling_enabled)
        self.ai_card.setVisible(ai_enabled)
        self.ai_warning.setVisible(ai_enabled)
        self.rife_type_row.setVisible(rife_enabled)
        self.rife_model_row.setVisible(rife_enabled)
        self.rife_provider_row.setVisible(rife_enabled)
        self.rife_device_row.setVisible(rife_enabled)
        self.generated_frames_row.setVisible(rife_enabled)
        self.regional_transform.setVisible(rife_enabled)
        self.regional_budget_row.setVisible(
            rife_enabled and self.regional_transform.isChecked()
        )
        self.regional_debug_overlay_row.setVisible(
            rife_enabled and self.regional_transform.isChecked()
        )
        self.device_id.setEnabled(self.provider_combo.currentData() == "directml")
        self.rife_device_id.setEnabled(self.rife_provider_combo.currentData() == "directml")
        fsr_enabled = (
            self.upscaling_control.value() == "shader"
            and self.method_combo.currentData() == "fsr1_like"
        )
        self.fsr_sharpening_row.setVisible(fsr_enabled)
        self.fsr_sharpening_strength_row.setVisible(fsr_enabled and self.fsr_sharpening.isChecked())
        self.fsr_edge_strength_row.setVisible(fsr_enabled)
        self.browse_model_button.setVisible(
            ai_enabled and self.model_combo.currentData() == "custom"
        )
        self.custom_fps.setVisible(self.fps_combo.currentData() == "custom")
        self._update_compatibility()
        self._apply_responsive_layout()

    def _update_compatibility(self) -> None:
        message = ""
        try:
            if self.window_combo.currentData() is not None:
                self._configuration().validate()
        except (TypeError, ValueError) as error:
            message = str(error)
        self.compatibility_label.setText(message)
        self.compatibility_label.setVisible(bool(message))

    # ---- window enumeration and runtime -------------------------------
    def refresh_windows(self) -> None:
        previous = self.window_combo.currentData()
        self.window_combo.blockSignals(True)
        self.window_combo.clear()
        try:
            windows = self.window_provider()
        except Exception as error:
            windows = []
            self._show_error(f"Unable to enumerate windows: {error}")
        selected = -1
        for window in windows:
            dimensions = ""
            try:
                region = get_client_capture_region(window.hwnd)
                dimensions = f"  ·  {region.width}×{region.height}"
            except WindowCaptureError:
                pass
            self.window_combo.addItem(f"{window.title}{dimensions}", window.hwnd)
            if window.hwnd == previous:
                selected = self.window_combo.count() - 1
        if selected >= 0:
            self.window_combo.setCurrentIndex(selected)
        elif previous is not None:
            self.window_combo.setCurrentIndex(-1)
            self.status_label.setText("Target window no longer available")
        elif self.window_combo.count():
            self.window_combo.setCurrentIndex(0)
        self.window_combo.blockSignals(False)
        self._update_start_enabled()

    def _configuration(self) -> RuntimeConfiguration:
        fps_value = self.fps_combo.currentData()
        if fps_value == "auto":
            target = None
        elif fps_value == "custom":
            target = self.custom_fps.value()
        else:
            target = float(fps_value)
        return RuntimeConfiguration(
            hwnd=int(self.window_combo.currentData() or 0),
            upscaling_mode=self.upscaling_control.value(),
            upscaling_method=str(self.method_combo.currentData() or "bilinear"),
            frame_generation=self.frame_generation_control.value(),
            generated_frames=int(self.generated_frames_combo.currentData() or 1),
            rife_model_path=Path(self.rife_model_combo.currentData() or RIFE_MODEL),
            warmup_seconds=5.0,
            preset=str(self.preset_combo.currentData()),
            target_fps=target,
            provider=str(self.provider_combo.currentData()),
            device_id=self.device_id.value(),
            rife_device_id=self.rife_device_id.value(),
            ai_tile=str(self.tile_combo.currentData()),
            ai_tile_overlap=self.tile_overlap.value(),
            ai_input_width=self.ai_width.value(),
            ai_input_height=self.ai_height.value(),
            frame_pacing=str(self.pacing_combo.currentData()),
            max_frame_latency_ms=self.max_latency.value(),
            queue_depth=self.queue_depth.value(),
            allow_provider_fallback=self.provider_fallback.isChecked(),
            show_performance_overlay=self.overlay.isChecked(),
            output_refinement=(float(self.output_refinement_combo.currentData() or 0.0)
                               if self.pipeline_combo.currentData() == "cpu" else 0.0),
            temporal_stabilization=self.temporal_stabilization.isChecked(),
            regional_transform=self.regional_transform.isChecked(),
            regional_budget_ms=self.regional_budget.value(),
            regional_debug_overlay=self.regional_debug_overlay.isChecked(),
            model_path=(
                Path(self._custom_model_path)
                if self.model_combo.currentData() == "custom"
                else SRVGG_MODEL
            ),
            ai_scale=str(self.ai_scale_combo.currentData()),
            ai_input_layout=str(self.ai_input_layout_combo.currentData()),
            ai_output_layout=str(self.ai_output_layout_combo.currentData()),
            ai_color_order=str(self.ai_color_order_combo.currentData()),
            capture_backend=str(self.capture_backend_combo.currentData()),
            pipeline=str(self.pipeline_combo.currentData()),
            fsr1_like_sharpening=self.fsr_sharpening.isChecked(),
            fsr1_like_sharpening_strength=self.fsr_sharpening_strength.value(),
            fsr1_like_edge_strength=self.fsr_edge_strength.value(),
        )

    def _update_start_enabled(self) -> None:
        valid = self.window_combo.currentData() is not None
        if valid:
            try:
                self._configuration().validate()
            except ValueError:
                valid = False
        self.start_button.setEnabled(self.controller.running or valid)
        self._update_compatibility()

    def _start_stop(self) -> None:
        self.error_label.clear()
        if self.controller.running:
            self.controller.stop()
            return
        try:
            self.controller.start(self._configuration())
        except ValueError as error:
            self._show_error(str(error))

    def _stop_if_running(self) -> None:
        if self.controller.running:
            self.controller.stop()

    def _running_changed(self, running: bool) -> None:
        if running:
            self.frame_count_metric.set_value("0 / 0")
        for widget in (
            self.frame_card,
            self.upscaling_card,
            self.capture_card,
            self.rendering_card,
            self.ai_card,
            self.add_profile_button,
            self.rename_profile_button,
            self.delete_profile_button,
            self.general_page,
        ):
            widget.setEnabled(not running)
        self.start_button.setText("Stop · Esc" if running else "Start")
        self.start_button.setProperty("danger", running)
        self.start_button.style().unpolish(self.start_button)
        self.start_button.style().polish(self.start_button)
        self.start_button.setEnabled(True if running else self.window_combo.currentData() is not None)

    def _state_changed(self, state: str) -> None:
        if state.startswith("warming_up_"):
            seconds = state.rsplit("_", 1)[-1]
            self.status_label.setText(f"Warming up · {seconds}s")
            self.start_button.setText(seconds)
        else:
            self.status_label.setText(state.replace("_", " ").title())
            if self.controller.running:
                self.start_button.setText("Stop · Esc")

    def _telemetry_changed(self, data: dict) -> None:
        presentation = data.get("presentation_fps", data.get("fps", 0.0))
        latency = data.get("total_frame_ms", 0.0)
        provider = str(data.get("active_provider", "—")).replace("ExecutionProvider", "")
        dropped = data.get("dropped_frames", 0)
        self.fps_metric.set_value(f"{presentation:.1f} FPS")
        self.latency_metric.set_value(f"{latency:.1f} ms")
        self.provider_metric.set_value(provider)
        self.dropped_metric.set_value(str(dropped))
        real_frames = int(data.get("presented_real_frames", 0))
        generated_frames = int(data.get("presented_generated_frames", 0))
        self.frame_count_metric.set_value(f"{real_frames} / {generated_frames}")
        self.capture_metric.set_value(f"{float(data.get('capture_ms', 0.0)):.1f} ms")
        self.inference_metric.set_value(f"{float(data.get('inference_ms', 0.0)):.1f} ms")
        self.interpolation_metric.set_value(f"{float(data.get('interpolation_ms', 0.0)):.1f} ms")
        self.p95_metric.set_value(f"{float(data.get('p95_frame_ms', 0.0)):.1f} ms")
        self.pacing_error_metric.set_value(f"{float(data.get('pacing_error_ms', 0.0)):.1f} ms")
        self.source_fps_metric.set_value(f"{float(data.get('estimated_source_fps', 0.0)):.1f} FPS")
        input_dimensions = data.get("ai_input_dimensions")
        output_dimensions = data.get("ai_output_dimensions")
        if input_dimensions and output_dimensions:
            dimensions = (
                f"{input_dimensions[0]}×{input_dimensions[1]} → "
                f"{output_dimensions[0]}×{output_dimensions[1]}"
            )
        else:
            dimensions = "—"
        self.dimensions_metric.set_value(dimensions)
        self.tile_metric.set_value(str(data.get("tile_mode", "—")))
        recoveries = int(data.get("successful_recoveries", 0))
        fallbacks = int(data.get("fallback_activations", 0))
        self.recovery_metric.set_value(f"{recoveries} / {fallbacks}")

    def _show_error(self, message: str) -> None:
        self.error_label.setText(message)
        self.status_label.setText("Failed")

    def _append_log(self, message: str) -> None:
        self._logs.append(message)
        self._logs = self._logs[-200:]
        if hasattr(self, "log_view"):
            self.log_view.setPlainText("\n".join(self._logs))

    # ---- responsive layout and shutdown -------------------------------
    def _apply_responsive_layout(self, width: int | None = None) -> None:
        if not hasattr(self, "workspace_grid"):
            return
        width = width if width is not None else self.settings_scroll.viewport().width()
        columns = 2 if width >= 760 else 1
        cards = [card for card in self._workspace_cards if card is not self.ai_card or self.upscaling_control.value() == "ai"]
        signature = tuple(id(card) for card in cards)
        if (
            self.workspace_grid.count()
            and columns == self.workspace_column_count
            and signature == self._workspace_layout_signature
        ):
            return
        while self.workspace_grid.count():
            self.workspace_grid.takeAt(0)
        self.workspace_column_count = columns
        self._workspace_layout_signature = signature
        for index, card in enumerate(cards):
            row, column = divmod(index, columns)
            self.workspace_grid.addWidget(card, row, column)
        for column in range(columns):
            self.workspace_grid.setColumnStretch(column, 1)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        QTimer.singleShot(0, self._apply_responsive_layout)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._save_settings()
        if self.controller.running:
            self.status_label.setText("Stopping")
            if not self.controller.shutdown():
                self._show_error("The runtime did not stop within the shutdown timeout.")
                event.ignore()
                return
        event.accept()
