from pathlib import Path
 
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QComboBox,
    QGroupBox, QPushButton, QSpinBox, QDoubleSpinBox,
    QLineEdit, QProgressBar,
)
from PyQt6.QtCore import Qt
 
import config as config
from ui.widgets.slider_helper import make_slider
 
 
class PanelsMixin:
 
    def _build_ui(self):
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)
 
        root_layout.addWidget(self._build_left_panel(),   stretch=1)
        root_layout.addWidget(self._build_center_panel(), stretch=3)
        root_layout.addWidget(self._build_right_panel(),  stretch=1)
 
        self.setCentralWidget(root)
 
    # ── Left panel ────────────────────────────────────────────────────────────
    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(6)
 
        layout.addWidget(self._build_selector_group())
        layout.addWidget(self._build_position_group())
        layout.addWidget(self._build_rotation_group())
        layout.addWidget(self._build_lighting_group())
        layout.addStretch()
        return panel
 
    # ── Selector group ────────────────────────────────────────────────────────
    def _build_selector_group(self) -> QGroupBox:
        box = QGroupBox("Selectors")
        layout = QVBoxLayout(box)
        layout.setSpacing(4)
 
        # 1. 3-D Model
        layout.addWidget(QLabel("3D Model"))
        self.combo_3d_model = QComboBox()
        for label, _ in config.AVAILABLE_3D_MODELS:
            self.combo_3d_model.addItem(label)
        self.combo_3d_model.currentIndexChanged.connect(self._on_3d_model_changed)
        layout.addWidget(self.combo_3d_model)
 
        # 2. Background Image — dropdown of presets; 3D model always stays on top
        layout.addWidget(QLabel("Background Image"))
        self.combo_image = QComboBox()
        for label, _ in config.AVAILABLE_IMAGES:
            self.combo_image.addItem(label)
        self.combo_image.currentIndexChanged.connect(self._on_image_changed)
        layout.addWidget(self.combo_image)
 
        # 3. Vision Model
        layout.addWidget(QLabel("Vision Model"))
        self.combo_vision_model = QComboBox()
        for label, _ in config.AVAILABLE_VISION_MODELS:
            self.combo_vision_model.addItem(label)
        self.combo_vision_model.currentIndexChanged.connect(self._on_vision_model_changed)
        layout.addWidget(self.combo_vision_model)
 
        return box
 
    # ── Position group ────────────────────────────────────────────────────────
    def _build_position_group(self) -> QGroupBox:
        box = QGroupBox("Position")
        layout = QVBoxLayout(box)
 
        for attr, axis, lo, hi, init in [
            ("slider_px", "X", -5.0, 5.0, 0.0),
            ("slider_py", "Y", -5.0, 5.0, 0.0),
            ("slider_pz", "Z", -5.0, 5.0, 0.0),
        ]:
            slider, val_lbl = make_slider(lo, hi, init)
            slider.valueChanged.connect(
                lambda s, a=axis, sl=slider: self._on_pos(a, sl._from_s(s))
            )
            setattr(self, attr, slider)
            row = QHBoxLayout()
            row.addWidget(QLabel(axis))
            row.addWidget(slider)
            row.addWidget(val_lbl)
            layout.addLayout(row)
 
        return box
 
    # ── Rotation group ────────────────────────────────────────────────────────
    def _build_rotation_group(self) -> QGroupBox:
        box = QGroupBox("Rotation")
        layout = QVBoxLayout(box)
 
        for attr, axis, lo, hi, init in [
            ("slider_rx", "X", -180.0, 180.0, 0.0),
            ("slider_ry", "Y", -180.0, 180.0, 0.0),
            ("slider_rz", "Z", -180.0, 180.0, 0.0),
        ]:
            slider, val_lbl = make_slider(lo, hi, init)
            slider.valueChanged.connect(
                lambda s, a=axis, sl=slider: self._on_rot(a, sl._from_s(s))
            )
            setattr(self, attr, slider)
            row = QHBoxLayout()
            row.addWidget(QLabel(axis))
            row.addWidget(slider)
            row.addWidget(val_lbl)
            layout.addLayout(row)
 
        return box
 
    # ── Lighting group ────────────────────────────────────────────────────────
    def _build_lighting_group(self) -> QGroupBox:
        box = QGroupBox("Lighting")
        layout = QVBoxLayout(box)
 
        for attr, axis, lo, hi, init in [
            ("slider_lx", "LX", -5.0, 5.0,  0.0),
            ("slider_ly", "LY", -5.0, 5.0,  5.0),
            ("slider_lz", "LZ", -5.0, 5.0, -5.0),
        ]:
            slider, val_lbl = make_slider(lo, hi, init)
            slider.valueChanged.connect(
                lambda s, a=axis, sl=slider: self._on_light(a, sl._from_s(s))
            )
            setattr(self, attr, slider)
            row = QHBoxLayout()
            row.addWidget(QLabel(axis))
            row.addWidget(slider)
            row.addWidget(val_lbl)
            layout.addLayout(row)
 
        self.slider_amb, amb_lbl = make_slider(0.0, 1.0, 0.5)
        self.slider_amb.valueChanged.connect(
            lambda s, sl=self.slider_amb: self._on_ambient(sl._from_s(s))
        )
        row = QHBoxLayout()
        row.addWidget(QLabel("Amb"))
        row.addWidget(self.slider_amb)
        row.addWidget(amb_lbl)
        layout.addLayout(row)
 
        return box
 
    # ── Centre panel ──────────────────────────────────────────────────────────
    def _build_center_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
 
        self.render_label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.render_label.setMinimumSize(512, 512)
        layout.addWidget(self.render_label, stretch=1)
 
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
 
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
 
        btn_row = QHBoxLayout()
        self.btn_detect = QPushButton("Detect Object")
        self.btn_detect.setObjectName("detect_btn")
        self.btn_save   = QPushButton("Save Image")
        self.btn_start  = QPushButton("▶ Start Optimisation")
        self.btn_start.setObjectName("start_btn")
        self.btn_stop   = QPushButton("■ Stop")
        self.btn_stop.setObjectName("stop_btn")
        for b in (self.btn_detect, self.btn_save, self.btn_start, self.btn_stop):
            btn_row.addWidget(b)
        layout.addLayout(btn_row)
 
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Target class:"))
        self.target_spin = QSpinBox()
        self.target_spin.setRange(0, 999)
        target_row.addWidget(self.target_spin)
        self.target_search = QLineEdit()
        self.target_search.setPlaceholderText("Search class name…")
        target_row.addWidget(self.target_search)
        self.btn_search = QPushButton("Search")
        target_row.addWidget(self.btn_search)
        layout.addLayout(target_row)
 
        opt_row = QHBoxLayout()
        opt_row.addWidget(QLabel("LR:"))
        self._lr_spin = QDoubleSpinBox()
        self._lr_spin.setRange(0.0001, 1.0)
        self._lr_spin.setSingleStep(0.01)
        self._lr_spin.setDecimals(4)
        self._lr_spin.setValue(0.05)
        opt_row.addWidget(self._lr_spin)
        opt_row.addWidget(QLabel("Steps:"))
        self._steps_spin = QSpinBox()
        self._steps_spin.setRange(1, 10000)
        self._steps_spin.setValue(200)
        opt_row.addWidget(self._steps_spin)
        layout.addLayout(opt_row)
 
        self._step_label    = QLabel("Step: —")
        self._prob_label    = QLabel("Target prob: —")
        self._prob_label.setObjectName("prob_label")
        self._success_label = QLabel("")
        self._success_label.setObjectName("success_label")
        self._saved_label   = QLabel("")
        self._saved_label.setObjectName("warn_label")
        for lbl in (self._step_label, self._prob_label,
                    self._success_label, self._saved_label):
            layout.addWidget(lbl)
 
        return panel
 
    # ── Right panel ───────────────────────────────────────────────────────────
    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("Top-5 Predictions"))
 
        self.pred_labels: list[QLabel]       = []
        self.pred_bars:   list[QProgressBar] = []
        for _ in range(5):
            lbl = QLabel("—")
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(False)
            layout.addWidget(lbl)
            layout.addWidget(bar)
            self.pred_labels.append(lbl)
            self.pred_bars.append(bar)
 
        layout.addWidget(QLabel("Target probability"))
        self.target_prob_label = QLabel("0.00 %")
        self.target_prob_bar   = QProgressBar()
        self.target_prob_bar.setRange(0, 100)
        layout.addWidget(self.target_prob_label)
        layout.addWidget(self.target_prob_bar)
 
        layout.addStretch()
        return panel