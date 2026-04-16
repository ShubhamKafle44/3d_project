from __future__ import annotations

import math

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.classifier import IMAGENET_LABELS
from ui.stylesheets import ACCENT2, BORDER, PANEL_BG, TEXT_DIM
from ui.widgets.slider_helper import make_slider


class PanelsMixin:

    # ── Left panel: Object & Lighting Controls ──────────────────────────────────

    def _build_left_panel(self) -> QScrollArea:
        # Create a scrollable area so the controls don't get cut off on smaller screens
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(14)
        layout.setContentsMargins(10, 10, 10, 10)

        # --- Position Controls ---
        # Group box to keep X, Y, Z movement sliders together
        pos_grp = QGroupBox("Position")
        pg = QGridLayout(pos_grp)
        pg.setSpacing(10)
        pg.setContentsMargins(10, 16, 10, 10)
        pg.setColumnStretch(1, 1)
        
        self._pos_sliders: dict = {}
        # Set up sliders for moving the object within a 10-unit range (-5 to 5)
        for i, (ax, init) in enumerate([("X", 0.0), ("Y", 0.0), ("Z", -2.5)]):
            lbl = QLabel(ax)
            lbl.setFixedWidth(20)
            # Create the slider and link it to the position update function
            s, v = make_slider(-5.0, 5.0, init,
                               on_change=lambda val, a=ax: self._on_pos(a, val))
            self._pos_sliders[ax] = s
            s.setMinimumHeight(28)
            pg.addWidget(lbl, i, 0)
            pg.addWidget(s,   i, 1)
            pg.addWidget(v,   i, 2)
        layout.addWidget(pos_grp)

        # --- Rotation Controls ---
        # Group box for spinning the object (uses radians: -pi to pi)
        rot_grp = QGroupBox("Rotation (radians)")
        rg = QGridLayout(rot_grp)
        rg.setSpacing(10)
        rg.setContentsMargins(10, 16, 10, 10)
        rg.setColumnStretch(1, 1)
        
        self._rot_sliders: dict = {}
        for i, ax in enumerate(["X", "Y", "Z"]):
            lbl = QLabel(ax)
            lbl.setFixedWidth(20)
            s, v = make_slider(-math.pi, math.pi, 0.0,
                               on_change=lambda val, a=ax: self._on_rot(a, val))
            self._rot_sliders[ax] = s
            s.setMinimumHeight(28)
            rg.addWidget(lbl, i, 0)
            rg.addWidget(s,   i, 1)
            rg.addWidget(v,   i, 2)
        layout.addWidget(rot_grp)

        # --- Lighting Controls ---
        # Controls for the overall brightness (Ambient) and the light's position (LX, LY, LZ)
        light_grp = QGroupBox("Lighting")
        lg = QGridLayout(light_grp)
        lg.setSpacing(10)
        lg.setContentsMargins(10, 16, 10, 10)
        lg.setColumnStretch(1, 1)

        la = QLabel("Ambient")
        la.setFixedWidth(60)
        self._ambient_slider, av = make_slider(0.0, 1.0, 0.5, on_change=self._on_ambient)
        self._ambient_slider.setMinimumHeight(28)
        lg.addWidget(la, 0, 0)
        lg.addWidget(self._ambient_slider, 0, 1)
        lg.addWidget(av, 0, 2)

        self._lp_sliders: dict = {}
        for i, (ax, init) in enumerate([("LX", 0.0), ("LY", 1.0), ("LZ", -2.0)]):
            lbl = QLabel(ax)
            lbl.setFixedWidth(60)
            s, v = make_slider(-5.0, 5.0, init,
                               on_change=lambda val, a=ax: self._on_light(a, val))
            self._lp_sliders[ax] = s
            s.setMinimumHeight(28)
            lg.addWidget(lbl, i + 1, 0)
            lg.addWidget(s,   i + 1, 1)
            lg.addWidget(v,   i + 1, 2)
        layout.addWidget(light_grp)

        # Push everything to the top and finish the scroll area
        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    # ── Center panel: The Main Viewport ────────────────────────────────────────

    def _build_center_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)
        layout.setContentsMargins(6, 6, 6, 6)

        # Aesthetic header for the viewport
        hdr = QLabel("RENDER VIEW")
        hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hdr.setStyleSheet(f"color:{TEXT_DIM}; font-size:10px; letter-spacing:1.5px;")
        layout.addWidget(hdr)

        # This is where the actual image from PyTorch3D will be displayed
        self.render_label = QLabel()
        self.render_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.render_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.render_label.setMinimumSize(400, 400)
        self.render_label.setStyleSheet(
            f"background:{PANEL_BG}; border:1px solid {BORDER};"
        )
        layout.addWidget(self.render_label, 1)

        # Action buttons for initiating the model detection and saving results
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self._detect_btn = QPushButton(" Detect Object")
        self._detect_btn.setObjectName("detect_btn")
        self._detect_btn.setToolTip(
            "Classify current render, pick an adversarial target, and start optimisation."
        )
        self._detect_btn.clicked.connect(self._on_detect)
        btn_row.addWidget(self._detect_btn, 2)

        save_btn = QPushButton(" Save Image")
        save_btn.clicked.connect(self._on_save_image)
        btn_row.addWidget(save_btn, 1)

        layout.addLayout(btn_row)
        return widget

    # ── Right panel: Model Results & Attack Settings ───────────────────────────────

    def _build_right_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(14)
        layout.setContentsMargins(10, 10, 10, 10)

        # --- Model Classifier Output ---
        # Displays the top 5 guesses from the neural network with tiny progress bars for confidence
        pred_grp = QGroupBox("Classifier")
        pg = QVBoxLayout(pred_grp)
        pg.setSpacing(8)
        pg.setContentsMargins(10, 16, 10, 10)
        self._pred_labels: list = []
        for _ in range(5):
            row = QHBoxLayout()
            row.setSpacing(8)
            name_lbl = QLabel("—")
            name_lbl.setFixedWidth(200)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFixedHeight(3)
            row.addWidget(name_lbl)
            row.addWidget(bar)
            pg.addLayout(row)
            self._pred_labels.append((name_lbl, bar))
        layout.addWidget(pred_grp)

        # --- Adversarial Attack Controls ---
        # Settings for "tricking" the Model into seeing a different object
        adv_grp = QGroupBox("Adversarial Optimisation")
        ag = QVBoxLayout(adv_grp)
        ag.setSpacing(10)
        ag.setContentsMargins(10, 16, 10, 10)

        # Selector for the 'Target' class (what we want the Model to wrongly think it sees)
        tc_row = QHBoxLayout()
        tc_row.setSpacing(8)
        tc_row.addWidget(QLabel("Target class:"))
        self._target_spin = QSpinBox()
        self._target_spin.setRange(0, 999)
        self._target_spin.setValue(1)
        self._target_spin.valueChanged.connect(self._on_target_changed)
        tc_row.addWidget(self._target_spin)
        ag.addLayout(tc_row)

        # Display the human-readable name of the target class (e.g., 'Goldfish')
        self._target_name = QLabel(IMAGENET_LABELS[1] if len(IMAGENET_LABELS) > 1 else "—")
        self._target_name.setStyleSheet(f"color:{ACCENT2}; font-size:13px;")
        ag.addWidget(self._target_name)

        # Quick search box to find specific categories among the 1,000 possibilities
        search_row = QHBoxLayout()
        search_row.setSpacing(6)
        search_row.addWidget(QLabel("Search:"))
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("e.g. cat, tench, warplane…")
        self._search_edit.returnPressed.connect(self._on_search_class)
        search_row.addWidget(self._search_edit)
        sb = QPushButton("→")

        sb.setFixedWidth(36)

        sb.clicked.connect(self._on_search_class)

        search_row.addWidget(sb)

        ag.addLayout(search_row)


        # Learning rate

        lr_row = QHBoxLayout()

        lr_row.setSpacing(8)

        lr_row.addWidget(QLabel("Learning rate:"))

        self._lr_spin = QDoubleSpinBox()

        self._lr_spin.setRange(0.001, 1.0)

        self._lr_spin.setSingleStep(0.005)

        self._lr_spin.setValue(0.02)

        self._lr_spin.setDecimals(3)

        lr_row.addWidget(self._lr_spin)

        ag.addLayout(lr_row)


        # Max steps

        steps_row = QHBoxLayout()

        steps_row.setSpacing(8)

        steps_row.addWidget(QLabel("Max steps:"))

        self._steps_spin = QSpinBox()

        self._steps_spin.setRange(10, 2000)

        self._steps_spin.setValue(200)

        steps_row.addWidget(self._steps_spin)

        ag.addLayout(steps_row)


        # Start / Stop

        btn_row = QHBoxLayout()

        btn_row.setSpacing(8)

        self._start_btn = QPushButton("▶  Start Optimisation")

        self._start_btn.setObjectName("start_btn")

        self._start_btn.clicked.connect(self._on_start_optimise)

        self._stop_btn = QPushButton("■  Stop")

        self._stop_btn.setObjectName("stop_btn")

        self._stop_btn.clicked.connect(self._on_stop_optimise)

        self._stop_btn.setEnabled(False)

        btn_row.addWidget(self._start_btn)

        btn_row.addWidget(self._stop_btn)

        ag.addLayout(btn_row)


        self._step_label = QLabel("Step: —")

        self._step_label.setStyleSheet(f"color:{TEXT_DIM}; font-size:12px;")

        ag.addWidget(self._step_label)


        self._prob_label = QLabel("Target prob: —")

        self._prob_label.setObjectName("prob_label")

        ag.addWidget(self._prob_label)


        self._success_label = QLabel("")

        self._success_label.setObjectName("success_label")

        ag.addWidget(self._success_label)


        self._saved_label = QLabel("")

        self._saved_label.setWordWrap(True)

        self._saved_label.setObjectName("warn_label")

        ag.addWidget(self._saved_label)


        layout.addWidget(adv_grp)

        layout.addStretch()

        return widget 