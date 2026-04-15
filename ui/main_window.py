import logging
import os
import numpy as np
import torch

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout,
    QStatusBar, QSplitter, QFileDialog
)
from PyQt6.QtCore import Qt, QThread

# Core logic for 3D rendering and  classification
from core.renderer import MeshScene
from core.classifier import ImageClassifier, IMAGENET_LABELS, AdversarialOptimiser

# The background worker that keeps the UI smooth during heavy math
from ui.worker import OptimWorker

# These Mixins are the 'brains' we built in previous files; we combine them all here
from ui.mixins.panels import PanelsMixin
from ui.mixins.actions import ActionsMixin
from ui.mixins.slider_callbacks import SliderCallbacksMixin
from ui.mixins.render_view import RenderViewMixin
from ui.mixins.optimisation import OptimisationMixin
from config import OBJ_PATH
from ui.stylesheets import STYLESHEET

logger = logging.getLogger(__name__)

class MainWindow(
    QMainWindow,
    PanelsMixin,
    ActionsMixin,
    SliderCallbacksMixin,
    RenderViewMixin,
    OptimisationMixin,
):
    """The main control center that ties the 3D scene, the AI, and the UI together."""

    def __init__(self, device: torch.device):
        super().__init__()

        # Store the hardware choice (CPU or GPU) and initialize our core engines
        self.device = device
        self.scene = MeshScene(device)
        self.classifier = ImageClassifier(device)

        # State management for the optimization process
        self.optimiser = None
        self.opt_thread = None
        self._worker = None
        self._last_img_np = None # Stores the latest pixels for saving to disk
        self._adv_save_path = None
        self._original_top_class = -1 # Keeps track of what the model thought BEFORE the attack

        # Basic window setup
        self.setWindowTitle("Adversarial 3D Mesh Renderer")
        self.resize(1340, 860)
        self.setStyleSheet(STYLESHEET)

        # Create the interface, draw the first frame, and update the status bar
        self._setup_main_layout()
        self._refresh_render()
        self.status_bar.showMessage(f"Device: {self.device} | System Ready")

    def _setup_main_layout(self):
        """Creates the three-column layout (Controls | Viewport | Results)."""
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Build panels using the logic from PanelsMixin
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        
        # Set the starting width for each panel (Left, Center, Right)
        splitter.setSizes([300, 640, 380])

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(splitter)
        layout.setContentsMargins(8, 8, 8, 8)
        self.setCentralWidget(central)

        # The bottom bar that shows status messages to the user
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def _refresh_render(self):
        """Forces the 3D engine to draw and the model to re-evaluate the image."""
        img_np, img_t = self.scene.render()
        self._show_image(img_np)
        
        # Get the AI's guesses and update those little progress bars on the right
        top5 = self.classifier.top_predictions(img_t)
        self._update_predictions(top5)

    def _on_opt_step(self, result: dict):
        """Callback that runs every single time the optimizer finishes a 'loop'."""
        step = result.get("step", 0)
        prob = result.get("prob", 0.0)
        img_np = result.get("img_np")

        # Update the iteration count and how much we have targeted the class
        self._step_label.setText(f"Step: {step}")
        self._prob_label.setText(f"Target prob: {prob * 100:.2f}%")

        # If we got a fresh image from the math loop, display it
        if img_np is not None:
            self._show_image(img_np)
        
        # Update the AI's current top 5 guesses as they shift during the attack
        if "top5" in result:
            self._update_predictions(result["top5"])

        # Move the 3D sliders to match the new 'adversarial' position/rotation
        self._sync_sliders()

        # If we hit our goal (fooled the classifier), handle the celebration and saving
        if result.get("success"):
            target = self._target_spin.value()
            name = IMAGENET_LABELS[target]
            self._success_label.setText(f"Classifier: '{name}' ({prob*100:.1f}%)")

            if img_np is not None:
                # Save the 'poisoned' image so we have proof the attack worked
                save_path = f"adversarial_{name.replace(' ', '_')}.png"
                self._save_numpy_image(img_np, save_path)
                self._adv_save_path = save_path
                self.status_bar.showMessage(f"Saved → {save_path}")

    def _on_opt_finished(self):
        """Cleanup logic for when the optimization thread stops or finishes."""
        # Turn the UI buttons back on so the user can try again
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._detect_btn.setEnabled(True)

        # If the progress bar stopped without reaching 100%, let the user know
        if not self._success_label.text():
            self.status_bar.showMessage("Optimisation finished (target not reached).")