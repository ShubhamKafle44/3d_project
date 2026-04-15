from __future__ import annotations

import logging
import os

import numpy as np
from PyQt6.QtWidgets import QFileDialog

from core.classifier import IMAGENET_LABELS

# Standard logger setup to track what the user is doing in the background
logger = logging.getLogger(__name__)


class ActionsMixin:
    """Button-triggered actions: detect, load mesh, save image, class search."""

    def _on_detect(self) -> None:
        """
        1. Classify the current render.
        2. Pick the second-best class as the adversarial target.
        3. Auto-start optimisation.
        """
        # Clear out any old status messages before starting a new run
        self._success_label.setText("")
        self._saved_label.setText("")

        # Grab the latest frame from our 3D scene and display it in the UI
        img_np, img_t = self.scene.render()
        self._show_image(img_np)

        # Ask the model what it thinks it sees (returns the top 5 guesses)
        top5 = self.classifier.top_predictions(img_t)
        self._update_predictions(top5)

        if not top5:
            self.status_bar.showMessage("Classifier returned no results.")
            return

        # Figure out the current #1 guess
        top_label, top_prob = top5[0]
        self._original_top_class = self.classifier.top_class_index(img_t)
        logger.info("Detect — top prediction: %s (%.2f%%)", top_label, top_prob * 100)
        self.status_bar.showMessage(
            f"Detected: {top_label}  ({top_prob * 100:.1f}%)  |  selecting adversarial target…"
        )

        # ADVERSARIAL LOGIC: 
        # We want to trick the AI. The easiest way is to pick its 'second' guess 
        # and try to make that become the 'first' guess.
        if len(top5) > 1:
            adv_label, _ = top5[1]
            # Find the ID number of that second-place label
            adv_idx = next(
                (i for i, n in enumerate(IMAGENET_LABELS) if n == adv_label),
                (self._original_top_class + 1) % 1000,
            )
        else:
            # If it only has one guess, just pick the next ID in the list as the target
            adv_idx   = (self._original_top_class + 1) % 1000
            adv_label = IMAGENET_LABELS[adv_idx]

        # Update the target selector in the UI so the user sees what we're aiming for
        self._target_spin.setValue(adv_idx)
        self._target_name.setText(adv_label)
        logger.info("Adversarial target: [%d] %s", adv_idx, adv_label)

        # Immediately kick off the math loop to start changing the 3D scene
        self._on_start_optimise()

    def _on_load_mesh(self) -> None:
        """Opens a standard Windows/Mac/Linux file picker to find a 3D .obj file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open OBJ Mesh", "", "OBJ Files (*.obj);;All Files (*)"
        )
        if path:
            # Tell the 3D scene to throw out the old model and load the new one
            self.scene.load_mesh(path)
            self._refresh_render()
            self.status_bar.showMessage(
                f"Device: {self.device}  |  Loaded: {os.path.basename(path)}"
            )

    def _on_save_image(self) -> None:
        """Triggered when the user clicks 'Save'. Asks where to put the file."""
        if self._last_img_np is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "render.png", "PNG (*.png);;JPEG (*.jpg)"
        )
        if path:
            # Take the data currently in memory and write it to the hard drive
            self._save_numpy_image(self._last_img_np, path)
            self.status_bar.showMessage(f"Saved → {path}")

    def _save_numpy_image(self, img_np: np.ndarray, path: str) -> None:
        """Helper to convert raw math numbers into a real image file (0.0-1.0 -> 0-255)."""
        from PIL import Image as PILImage
        # Clip ensures no numbers go below 0 or above 1 before we multiply by 255
        img_u8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
        PILImage.fromarray(img_u8).save(path)
        logger.info("Image saved: %s", path)

    def _on_target_changed(self, idx: int) -> None:
        """Updates the text label whenever the user manually spins the target ID box."""
        if idx < len(IMAGENET_LABELS):
            self._target_name.setText(IMAGENET_LABELS[idx])

    def _on_search_class(self) -> None:
        """Allows the user to type 'Dog' and automatically find the right ImageNet ID."""
        query = self._search_edit.text().strip().lower()
        if not query:
            return
        
        # Loop through all 1000 categories to find a name match
        for i, name in enumerate(IMAGENET_LABELS):
            if query in name.lower():
                # Update the spin box, which will trigger _on_target_changed too
                self._target_spin.setValue(i)
                return
        
        self.status_bar.showMessage(f"No class matched '{query}'")