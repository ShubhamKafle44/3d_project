from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PyQt6.QtWidgets import QMessageBox

import config


class ActionsMixin:

    # ──────────────────────────────────────────────────────────────────────────
    # Button wiring
    # ──────────────────────────────────────────────────────────────────────────

    def _connect_actions(self):
        self.btn_detect.clicked.connect(self._on_detect)
        self.btn_save.clicked.connect(self._on_save)
        self.btn_start.clicked.connect(self._on_start_optimise)
        self.btn_stop.clicked.connect(self._on_stop_optimise)
        self.btn_search.clicked.connect(self._on_search_class)

    # ──────────────────────────────────────────────────────────────────────────
    # Original button handlers
    # ──────────────────────────────────────────────────────────────────────────

    def _on_detect(self):
        img = self._current_classify_tensor()
        if img is None:
            return
        results = self.classifier.classify(img)
        self._update_predictions(results)
        # Auto-set adversarial target to the second-best prediction
        if len(results) > 1:
            idx = self._label_to_index(results[1][0])
            if idx is not None:
                self.target_spin.setValue(idx)

    def _on_save(self):
        img_np = getattr(self, "_last_render", None)
        if img_np is None:
            return
        path = "adversarial_output.png"
        Image.fromarray(
            (img_np * 255).clip(0, 255).astype(np.uint8)
        ).save(path)
        self.status_label.setText(f"Saved → {path}")

    def _on_search_class(self):
        query = self.target_search.text().strip().lower()
        for idx, label in enumerate(self.classifier.categories):
            if query in label.lower():
                self.target_spin.setValue(idx)
                self.status_label.setText(f"Found: {label} (#{idx})")
                return
        self.status_label.setText(f"No class matching '{query}'")

    # ──────────────────────────────────────────────────────────────────────────
    # NEW: Dropdown handlers
    # ──────────────────────────────────────────────────────────────────────────

    def _on_3d_model_changed(self, index: int):
        """Load the OBJ at AVAILABLE_3D_MODELS[index] into the scene."""
        label, obj_path = config.AVAILABLE_3D_MODELS[index]

        if not Path(obj_path).exists():
            QMessageBox.warning(
                self, "Model not found",
                f"OBJ file not found:\n{obj_path}\n\n"
                "Add the file to your models/ directory and try again.",
            )
            # Revert combo to the last successfully loaded index
            self.combo_3d_model.blockSignals(True)
            self.combo_3d_model.setCurrentIndex(self._current_3d_model_index)
            self.combo_3d_model.blockSignals(False)
            return

        self._current_3d_model_index = index
        self.status_label.setText(f"Loading 3D model: {label}…")
        try:
            # MeshScene.load_mesh(path) is the real method from renderer.py
            self.scene.load_mesh(obj_path)
            self._render_and_display()
            self.status_label.setText(f"3D model: {label}")
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            self.status_label.setText("Failed to load model.")

    # ------------------------------------------------------------------
    def _on_image_changed(self, index: int):
        """Switch between live render and a static image file."""
        label, img_path = config.AVAILABLE_IMAGES[index]
        self._static_image_path = img_path   # None → live render

        if img_path is None:
            self.status_label.setText("Image input: live 3D render")
        else:
            if not Path(img_path).exists():
                QMessageBox.warning(
                    self, "Image not found",
                    f"Image file not found:\n{img_path}",
                )
                self.combo_image.blockSignals(True)
                self.combo_image.setCurrentIndex(0)   # revert to live render
                self.combo_image.blockSignals(False)
                self._static_image_path = None
                return
            self.status_label.setText(f"Image input: {Path(img_path).name}")

        self._render_and_display()

    # ------------------------------------------------------------------
    def _on_vision_model_changed(self, index: int):
        """Hot-swap the torchvision classifier backbone."""
        label, model_key = config.AVAILABLE_VISION_MODELS[index]
        self.status_label.setText(f"Loading vision model: {label}…")
        try:
            self.classifier.swap_model(model_key)
            self.status_label.setText(f"Vision model: {label}")
            self._render_and_display()
        except Exception as exc:
            QMessageBox.critical(self, "Model swap error", str(exc))
            self.status_label.setText("Failed to load vision model.")

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _current_classify_tensor(self) -> torch.Tensor | None:
        """Return the image tensor the classifier should process.

        Priority:
          1. static image path (set by Image Selector dropdown)
          2. last rendered numpy frame stored in self._last_render
        """
        path = getattr(self, "_static_image_path", None)
        if path is not None:
            try:
                arr = (
                    np.array(Image.open(path).convert("RGB"))
                    .astype(np.float32) / 255.0
                )
                return torch.from_numpy(arr).to(config.device)
            except Exception:
                pass  # fall through to render

        img_np = getattr(self, "_last_render", None)
        if img_np is None:
            return None
        return torch.from_numpy(img_np.astype(np.float32)).to(config.device)

    def _label_to_index(self, label: str) -> int | None:
        for i, cat in enumerate(self.classifier.categories):
            if cat == label:
                return i
        return None