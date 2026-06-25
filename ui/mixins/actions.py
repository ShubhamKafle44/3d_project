from __future__ import annotations 
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from PyQt6.QtWidgets import QMessageBox
import config

class ActionsMixin:
 
    # ── Button wiring ─────────────────────────────────────────────────────────
    def _connect_actions(self):
        self.btn_detect.clicked.connect(self._on_detect)
        self.btn_save.clicked.connect(self._on_save)
        self.btn_start.clicked.connect(self._on_start_optimise)
        self.btn_stop.clicked.connect(self._on_stop_optimise)
        self.btn_search.clicked.connect(self._on_search_class)
 
    # ── Button handlers ───────────────────────────────────────────────────────
    def _on_detect(self):
        img = self._current_classify_tensor()
        if img is None:
            return
        results = self.classifier.classify(img)
        self._update_predictions(results)
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
 
    # ── Dropdown handlers ─────────────────────────────────────────────────────
    def _on_3d_model_changed(self, index: int):
        label, obj_path = config.AVAILABLE_3D_MODELS[index]
 
        if not Path(obj_path).exists():
            QMessageBox.warning(
                self, "Model not found",
                f"OBJ file not found:\n{obj_path}\n\n"
                "Add the file to your models/ directory and try again.",
            )
            self.combo_3d_model.blockSignals(True)
            self.combo_3d_model.setCurrentIndex(self._current_3d_model_index)
            self.combo_3d_model.blockSignals(False)
            return
 
        self._current_3d_model_index = index
        self.status_label.setText(f"Loading 3D model: {label}…")
        try:
            self.scene.load_mesh(obj_path)
            self._reset_sliders()
            self._render_and_display()
            self.status_label.setText(f"3D model: {label}")
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            self.status_label.setText("Failed to load model.")
 
    def _on_image_changed(self, index: int):
        """Swap the background image. The 3D model always stays on top."""
        label, img_path = config.AVAILABLE_IMAGES[index]
        
        if not Path(img_path).exists():
            QMessageBox.warning(
                self, "Background not found",
                f"Image file not found:\n{img_path}\n\n"
                "Add the file to your background/ directory and try again.",
            )
            # Revert combo to whichever entry matches the current BACKGROUND_PATH
            current = config.BACKGROUND_PATH
            for i, (_, p) in enumerate(config.AVAILABLE_IMAGES):
                if p == current:
                    self.combo_image.blockSignals(True)
                    self.combo_image.setCurrentIndex(i)
                    self.combo_image.blockSignals(False)
                    break
            return
 
        # Update the global background path so that the renderer uses the new image
        config.BACKGROUND_PATH = img_path
        self.status_label.setText(f"Background: {label}")
        self._render_and_display()
 
    def _on_vision_model_changed(self, index: int):
        label, model_key = config.AVAILABLE_VISION_MODELS[index]
        self.status_label.setText(f"Loading vision model: {label}…")
        try:
            self.classifier.swap_model(model_key)
            self.status_label.setText(f"Vision model: {label}")
            self._render_and_display()
        except Exception as exc:
            QMessageBox.critical(self, "Model swap error", str(exc))
            self.status_label.setText("Failed to load vision model.")
 
    # ── Helpers ───────────────────────────────────────────────────────────────
    def _reset_sliders(self) -> None:
        defaults = {
            "slider_px": 0.0, "slider_py": 0.0, "slider_pz": 0.0,
            "slider_rx": 0.0, "slider_ry": 0.0, "slider_rz": 0.0,
            "slider_lx": 0.0, "slider_ly": 5.0, "slider_lz": -5.0,
            "slider_amb": 0.5,
        }
        for attr, val in defaults.items():
            slider = getattr(self, attr, None)
            if slider is not None:
                slider.blockSignals(True)
                slider.setValue(slider._to_s(val))
                slider.blockSignals(False)
 
    def _current_classify_tensor(self) -> torch.Tensor | None:
        img_np = getattr(self, "_last_render", None)
        if img_np is None:
            return None
        return torch.from_numpy(img_np.astype(np.float32)).to(config.device)
 
    def _label_to_index(self, label: str) -> int | None:
        for i, cat in enumerate(self.classifier.categories):
            if cat == label:
                return i
        return None
