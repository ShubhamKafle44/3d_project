from __future__ import annotations

import numpy as np
import torch
from PIL import Image as PILImage

from PyQt6.QtGui  import QImage, QPixmap
from PyQt6.QtCore import Qt

import config


class RenderViewMixin:

    # ──────────────────────────────────────────────────────────────────────────
    def _render_and_display(self):
        """Render scene → optionally replace with static image → classify → show."""

        # 1. Always render the 3-D scene.
        #    scene.render() returns (img_np, img_t) — unpack the tuple.
        img_np, _img_t = self.scene.render()   # img_np: (H, W, 3) float32 [0,1]
        self._last_render = img_np             # kept for Save button & optimiser

        # 2. Decide which image to display & classify
        static_path = getattr(self, "_static_image_path", None)
        if static_path is not None:
            try:
                display_np = (
                    np.array(PILImage.open(static_path).convert("RGB"))
                    .astype(np.float32) / 255.0
                )
            except Exception:
                display_np = img_np            # fall back to render on I/O error
        else:
            display_np = img_np

        # 3. Show in the centre panel
        self._show_numpy(display_np)

        # 4. Classify and update the right-hand prediction bars
        img_tensor = torch.from_numpy(display_np).to(config.device)
        results    = self.classifier.classify(img_tensor)
        self._update_predictions(results)

    # ──────────────────────────────────────────────────────────────────────────
    def _refresh_render(self):
        """Called by slider callbacks — just re-render without re-classifying
        on every tiny slider tick to keep the UI snappy.
        Full classify only happens when the user clicks Detect or changes a
        dropdown.  Override this method to add throttling / debouncing."""
        self._render_and_display()

    # ──────────────────────────────────────────────────────────────────────────
    def _show_numpy(self, img: np.ndarray):
        """Convert (H, W, 3) float32 [0,1] → QPixmap and paint render_label."""
        rgb8  = (img.clip(0.0, 1.0) * 255).astype(np.uint8)
        h, w, _ = rgb8.shape
        qimg    = QImage(rgb8.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
        pixmap  = QPixmap.fromImage(qimg).scaled(
            self.render_label.width(),
            self.render_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.render_label.setPixmap(pixmap)

    # ──────────────────────────────────────────────────────────────────────────
    def _update_predictions(self, results: list[tuple[str, float]]):
        """Populate the five prediction label + progress-bar pairs."""
        for i, (label, prob) in enumerate(results[:5]):
            self.pred_labels[i].setText(f"{label}  {prob * 100:.1f}%")
            self.pred_bars[i].setValue(int(prob * 100))
        for i in range(len(results), 5):
            self.pred_labels[i].setText("—")
            self.pred_bars[i].setValue(0)

    # ──────────────────────────────────────────────────────────────────────────
    def _sync_sliders(self):
        """Push current scene tensor values back into the UI sliders.
        Called after each optimisation step so the sliders reflect the
        adversarially optimised pose / lighting."""
        with torch.no_grad():
            pos  = self.scene.pos.cpu().tolist()
            rot  = self.scene.rot.cpu().tolist()
            lpos = self.scene.light_pos[0].cpu().tolist()
            amb  = self.scene.ambient_intensity.item()

        for slider, val in [
            (self.slider_px, pos[0]),
            (self.slider_py, pos[1]),
            (self.slider_pz, pos[2]),
            (self.slider_rx, rot[0]),
            (self.slider_ry, rot[1]),
            (self.slider_rz, rot[2]),
            (self.slider_lx, lpos[0]),
            (self.slider_ly, lpos[1]),
            (self.slider_lz, lpos[2]),
            (self.slider_amb, amb),
        ]:
            slider.blockSignals(True)
            slider.setValue(slider._to_s(val))
            slider.blockSignals(False)