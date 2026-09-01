from __future__ import annotations

import numpy as np
import torch

from PyQt6.QtGui  import QImage, QPixmap
from PyQt6.QtCore import Qt

import config as config


class RenderViewMixin:

    def _render_and_display(self):
        """Render the 3-D scene (with current background) → display → classify."""

        img_np, _img_t = self.scene.render()   # (H, W, 3) float32 [0,1]
        self._last_render = img_np

        self._show_numpy(img_np)

        img_tensor = torch.from_numpy(img_np).to(config.device)
        results    = self.classifier.classify(img_tensor)
        self._update_predictions(results)

    def _refresh_render(self):
        """Called by slider callbacks to redraw without reclassifying."""
        self._render_and_display()
        
    def _show_numpy(self, img: np.ndarray):
        """Convert (H, W, 3) float32 [0,1] → QPixmap and paint render_label."""
        rgb8    = (img.clip(0.0, 1.0) * 255).astype(np.uint8)
        h, w, _ = rgb8.shape
        qimg    = QImage(rgb8.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)

        # Use the actual label size; fall back to IMAGE_SIZE if not yet shown
        lw = self.render_label.width()  or config.IMAGE_SIZE
        lh = self.render_label.height() or config.IMAGE_SIZE

        pixmap = QPixmap.fromImage(qimg).scaled(
            lw, lh,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.render_label.setPixmap(pixmap)
        self.render_label.repaint()   # force Qt to flush the new frame immediately

    def _update_predictions(self, results: list[tuple[str, float]]):
        for i, (label, prob) in enumerate(results[:5]):
            self.pred_labels[i].setText(f"{label}  {prob * 100:.1f}%")
            self.pred_bars[i].setValue(int(prob * 100))
        for i in range(len(results), 5):
            self.pred_labels[i].setText("—")
            self.pred_bars[i].setValue(0)

    def _sync_sliders(self):
        """Push optimised scene values back into the UI sliders."""
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