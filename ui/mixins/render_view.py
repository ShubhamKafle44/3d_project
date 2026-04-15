from __future__ import annotations

import numpy as np
import torch
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap


class RenderViewMixin:
    """Handles rendering, image display, prediction bars, and slider sync."""

    def _refresh_render(self) -> None:
        """Re-render the scene and refresh all classifier prediction bars."""
        # We use torch.no_grad() because this is just for display; 
        # we don't need to keep track of math history for gradients here.
        with torch.no_grad():
            # Get the current 3D view as both a numpy image and a torch tensor
            img_np, img_t = self.scene.render()
            
        # Push the picture to the screen
        self._show_image(img_np)
        
        # Ask the Model what it sees in the new image and update the result bars
        self._update_predictions(self.classifier.top_predictions(img_t))

    def _show_image(self, img_np: np.ndarray) -> None:
        """Convert a float32 HxWx3 array to QPixmap and display it."""
        # Convert the raw 0.0-1.0 math numbers into standard 0-255 colors (integers)
        img_u8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
        h, w, c = img_u8.shape
        
        # Turn the raw data into a format that the windowing system understands
        q_img   = QImage(img_u8.tobytes(), w, h, w * c, QImage.Format.Format_RGB888)
        pixmap  = QPixmap.fromImage(q_img)
        
        # Scale the image so it fits nicely in our window without looking stretched
        self.render_label.setPixmap(
            pixmap.scaled(
                self.render_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        
        # Keep a copy of the latest image in case the user wants to hit "Save"
        self._last_img_np = img_np.copy()

    def _update_predictions(self, top5: list[tuple[str, float]]) -> None:
        """Populate the five prediction rows with labels and confidence bars."""
        # Loop through our five UI result rows and fill them with the AI's guesses
        for i, (name_lbl, bar) in enumerate(self._pred_labels):
            if i < len(top5):
                label, prob = top5[i]
                # Cut the label text if it's too long, then set the bar to show the %
                name_lbl.setText(label[:28])
                bar.setValue(int(prob * 100))
                bar.setFormat(f"{prob * 100:.1f}%")
            else:
                # If there are fewer than 5 results, clear the leftover rows
                name_lbl.setText("—")
                bar.setValue(0)
                bar.setFormat("")

    def _sync_sliders(self) -> None:
        """Move sliders to match current (post-optimisation) scene parameters."""
        # Pull the values out of the 3D math engine (PyTorch)
        # We detach them to make sure we're just getting the current values
        pos = self.scene.pos.detach().cpu()
        rot = self.scene.rot.detach().cpu()
        amb = float(self.scene.ambient_intensity.item())
        lp  = self.scene.light_pos.detach().cpu().squeeze()

        # Update Position and Rotation sliders
        for i, ax in enumerate(["X", "Y", "Z"]):
            for slider_dict, vals in (
                (self._pos_sliders, pos),
                (self._rot_sliders, rot),
            ):
                s = slider_dict[ax]
                # CRITICAL: We block signals here so the slider doesn't trigger 
                # a "value changed" event and start an infinite loop of updates.
                s.blockSignals(True)
                s.setValue(s._to_s(vals[i].item())) # Convert math value back to slider steps
                s.blockSignals(False)

        # Update the Ambient Light slider
        self._ambient_slider.blockSignals(True)
        self._ambient_slider.setValue(self._ambient_slider._to_s(amb))
        self._ambient_slider.blockSignals(False)

        # Update the Light Position (LX, LY, LZ) sliders
        for i, k in enumerate(["LX", "LY", "LZ"]):
            s = self._lp_sliders[k]
            s.blockSignals(True)
            s.setValue(s._to_s(lp[i].item()))
            s.blockSignals(False)
