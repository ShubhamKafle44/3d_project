import torch
class SliderCallbacksMixin:
    """Translates slider value-change events into 3-D scene parameter updates."""

    # Map the string labels (like 'X' or 'LY') to their numerical index in a 3D tensor [0, 1, 2]
    _AXIS_IDX = {"X": 0, "Y": 1, "Z": 2, "LX": 0, "LY": 1, "LZ": 2}

    def _on_pos(self, axis: str, val: float) -> None:
        """Updates the object's physical location when a position slider is moved."""
        # We use no_grad because manual slider tweaks shouldn't be recorded as Model training steps
        with torch.no_grad():
            self.scene.pos[self._AXIS_IDX[axis]] = val
        # Redraw the screen so the user sees the object move immediately
        self._refresh_render()

    def _on_rot(self, axis: str, val: float) -> None:
        """Updates the object's rotation angles when a rotation slider is moved."""
        with torch.no_grad():
            self.scene.rot[self._AXIS_IDX[axis]] = val
        # Redraw to show the new orientation
        self._refresh_render()

    def _on_ambient(self, val: float) -> None:
        """Adjusts the overall 'room brightness' (ambient light)."""
        with torch.no_grad():
            # fill_ overwrites the entire tensor value with the new slider percentage
            self.scene.ambient_intensity.fill_(val)
        self._refresh_render()

    def _on_light(self, axis: str, val: float) -> None:
        """Moves the light source around the scene."""
        with torch.no_grad():
            # light_pos is a batch tensor [1, 3], so we target the first batch at the specific axis
            self.scene.light_pos[0, self._AXIS_IDX[axis]] = val
        self._refresh_render()
