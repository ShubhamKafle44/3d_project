from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QSlider


def make_slider(
    lo: float,
    hi: float,
    init: float,
    decimals: int = 2,
    on_change: Callable[[float], None] | None = None,
) -> tuple[QSlider, QLabel]:
    """Create a float QSlider mapped linearly over [lo, hi] with 1 000 steps.

    The returned slider exposes two helpers:
        slider._to_s(float) -> int    convert float → slider integer
        slider._from_s(int) -> float  convert slider integer → float

    These are used by ``RenderViewMixin._sync_sliders`` to keep the UI in sync
    with the optimised scene parameters.
    """
    # Since QSliders only handle whole numbers, we define a fixed range of 1000 'notches'
    steps  = 1000
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(0, steps)

    # Helper function to find where a specific float sits on our 0-1000 scale
    def _to_s(v: float) -> int:
        # Normalize the value (0.0 to 1.0) and then multiply by our total steps
        return int((v - lo) / (hi - lo) * steps)

    # Helper function to turn a slider 'notch' back into a meaningful 3D coordinate or angle
    def _from_s(s: int) -> float:
        # Calculate the percentage across the slider and apply it to our float range
        return lo + s / steps * (hi - lo)

    # We 'monkey-patch' these helpers onto the slider object so they are easy to find later
    slider._to_s   = _to_s
    slider._from_s = _from_s
    # Set the slider's handle to the starting position based on the initial value provided
    slider.setValue(_to_s(init))

    # Create the text label that sits next to the slider to show the current number
    val_lbl = QLabel(f"{init:.{decimals}f}")
    val_lbl.setObjectName("value_label")
    val_lbl.setFixedWidth(50)

    # This inner function runs every single time the user drags the slider handle
    def _changed(s: int) -> None:
        # Convert the integer position to a float
        v = _from_s(s)
        # Update the text label to reflect the new number
        val_lbl.setText(f"{v:.{decimals}f}")
        # If we provided a callback (like moving the 3D mesh), trigger it now
        if on_change:
            on_change(v)

    # Link the physical movement of the slider to our logic above
    slider.valueChanged.connect(_changed)
    return slider, val_lbl