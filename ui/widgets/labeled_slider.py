from PyQt6 import QtCore, QtWidgets


class LabeledSlider(QtWidgets.QWidget):
    """A horizontal QSlider paired with a live-updating float label."""

    # Create a custom signal so other parts of the app can "hear" when the decimal value changes
    valueChanged = QtCore.pyqtSignal(float)

    def __init__(
        self,
        label: str,
        min_val: float,
        max_val: float,
        step: float,
        default: float,
        parent=None,
    ):
        super().__init__(parent)
        # We store the step size because QSlider only understands whole numbers (integers)
        self._step       = step
        self._label_text = label

        # Set up the text label that shows the user the actual decimal value
        self._label  = QtWidgets.QLabel(f"{label}: {default:.2f}")
        self._slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        
        # MATH TRICK: Since QSlider only does integers, we divide our floats by the step size.
        # For example, if min is 0.0 and step is 0.1, the slider's internal minimum becomes 0.
        self._slider.setMinimum(int(min_val / step))
        self._slider.setMaximum(int(max_val / step))
        self._slider.setValue(int(default / step))
        self._slider.setSingleStep(1)
        
        # When the physical slider moves, trigger our internal conversion logic
        self._slider.valueChanged.connect(self._on_change)

        # Stack the label on top of the slider
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)
        layout.addWidget(self._slider)

    # ── Public API ─────────────────────────────────────────────────────────────

    def setValue(self, val: float) -> None:
        """Move the slider to the given float value."""
        # Convert the float back into the slider's "integer language"
        self._slider.setValue(int(val / self._step))

    @property
    def value(self) -> float:
        """Current float value of the slider."""
        # Convert the slider's integer position back into a real decimal number
        return self._slider.value() * self._step

    # ── Internal ───────────────────────────────────────────────────────────────

    def _on_change(self, raw: int) -> None:
        """Handles the conversion from integer slider position to a real-world float."""
        # Calculate the actual decimal value (e.g., slider at 50 with 0.01 step = 0.50)
        real = raw * self._step
        
        # Update the text label so the user sees exactly what value they've selected
        self._label.setText(f"{self._label_text}: {real:.2f}")
        
        # Send out the float value to anyone listening to this widget
        self.valueChanged.emit(real)