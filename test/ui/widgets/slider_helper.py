from PyQt6.QtCore    import Qt
from PyQt6.QtWidgets import QLabel, QSlider

_STEPS = 1000   # resolution: 1/1000 of the full range per tick


def make_slider(
    lo: float,
    hi: float,
    init: float,
    decimals: int = 2,
) -> tuple[QSlider, QLabel]:
    """Create a horizontal QSlider that represents floats in [lo, hi].

    The slider's integer range is always [0, 1000], linearly mapped to [lo, hi].

    Two helpers are monkey-patched onto the returned slider object:
        slider._to_s(v: float) -> int   — convert a float value to a slider int
        slider._from_s(s: int) -> float — convert a slider int to a float value

    Returns
    -------
    slider : QSlider
    val_lbl : QLabel   — updates automatically as the slider moves
    """
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(0, _STEPS)

    # ── Conversion helpers (monkey-patched so panels & sync_sliders can use them) ──
    def _to_s(v: float) -> int:
        return int((v - lo) / (hi - lo) * _STEPS)

    def _from_s(s: int) -> float:
        return lo + s / _STEPS * (hi - lo)

    slider._to_s   = _to_s
    slider._from_s = _from_s

    slider.setValue(_to_s(init))

    # ── Paired value label ────────────────────────────────────────────────────
    val_lbl = QLabel(f"{init:.{decimals}f}")
    val_lbl.setObjectName("value_label")
    val_lbl.setFixedWidth(52)

    def _on_change(s: int) -> None:
        val_lbl.setText(f"{_from_s(s):.{decimals}f}")

    slider.valueChanged.connect(_on_change)

    return slider, val_lbl