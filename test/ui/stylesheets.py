# -- Colour tokens --
BG       = "#f6f8fa"  # GitHub-style light grey background
PANEL_BG = "#ffffff"  # Pure white panels
BORDER   = "#d0d7de"  # Soft grey borders
BORDER2  = "#e8ecf0"  # Lighter inner borders
TEXT     = "#1f2328"  # Near-black readable text
TEXT_DIM = "#656d76"  # Muted grey for descriptions
TEXT_SUB = "#0969da"  # Blue for headers
ACCENT   = "#0969da"  # Primary blue
ACCENT2  = "#8250df"  # Purple for adversarial
GRN      = "#1a7f37"  # Success green
DANGER   = "#cf222e"  # Alert red
WARN     = "#9a6700"  # Warning amber

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-family: "Segoe UI", "Roboto", "Consolas", monospace;
    font-size: 13px;
}}
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-top: 18px;
    padding-top: 14px;
    color: {TEXT_SUB};
    font-size: 10px;
    font-weight: bold;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    background-color: {PANEL_BG};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 5px;
    color: {TEXT_SUB};
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: {BORDER2};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
    background: {ACCENT};
    border: 2px solid {PANEL_BG};
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 2px;
}}
QPushButton {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 8px 16px;
    color: {TEXT};
    font-size: 13px;
    font-weight: 500;
}}
QPushButton:hover {{
    border-color: {ACCENT};
    background-color: #f3f8ff;
}}
QPushButton:pressed {{
    background-color: {BORDER2};
}}
QPushButton#detect_btn {{
    background-color: #dbeafe;
    border-color: #93c5fd;
    color: {ACCENT};
    font-weight: bold;
}}
QPushButton#detect_btn:hover {{
    background-color: #bfdbfe;
    border-color: {ACCENT};
}}
QPushButton#start_btn {{
    background-color: #dcfce7;
    border-color: #86efac;
    color: {GRN};
    font-weight: bold;
}}
QPushButton#start_btn:hover {{
    background-color: #bbf7d0;
    border-color: {GRN};
}}
QPushButton#stop_btn {{
    background-color: #fee2e2;
    border-color: #fca5a5;
    color: {DANGER};
    font-weight: bold;
}}
QPushButton#stop_btn:hover {{
    background-color: #fecaca;
    border-color: {DANGER};
}}
QLabel {{
    font-size: 13px;
    color: {TEXT};
}}
QLabel#value_label   {{ color: {ACCENT}; min-width: 48px; font-family: "Consolas"; }}
QLabel#prob_label    {{ color: {ACCENT2}; font-size: 15px; font-weight: bold; }}
QLabel#success_label {{ color: {GRN}; font-size: 13px; font-weight: bold; }}
QLabel#warn_label    {{ color: {WARN}; font-size: 12px; }}
QProgressBar {{
    border: none;
    border-radius: 2px;
    background: {BORDER2};
    color: transparent;
    height: 4px;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 2px;
}}
QStatusBar {{
    background: {PANEL_BG};
    color: {TEXT_DIM};
    border-top: 1px solid {BORDER};
    font-size: 11px;
}}
QScrollArea {{
    border: none;
    background: transparent;
}}
QLineEdit, QSpinBox, QDoubleSpinBox {{
    background: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 6px 10px;
    color: {TEXT};
    font-family: "Consolas";
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {ACCENT};
    background: #f0f6ff;
}}
QSplitter::handle {{
    background: {BORDER};
}}
QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {TEXT_DIM};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
"""