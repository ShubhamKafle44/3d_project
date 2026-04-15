# --Colour tokens --
BG       = "#0d1117"  # Deep navy slate
PANEL_BG = "#161b22"  # Lighter slate for depth
BORDER   = "#30363d"  # Subtle grey-blue borders
BORDER2  = "#21262d"  # Darker contrast border
TEXT     = "#c9d1d9"  # High-readability off-white
TEXT_DIM = "#8b949e"  # Muted grey for descriptions
TEXT_SUB = "#58a6ff"  # Soft blue for headers (changed from dull slate)
ACCENT   = "#58a6ff"  # AI Blue (Electric)
ACCENT2  = "#bc8cff"  # Adversarial Purple (Soft Lavender)
GRN      = "#3fb950"  # Success Emerald
DANGER   = "#f85149"  # Alert Crimson
WARN     = "#d29922"  # Warning Gold

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
    background: {TEXT};
    border: 2px solid {BG};
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
QPushButton:hover   {{ 
    border-color: {TEXT_DIM}; 
    background-color: #21262d; 
}}
QPushButton:pressed {{ 
    background-color: #0d1117; 
}}
QPushButton#detect_btn {{
    background-color: #121d2f;
    border-color: #1f3a5f;
    color: {ACCENT};
    font-weight: bold;
}}
QPushButton#detect_btn:hover {{ 
    background-color: #16263f; 
    border-color: {ACCENT}; 
}}
QPushButton#start_btn {{
    background-color: #0f1c14;
    border-color: #1b3a24;
    color: {GRN};
    font-weight: bold;
}}
QPushButton#start_btn:hover {{ 
    background-color: #142a1d; 
    border-color: {GRN}; 
}}
QPushButton#stop_btn {{
    background-color: #211314;
    border-color: #442326;
    color: {DANGER};
    font-weight: bold;
}}
QPushButton#stop_btn:hover {{ 
    background-color: #2d191b; 
    border-color: {DANGER}; 
}}
QLabel {{
    font-size: 13px;
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
    background: #0d1117;
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 6px 10px;
    color: {TEXT};
    font-family: "Consolas";
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {ACCENT};
    background: #161b22;
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