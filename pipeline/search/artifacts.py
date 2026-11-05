import csv
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def save_image(img: np.ndarray, path: str) -> None:
    Image.fromarray(img.astype(np.uint8)).save(path)


def initialize_metrics_file(metrics_csv: Optional[str]) -> None:
    if metrics_csv is None:
        return
    metrics_path = Path(metrics_csv)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as csv_file:
        csv.writer(csv_file).writerow(
            ("step", "parameter", "probability_percent", "snapshot_name")
        )


def append_metrics_row(
    metrics_csv: Optional[str],
    step: int,
    parameter: str,
    probability: float,
    snapshot_name: str,
) -> None:
    if metrics_csv is None:
        return
    with Path(metrics_csv).open("a", newline="", encoding="utf-8") as csv_file:
        csv.writer(csv_file).writerow(
            (step, parameter, probability * 100.0, snapshot_name)
        )
