import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from openpyxl import Workbook, load_workbook
import torch

import numpy as np
from PIL import Image


def create_optimization_workbook(metrics_csv: Optional[str]) -> Optional[Path]:
    """Create a unique workbook beside metrics.csv for a single search run."""
    if metrics_csv is None:
        return None
    metrics_path = Path(metrics_csv)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    workbook_path = metrics_path.parent / f"gradient_descent_{timestamp}.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Optimization"
    sheet.append((
        "step", "property", "loss", "validation_probability", "position",
        "optimized_parameters", "gradients", "optimizer_state",
        "learning_rates", "snapshot_name",
    ))
    workbook.save(workbook_path)
    return workbook_path


def append_optimization_record(
    workbook_path: Optional[Path], *, step: int, property_name: str,
    loss: Optional[float], validation_probability: Optional[float], position,
    parameters, gradients, optimizer, snapshot_name: str = "",
) -> None:
    """Append one optimizer-step observation, keeping tensors Excel-friendly."""
    if workbook_path is None:
        return

    def encode(value):
        def normalize(item):
            if isinstance(item, torch.Tensor):
                item = item.detach().cpu().numpy()
            if isinstance(item, np.ndarray):
                item = item.tolist()
            if isinstance(item, dict):
                return {str(key): normalize(val) for key, val in item.items()}
            if isinstance(item, (list, tuple)):
                return [normalize(val) for val in item]
            if isinstance(item, np.generic):
                return item.item()
            return item
        return json.dumps(normalize(value), sort_keys=True)

    optimizer_state = []
    for parameter in parameters:
        state = optimizer.state.get(parameter, {}) if optimizer is not None else {}
        optimizer_state.append({
            key: (value.item() if isinstance(value, torch.Tensor) and value.numel() == 1
                  else value.detach().cpu().tolist() if isinstance(value, torch.Tensor)
                  else value)
            for key, value in state.items()
        })
    learning_rates = [group["lr"] for group in optimizer.param_groups] if optimizer is not None else []

    workbook = load_workbook(workbook_path)
    workbook.active.append((
        step, property_name, loss, validation_probability,
        encode(position), encode(parameters), encode(gradients),
        json.dumps(optimizer_state, sort_keys=True, default=str),
        encode(learning_rates), snapshot_name,
    ))
    workbook.save(workbook_path)


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
