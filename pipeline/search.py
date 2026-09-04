from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np
from PIL import Image

import config
from renderer import DifferentiableScene

if TYPE_CHECKING:
    from detector import HumanDetectionClassifier

PROPERTIES = ("POSITION", "ROTATION", "LIGHTING", "CLOTHING")


def _get_property(scene: DifferentiableScene, prop: str):
    if prop == "POSITION":
        return scene.get_position()
    if prop == "ROTATION":
        return scene.get_rotation_deg()
    if prop == "LIGHTING":
        return scene.get_lighting()
    if prop == "CLOTHING":
        return scene.get_material_color("shirt")
    raise ValueError(prop)


def _set_property(scene: DifferentiableScene, prop: str, value) -> None:
    if prop == "POSITION":
        scene.set_position(*value)
    elif prop == "ROTATION":
        scene.set_rotation_deg(*value)
    elif prop == "LIGHTING":
        scene.set_lighting(float(value))
    elif prop == "CLOTHING":
        scene.set_material_color("shirt", tuple(value))
    else:
        raise ValueError(prop)


def _perturb(current, prop: str, step_size: float):
    lo, hi = config.PROPERTY_BOUNDS[prop]
    if prop in ("POSITION", "ROTATION", "CLOTHING"):
        noise = np.random.normal(0, step_size, size=3)
        return np.clip(np.asarray(current) + noise, lo, hi)
    else:  # LIGHTING - scalar
        noise = np.random.normal(0, step_size)
        return float(np.clip(current + noise, lo, hi))


def _format_property(prop: str, value) -> str:
    if prop in ("POSITION", "ROTATION", "CLOTHING"):
        a, b, c = np.asarray(value).tolist()
        return f"{prop.lower()}=({a:+.3f}, {b:+.3f}, {c:+.3f})"
    return f"{prop.lower()}=({float(value):+.3f})"


@dataclass
class SearchResult:
    success: bool
    best_step: int
    best_prob: float
    best_image: np.ndarray
    best_property_value: object


def run_adversarial_search(
    scene: DifferentiableScene,
    classifier: "HumanDetectionClassifier",
    property_name: str,
    epochs: int = 100,
    step_size: float = 0.1,
    success_threshold: Optional[float] = None,
    verbose: bool = True,
) -> SearchResult:
    if property_name not in PROPERTIES:
        raise ValueError(f"property must be one of {PROPERTIES}, got {property_name!r}")
    success_threshold = success_threshold if success_threshold is not None else config.SEARCH["success_threshold"]

    initial_img = scene.render()
    if initial_img is None:
        raise RuntimeError("Initial render returned None - check mesh paths in config.py")

    initial_result = classifier.classify(initial_img)
    best_prob = initial_result["human_prob"]
    best_img = initial_img.copy()
    best_step = 0
    current_value = _get_property(scene, property_name)
    best_value = current_value

    if verbose:
        print(f"Initial human_prob = {best_prob * 100:.2f}%")
        print(f"{'Step':>4} {'HumanProb':>10} {'Persons':>8}  {'TopLabel':<10} Parameters")
        print("-" * 70)

    for step in range(1, epochs + 1):
        proposal = _perturb(current_value, property_name, step_size)
        _set_property(scene, property_name, proposal)

        img = scene.render()
        if img is None:
            continue

        result = classifier.classify(img)
        human_prob = result["human_prob"]

        if human_prob < best_prob:
            best_prob, best_step, best_img, best_value = human_prob, step, img.copy(), proposal
            current_value = proposal
        else:
            _set_property(scene, property_name, current_value)  # revert

        if verbose:
            print(f"{step:4d} {human_prob * 100:9.2f}% {result['num_persons']:8d}  "
                  f"{result['top_label']:<10} {_format_property(property_name, proposal)}")

        if best_prob <= success_threshold:
            if verbose:
                print("-" * 70)
                print(f"Success at step {step}: human_prob {best_prob * 100:.2f}% <= "
                      f"{success_threshold * 100:.0f}%")
            _set_property(scene, property_name, best_value)
            return SearchResult(True, best_step, best_prob, best_img, best_value)

    if verbose:
        print("-" * 70)
        print(f"Did not reach threshold in {epochs} epochs. "
              f"Best: step {best_step}, human_prob {best_prob * 100:.2f}%")
    _set_property(scene, property_name, best_value)
    return SearchResult(False, best_step, best_prob, best_img, best_value)


def save_image(img: np.ndarray, path: str) -> None:
    Image.fromarray(img.astype(np.uint8)).save(path)
