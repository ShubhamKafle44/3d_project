from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from PIL import Image

import config
from renderer import DifferentiableScene

if TYPE_CHECKING:
    from detector import ObjectDetectionClassifier

PROPERTIES = ("POSITION", "ROTATION", "POSE", "LIGHTING", "CLOTHING")


def _get_property(scene: DifferentiableScene, prop: str):
    if prop == "POSITION":
        return scene.get_position()
    if prop == "ROTATION":
        return scene.get_rotation_deg()
    if prop == "POSE":
        return scene.get_position(), scene.get_rotation_deg()
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
    elif prop == "POSE":
        position, rotation = value
        # PyTorch3D projects bounds using the current orientation.
        scene.set_rotation_deg(*rotation)
        scene.set_position(*position)
    elif prop == "LIGHTING":
        scene.set_lighting(float(value))
    elif prop == "CLOTHING":
        scene.set_material_color("shirt", tuple(value))
    else:
        raise ValueError(prop)


def _format_property(prop: str, value) -> str:
    if prop == "POSE":
        position, rotation = value
        return (f"position=({position[0]:+.3f}, {position[1]:+.3f}, {position[2]:+.3f}) "
                f"rotation=({rotation[0]:+.1f}°, {rotation[1]:+.1f}°, {rotation[2]:+.1f}°)")
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


def run_gradient_descent(
    scene: DifferentiableScene,
    classifier: "ObjectDetectionClassifier",
    property_name: str,
    epochs: int = 100,
    learning_rate: float = 0.06,
    validate_every: int = 5,
    detector_input_size: int = 384,
    success_threshold: Optional[float] = None,
    verbose: bool = True,
    snapshot_dir: Optional[str] = None,
    snapshot_every: int = 10,
    metrics_csv: Optional[str] = None,
) -> SearchResult:
    """Use renderer gradients and Adam to minimize the target-class score."""

    if not hasattr(scene, "differentiable_parameters"):
        raise TypeError("Gradient search requires differentiable scene parameters")

    
    loss_function = getattr(classifier, "target_detection_loss", None)


    if loss_function is None:
        loss_function = getattr(classifier, "person_detection_loss", None)
    if loss_function is None:
        raise TypeError("Gradient search requires a differentiable target-detection loss")
    if epochs < 1 or validate_every < 1 or snapshot_every < 1:
        raise ValueError("epochs, validate_every, and snapshot_every must be positive")
    if snapshot_dir is not None:
        Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
    if metrics_csv is not None:
        metrics_path = Path(metrics_csv)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", newline="", encoding="utf-8") as csv_file:
            csv.writer(csv_file).writerow(
                ("step", "parameter", "probability_percent", "snapshot_name")
            )
    if success_threshold is None:
        success_threshold = config.SEARCH["success_threshold"]

    initial_position = scene.get_position().astype(np.float32)
    if property_name in ("POSITION", "POSE") and not scene.is_position_valid(initial_position):
        raise RuntimeError("Initial position is outside the frame or intersects scene geometry")

    scene._active_differentiable_property = property_name
    parameters = scene.differentiable_parameters()

    
    if property_name == "CLOTHING":
        has_shirt = bool(
            getattr(scene, "parts", {}).get("shirt")
            or getattr(scene, "_part_shapes", {}).get("shirt")
        )
        if not has_shirt:
            raise ValueError("CLOTHING optimization requires a model part named 'shirt'")
        if hasattr(scene, "_shirt_color_override"):
            scene._shirt_color_override = tuple(
                parameters[3].detach().cpu().numpy().tolist()
            )

    initial_img = scene.render()
    if initial_img is None:
        raise RuntimeError("Initial render returned None")
    initial_result = classifier.classify(initial_img)
    score_key = "target_prob" if "target_prob" in initial_result else "human_prob"
    target_name = getattr(classifier, "target_name", "person")
    best_prob = initial_result[score_key]
    best_img = initial_img.copy()
    best_step = 0
    best_value = _get_property(scene, property_name)
    initial_snapshot_name = ""
    if snapshot_dir is not None:
        initial_snapshot_name = "step_00000.png"
        save_image(initial_img, str(Path(snapshot_dir) / initial_snapshot_name))
    if metrics_csv is not None:
        with Path(metrics_csv).open("a", newline="", encoding="utf-8") as csv_file:
            csv.writer(csv_file).writerow(
                (
                    0,
                    _format_property(property_name, best_value),
                    best_prob * 100.0,
                    initial_snapshot_name,
                )
            )

    if verbose:
        print(f"Initial {target_name}_prob = {best_prob * 100:.2f}%")
        print(f"{'Step':>4} {'Loss':>10} {target_name + 'Prob':>10} Parameters")
        print("-" * 88)

    if best_prob <= success_threshold:
        if verbose:
            print(f"Success at step 0: {target_name}_prob {best_prob * 100:.2f}%")
        return SearchResult(True, 0, best_prob, best_img, best_value)

    parameter_indices = {
        "POSITION": (0,),
        "ROTATION": (1,),
        "POSE": (0, 1),
        "LIGHTING": (2,),
        "CLOTHING": (3,),
    }[property_name]
    optimized_parameters = [parameters[index] for index in parameter_indices]
    for model_parameter in classifier.model.parameters():
        model_parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(optimized_parameters, lr=learning_rate)

    for step in range(1, epochs + 1):
        previous_values = [parameter.detach().clone() for parameter in optimized_parameters]
        optimizer.zero_grad(set_to_none=True)

        differentiable_image = scene.render_differentiable()
        loss = loss_function(
            differentiable_image,
            input_size=detector_input_size,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite detector loss at gradient step {step}")
        loss.backward()
        if any(
            parameter.grad is None or not torch.isfinite(parameter.grad).all()
            for parameter in optimized_parameters
        ):
            raise RuntimeError(
                f"{type(scene).__name__} did not produce a finite "
                f"{property_name.lower()} gradient at step {step}"
            )
        loss_value = loss.detach().item()
        # The next Mitsuba/Dr.Jit render needs CUDA memory from a separate
        # allocator. Drop Torch's references to this step's graph before it
        # starts, then return cached freed blocks so Dr.Jit can reuse them.
        del differentiable_image, loss
        if classifier.device.type == "cuda":
            torch.cuda.empty_cache()
        optimizer.step()

        with torch.no_grad():
            if property_name in ("POSITION", "POSE"):
                position_parameter = parameters[0]
                position_parameter[1] = torch.as_tensor(
                    initial_position[1],
                    device=position_parameter.device,
                    dtype=position_parameter.dtype,
                )
            if property_name in ("ROTATION", "POSE"):
                # Bound pitch and roll to keep the subject near upright.
                rotation_parameter = parameters[1]
                rotation_parameter[1:].clamp_(-45.0, 45.0)
            if property_name == "LIGHTING":
                parameters[2].clamp_(min=config.PROPERTY_BOUNDS["LIGHTING"][0])
            if property_name == "CLOTHING":
                parameters[3].clamp_(0.0, 1.0)

        if property_name in ("POSITION", "POSE", "ROTATION"):
            # Reject updates that move any part of the subject out of view or
            # intersect the environment. The renderers read these live leaves.
            with torch.no_grad():
                if not scene.is_position_valid(scene.get_position()):
                    for parameter, previous_value in zip(optimized_parameters, previous_values):
                        parameter.copy_(previous_value)
                    for state in optimizer.state.values():
                        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                            if key in state:
                                state[key].zero_()

        should_snapshot = snapshot_dir is not None and step % snapshot_every == 0
        if step % validate_every != 0 and step != epochs and not should_snapshot:
            continue

        image = scene.render()
        if image is None:
            continue
        if should_snapshot:
            save_image(image, str(Path(snapshot_dir) / f"step_{step:05d}.png"))
        result = classifier.classify(image)
        target_prob = result[score_key]
        current_value = _get_property(scene, property_name)
        if should_snapshot and metrics_csv is not None:
            snapshot_name = f"step_{step:05d}.png"
            with Path(metrics_csv).open("a", newline="", encoding="utf-8") as csv_file:
                csv.writer(csv_file).writerow(
                    (
                        step,
                        _format_property(property_name, current_value),
                        target_prob * 100.0,
                        snapshot_name,
                    )
                )
        if target_prob < best_prob:
            best_prob = target_prob
            best_step = step
            best_img = image.copy()
            best_value = current_value

        if verbose:
            print(
                f"{step:4d} {loss_value:10.4f} {target_prob * 100:9.2f}% "
                f"{_format_property(property_name, current_value)}"
            )
        if best_prob <= success_threshold:
            break

    _set_property(scene, property_name, best_value)
    return SearchResult(
        best_prob <= success_threshold,
        best_step,
        best_prob,
        best_img,
        best_value,
    )


def save_image(img: np.ndarray, path: str) -> None:
    Image.fromarray(img.astype(np.uint8)).save(path)
