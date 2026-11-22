from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

import config
from renderer import DifferentiableScene
from .artifacts import (
    append_metrics_row, append_optimization_record, create_optimization_workbook,
    initialize_metrics_file, save_image, save_loss_plot,
)
from .constraints import enforce_scene_constraints
from .optimization import apply_parameter_bounds, take_gradient_step
from .properties import format_property, get_property, set_property
from .result import SearchResult

if TYPE_CHECKING:
    from detector import ObjectDetectionClassifier


PROPERTIES = ("POSITION", "ROTATION", "POSE", "LIGHTING", "TEXTURE")


def _prepare_output(snapshot_dir: Optional[str], metrics_csv: Optional[str]) -> None:
    if snapshot_dir is not None:
        Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
    initialize_metrics_file(metrics_csv)


def _save_initial_observation(
    image, property_name: str, value, probability: float,
    snapshot_dir: Optional[str], metrics_csv: Optional[str], snapshot_prefix: str,
    highest_class: str = "", highest_class_probability: Optional[float] = None,
) -> str:
    snapshot_name = ""
    if snapshot_dir is not None:
        snapshot_name = f"{snapshot_prefix}_step_00000.png" if snapshot_prefix else "step_00000.png"
        save_image(image, str(Path(snapshot_dir) / snapshot_name))
    append_metrics_row(
        metrics_csv, 0, format_property(property_name, value), probability,
        snapshot_name, highest_class, highest_class_probability,
    )
    return snapshot_name


def _initial_result(best_prob, best_img, best_value, verbose, target_name, success_threshold):
    if best_prob <= success_threshold:
        if verbose:
            print(f"Success at step 0: {target_name}_prob {best_prob * 100:.2f}%")
        return SearchResult(True, 0, best_prob, best_img, best_value)
    return None


def run_gradient_descent(
    scene: DifferentiableScene,
    classifier: "ObjectDetectionClassifier",
    property_name: str,
    epochs: int = 100,
    learning_rate: float = 0.06,
    validate_every: int = 5,
    success_threshold: Optional[float] = None,
    verbose: bool = True,
    snapshot_dir: Optional[str] = None,
    snapshot_every: int = 10,
    metrics_csv: Optional[str] = None,
    snapshot_prefix: str = "",
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

    _prepare_output(snapshot_dir, metrics_csv)
    optimization_workbook = create_optimization_workbook(metrics_csv)
    if success_threshold is None:
        success_threshold = config.SEARCH["success_threshold"]

    initial_position = scene.get_position().astype(np.float32)
    if property_name in ("POSITION", "POSE") and not scene.is_position_valid(initial_position):
        raise RuntimeError("Initial position is outside the frame or intersects scene geometry")

    scene._active_differentiable_property = property_name
    if property_name == "TEXTURE" and hasattr(scene, "set_texture_attack_target"):
        scene.set_texture_attack_target(getattr(scene, "texture_attack_target", None))
    parameters = scene.differentiable_parameters()
    texture_reference = (
        parameters[3].detach().clone() if property_name == "TEXTURE" else None
    )
    if property_name == "TEXTURE":
        if not hasattr(scene, "set_texture_attack_target"):
            has_target = bool(getattr(scene, "parts", {}).get("shirt"))
            if not has_target:
                raise ValueError("TEXTURE optimization requires a supported texture target")

    initial_img = scene.render()
    if initial_img is None:
        raise RuntimeError("Initial render returned None")
    initial_result = classifier.classify(initial_img)
    score_key = "target_prob" if "target_prob" in initial_result else "human_prob"
    target_name = getattr(classifier, "target_name", "person")
    best_prob = initial_result[score_key]
    detections = initial_result.get("detections", [])
    top_detection = max(
        detections, key=lambda detection: detection.get("confidence", 0.0),
        default=None,
    )
    best_img = initial_img.copy()
    best_step = 0
    best_value = get_property(scene, property_name)
    best_light_position = (
        parameters[4].detach().clone() if property_name == "LIGHTING" else None
    )
    append_optimization_record(
        optimization_workbook, step=0, property_name=property_name,
        loss=None, validation_probability=best_prob, position=scene.get_position(),
        parameters=parameters, gradients=[], optimizer=None,
        highest_class=(top_detection["label"] if top_detection else "none"),
        highest_class_probability=(top_detection["confidence"] if top_detection else 0.0),
    )
    _save_initial_observation(
        initial_img, property_name, best_value, best_prob,
        snapshot_dir, metrics_csv, snapshot_prefix,
        top_detection["label"] if top_detection else "none",
        top_detection["confidence"] if top_detection else 0.0,
    )

    if verbose:
        initial_line = f"Initial {target_name}_prob = {best_prob * 100:.2f}%"
        if top_detection is not None:
            initial_line += (
                f"; highest class = {top_detection['label']} "
                f"({top_detection['confidence'] * 100:.2f}%)"
            )
        if property_name == "LIGHTING":
            light_position = parameters[4].detach().cpu().numpy()
            initial_line += (
                " lighting=" + format_property(property_name, best_value)
                + " light_position=("
                + f"{light_position[0]:+.3f}, {light_position[1]:+.3f}, "
                + f"{light_position[2]:+.3f})"
            )
        print(initial_line)
    result = _initial_result(
        best_prob, best_img, best_value, verbose, target_name, success_threshold
    )
    if result is not None:
        return result

    parameter_indices = {
        "POSITION": (0,),
        "ROTATION": (1,),
        "POSE": (0, 1),
        "LIGHTING": (2, 4),
        "TEXTURE": (3,),
    }[property_name]
    optimized_parameters = [parameters[index] for index in parameter_indices]
    for model_parameter in classifier.model.parameters():
        model_parameter.requires_grad_(False)

    # Keep the lighting step size bounded; texture uses the configured rate
    # directly, like the other search properties.
    if property_name == "LIGHTING":
        optimizer_learning_rate = min(learning_rate, 0.05)
    else:
        optimizer_learning_rate = learning_rate

    if verbose:
        print(f"Optimizer learning rate: {optimizer_learning_rate:g}")
        print(
            f"{'Step':>4} {'Loss':>10} {target_name + 'Prob':>10} "
            f"{'Highest class':>18} {'Class prob':>11} Parameters"
        )
        print("-" * 116)

    optimizer = torch.optim.Adam(optimized_parameters, lr=optimizer_learning_rate)
    exploration_rng = np.random.default_rng()
    loss_history = []
    loss_plot_path = (
        Path(metrics_csv).parent / "loss_curve.png"
        if metrics_csv is not None
        else Path(snapshot_dir).parent / "loss_curve.png"
        if snapshot_dir is not None
        else None
    )

    for step in range(1, epochs + 1):
        previous_values = [parameter.detach().clone() for parameter in optimized_parameters]
        loss_value = take_gradient_step(
            scene, classifier, loss_function, optimized_parameters,
            optimizer, step, property_name,
            regularization_reference=texture_reference,
        )
        loss_history.append(loss_value)
        if loss_plot_path is not None:
            save_loss_plot(loss_history, str(loss_plot_path))
        apply_parameter_bounds(property_name, parameters, initial_position)
        enforce_scene_constraints(
            scene, property_name, parameters, optimized_parameters,
            previous_values, optimizer, exploration_rng,
        )

        should_snapshot = snapshot_dir is not None and step % snapshot_every == 0
        should_validate = step % validate_every == 0 or step == epochs or should_snapshot
        if not should_validate:
            append_optimization_record(
                optimization_workbook, step=step, property_name=property_name,
                loss=loss_value, validation_probability=None,
                position=scene.get_position(), parameters=parameters,
                gradients=[parameter.grad for parameter in optimized_parameters],
                optimizer=optimizer,
            )
            continue

        image = scene.render()
        if image is None:
            append_optimization_record(
                optimization_workbook, step=step, property_name=property_name,
                loss=loss_value, validation_probability=None,
                position=scene.get_position(), parameters=parameters,
                gradients=[parameter.grad for parameter in optimized_parameters],
                optimizer=optimizer,
            )
            continue
        snapshot_name = ""
        if should_snapshot:
            step_name = f"step_{step:05d}.png"
            snapshot_name = f"{snapshot_prefix}_{step_name}" if snapshot_prefix else step_name
            save_image(image, str(Path(snapshot_dir) / snapshot_name))
        result = classifier.classify(image)
        target_prob = result[score_key]
        detections = result.get("detections", [])
        top_detection = max(
            detections, key=lambda detection: detection.get("confidence", 0.0),
            default=None,
        )
        current_value = get_property(scene, property_name)
        append_optimization_record(
            optimization_workbook, step=step, property_name=property_name,
            loss=loss_value, validation_probability=target_prob,
            position=scene.get_position(), parameters=parameters,
            gradients=[parameter.grad for parameter in optimized_parameters],
            optimizer=optimizer, snapshot_name=snapshot_name,
            highest_class=(top_detection["label"] if top_detection else "none"),
            highest_class_probability=(top_detection["confidence"] if top_detection else 0.0),
        )
        if should_snapshot:
            append_metrics_row(
                metrics_csv, step, format_property(property_name, current_value),
                target_prob, snapshot_name,
                top_detection["label"] if top_detection else "none",
                top_detection["confidence"] if top_detection else 0.0,
            )
        if target_prob < best_prob:
            best_prob = target_prob
            best_step = step
            best_img = image.copy()
            best_value = current_value
            if property_name == "LIGHTING":
                best_light_position = parameters[4].detach().clone()

        if verbose:
            parameter_text = format_property(property_name, current_value)
            if property_name == "LIGHTING":
                light_position = parameters[4].detach().cpu().numpy()
                parameter_text += (
                    " light_position=("
                    f"{light_position[0]:+.3f}, {light_position[1]:+.3f}, "
                    f"{light_position[2]:+.3f})"
                )
            highest_class = top_detection["label"] if top_detection else "none"
            highest_prob = top_detection["confidence"] if top_detection else 0.0
            print(
                f"{step:4d} {loss_value:10.4f} {target_prob * 100:9.2f}% "
                f"{highest_class:>18} {highest_prob * 100:10.2f}% "
                f"{parameter_text}"
            )
        if best_prob <= success_threshold:
            break

    set_property(scene, property_name, best_value)
    if property_name == "LIGHTING":
        with torch.no_grad():
            parameters[4].copy_(best_light_position)
        scene.light_position = best_light_position.detach().cpu().numpy().astype(np.float32)
    return SearchResult(
        best_prob <= success_threshold, best_step, best_prob, best_img, best_value
    )
