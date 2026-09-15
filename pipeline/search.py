from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np
import torch
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
        if property_name == "POSITION" and not scene.is_position_valid(proposal):
            if verbose:
                print(f"{step:4d} {'rejected':>10} {'collision':>8}  {'':<10} "
                      f"{_format_property(property_name, proposal)}")
            continue
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


def run_3d_appearance_attack(
    scene,
    classifier: "HumanDetectionClassifier",
    epochs: int = 200,
    learning_rate: float = 0.03,
    validate_every: int = 10,
    detector_input_size: int = 384,
    success_threshold: Optional[float] = None,
    verbose: bool = True,
) -> SearchResult:
    """Optimize a renderable shirt appearance through PyTorch3D.

    The optimized values are mesh vertex colors, not image pixels.  Gradients
    flow from Faster R-CNN's pre-NMS RPN logits through the rendered shirt.
    """
    if not hasattr(scene, "render_differentiable") or not hasattr(scene, "set_vertex_colors"):
        raise TypeError("3D appearance attack requires the PyTorch3D renderer")
    success_threshold = success_threshold if success_threshold is not None else config.SEARCH["success_threshold"]
    # The detector supplies gradients to the rendered image but is never
    # trained; freezing it substantially reduces CUDA memory use.
    for parameter in classifier.model.parameters():
        parameter.requires_grad_(False)
    shirt = scene.parts["shirt"]
    vertex_count = shirt.verts_packed().shape[0]
    base_color = scene.get_material_color("shirt")
    base_logits = torch.logit(
        torch.tensor(base_color, device=scene.device, dtype=torch.float32).clamp(0.01, 0.99)
    )
    color_logits = torch.nn.Parameter(base_logits.expand(vertex_count, 3).clone())
    optimizer = torch.optim.Adam([color_logits], lr=learning_rate)

    initial_img = scene.render()
    if initial_img is None:
        raise RuntimeError("Initial render returned None")
    initial_result = classifier.classify(initial_img)
    best_prob, best_step, best_img = initial_result["human_prob"], 0, initial_img.copy()
    best_colors = torch.sigmoid(color_logits).detach().clone()

    if verbose:
        print(f"Initial human_prob = {best_prob * 100:.2f}%")
        print(f"{'Step':>4} {'Surrogate':>10} {'HumanProb':>10} {'Persons':>8}")
        print("-" * 46)

    for step in range(1, epochs + 1):
        colors = torch.sigmoid(color_logits)
        scene.set_vertex_colors("shirt", colors)
        rendered = scene.render_differentiable()
        loss = classifier.rpn_objectness_loss(
            rendered, input_size=detector_input_size
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % validate_every != 0 and step != epochs:
            continue
        with torch.no_grad():
            scene.set_vertex_colors("shirt", torch.sigmoid(color_logits))
            image = scene.render_differentiable()
            image_np = (image.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        result = classifier.classify(image_np)
        human_prob = result["human_prob"]
        if human_prob < best_prob:
            best_prob, best_step, best_img = human_prob, step, image_np.copy()
            best_colors = torch.sigmoid(color_logits).detach().clone()
        if verbose:
            print(f"{step:4d} {loss.item():10.4f} {human_prob * 100:9.2f}% {result['num_persons']:8d}")
        if best_prob <= success_threshold:
            break

    scene.set_vertex_colors("shirt", best_colors)
    return SearchResult(
        best_prob <= success_threshold, best_step, best_prob, best_img, best_colors.cpu().numpy()
    )


def save_image(img: np.ndarray, path: str) -> None:
    Image.fromarray(img.astype(np.uint8)).save(path)
