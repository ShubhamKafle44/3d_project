from __future__ import annotations

import os
import random
import numpy as np
import torch
from PIL import Image

import config
from core.renderer import MeshScene3D


COCO_PERSON_IDX = 1


class HumanDetectionClassifier:
    categories = ["__background__", "person"]

    def __init__(self, model_name: str = "fasterrcnn_resnet50_fpn_v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = torch.device(device)

        if model_name == "fasterrcnn_resnet50_fpn_v2":
            from torchvision.models.detection import (
                fasterrcnn_resnet50_fpn_v2,
                FasterRCNN_ResNet50_FPN_V2_Weights,
            )
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.05)
        elif model_name == "retinanet_resnet50_fpn_v2":
            from torchvision.models.detection import (
                retinanet_resnet50_fpn_v2,
                RetinaNet_ResNet50_FPN_V2_Weights,
            )
            weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
            self.model = retinanet_resnet50_fpn_v2(weights=weights, box_score_thresh=0.05)
        else:
            raise ValueError(f"Unknown detector: {model_name}")

        self.model.to(self.device).eval()
        self.transform = weights.transforms()

    @torch.no_grad()
    def classify(self, img: torch.Tensor | np.ndarray) -> dict:
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)
        img = img.to(self.device).float()

        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]

        if img.max() > 1.5:
            img = img / 255.0

        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1)

        predictions = self.model([img])
        pred = predictions[0]
        labels = pred["labels"]
        scores = pred["scores"]

        person_mask = labels == COCO_PERSON_IDX
        person_scores = scores[person_mask]

        if len(person_scores) > 0:
            human_prob = person_scores.max().item()
            top_label = "person"
        else:
            human_prob = 0.0
            top_label = "none"

        return {
            "human_prob": human_prob,
            "num_persons": int(person_mask.sum().item()),
            "top_label": top_label,
            "all_scores": person_scores.cpu().tolist(),
        }

    def human_probability(self, img) -> float:
        return self.classify(img)["human_prob"]


TARGET_COVERAGE = (0.20, 0.45)
MIN_CAM_DISTANCE = 1.0
MAX_ZOOM_ITERS = 10


def subject_coverage(img_np: np.ndarray) -> float:
    bg_color = img_np[0, 0].astype(np.int16)
    diff = np.abs(img_np.astype(np.int16) - bg_color).sum(axis=-1)
    mask = diff > 20
    return float(mask.mean())


def auto_zoom_to_subject(scene: MeshScene3D) -> float:
    lo, hi = TARGET_COVERAGE
    coverage = None
    for _ in range(MAX_ZOOM_ITERS):
        img_np = scene.render()
        if img_np is None:
            break
        coverage = subject_coverage(img_np)
        if lo <= coverage <= hi:
            break
        target_mid = (lo + hi) / 2
        scale = 0.5 if coverage <= 0 else (coverage / target_mid) ** 0.5
        scale = max(0.3, min(scale, 3.0))
        new_distance = max(MIN_CAM_DISTANCE, scene.cam_distance * scale)
        scene.set_camera_orbit(
            new_distance, scene.cam_elev, scene.cam_azim,
            scene.cam_target, focal_length=scene.cam_focal,
        )
    return coverage if coverage is not None else 0.0


def build_scene(device):
    scene = MeshScene3D(device=device, image_size=config.IMAGE_SIZE)
    for part_name, part_path in config.HUMAN_PARTS.items():
        scene.load_mesh(part_path, name=part_name)
    if getattr(config, "BACKGROUND_3D_PATH", None):
        scene.load_background(config.BACKGROUND_3D_PATH)
    scene.set_part_color("shirt", (0.2, 0.4, 0.8))
    scene.set_part_color("pants", (0.15, 0.15, 0.15))
    scene.set_part_color("body", (0.9, 0.75, 0.65))
    return scene


POSITION_BOUNDS = (-1.5, 1.5)
LIGHT_BOUNDS = (0.1, 1.0)
COLOR_BOUNDS = (0.0, 1.0)
SUCCESS_THRESHOLD = 0.05


def set_property(scene, prop_name: str, value):
    if prop_name == "POSITION":
        if hasattr(scene, "set_position"):
            scene.set_position(value[0], value[1], value[2])
        else:
            with torch.no_grad():
                scene.pos.copy_(torch.as_tensor(value, device=scene.pos.device))
    elif prop_name == "LIGHTING":
        if hasattr(scene, "set_lighting"):
            scene.set_lighting(value)
        else:
            with torch.no_grad():
                scene.ambient_intensity.fill_(value)
    elif prop_name == "CLOTHING":
        if hasattr(scene, "set_part_color"):
            scene.set_part_color("shirt", value)


def get_property(scene, prop_name: str):
    if prop_name == "POSITION":
        return scene.pos.detach().cpu().numpy().copy()
    elif prop_name == "LIGHTING":
        return scene.ambient_intensity.detach().cpu().item()
    elif prop_name == "CLOTHING":
        return scene.shirt_color.detach().cpu().numpy().copy()
    return None


def random_perturbation(current, prop_name: str, step_size: float = 0.1):
    if prop_name == "POSITION":
        noise = np.random.normal(0, step_size, size=3)
        new = current + noise
        return np.clip(new, *POSITION_BOUNDS)
    elif prop_name == "LIGHTING":
        noise = random.gauss(0, step_size)
        new = current + noise
        return float(np.clip(new, *LIGHT_BOUNDS))
    elif prop_name == "CLOTHING":
        noise = np.random.normal(0, step_size, size=3)
        new = current + noise
        return np.clip(new, *COLOR_BOUNDS)
    return current


def format_params(scene, prop_name: str) -> str:
    if prop_name == "POSITION":
        x, y, z = scene.pos.detach().cpu().tolist()
        return f"pos=({x:+.3f}, {y:+.3f}, {z:+.3f})"
    if prop_name == "LIGHTING":
        val = scene.ambient_intensity.detach().cpu().item()
        return f"light=({val:+.3f})"
    if prop_name == "CLOTHING":
        r, g, b = scene.shirt_color.detach().cpu().tolist()
        return f"shirt=({r:+.3f}, {g:+.3f}, {b:+.3f})"
    return ""

def main():
    print("=" * 70)
    print("  3D Render Human Detection Test")
    print("  Uses COCO Faster R-CNN (real \"person\" class)")
    print("=" * 70)


    property_name = "POSITION"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scene = build_scene(device)
    classifier = HumanDetectionClassifier(
        model_name="fasterrcnn_resnet50_fpn_v2",
        device=device,
    )

    model_label, model_path = config.AVAILABLE_3D_MODELS[0]
    background_path = getattr(config, "BACKGROUND_PATH", getattr(config, "BACKGROUND_3D_PATH", "n/a"))

    print(f"Vision     : {classifier.model_name}")
    print(f"Device     : {device}\n" + "-" * 70)

    with torch.no_grad():
        initial_render = scene.render()

    if initial_render is None:
        print("ERROR: initial render returned None. Check mesh paths.")
        return

    Image.fromarray(initial_render.astype(np.uint8)).save("test_initial_render.png")
    print("Saved initial render -> test_initial_render.png")

    result = classifier.classify(initial_render)
    print("\nINITIAL RENDER DETECTION:")
    print(f"  Human probability : {result['human_prob'] * 100:.2f}%")
    print(f"  Persons detected  : {result['num_persons']}")
    print(f"  Top label         : {result['top_label']}")
    if result["all_scores"]:
        print(f"  All person scores : {[f'{s:.3f}' for s in result['all_scores']]}\n")

    # Search Execution
    N_EPOCHS = 100
    LR = 0.05

    print(f"Starting {N_EPOCHS}-epoch random search")
    print(f"Property   : {property_name}")
    print(f"Target     : HUMAN (COCO person detection score)")
    print(f"Step size  : {LR}")
    print(f"Success at : human prob <= {int(SUCCESS_THRESHOLD * 100)}%\n" + "=" * 70)
    print(f"{'Step':>4} {'HumanProb':>10}  {'Persons':>8}  {'Top Label':<12}Parameters\n" + "-" * 70)

    best_prob = result["human_prob"]
    best_img = initial_render.copy()
    best_step = 0
    best_params = get_property(scene, property_name)
    current_params = best_params.copy() if hasattr(best_params, "copy") else best_params

    for step in range(1, N_EPOCHS + 1):
        proposal = random_perturbation(current_params, property_name, step_size=LR)
        set_property(scene, property_name, proposal)

        with torch.no_grad():
            img_np = scene.render()

        if img_np is None:
            continue

        res = classifier.classify(img_np)
        human_prob = res["human_prob"]

        if human_prob < best_prob:
            best_prob = human_prob
            best_step = step
            best_img = img_np.copy()
            best_params = proposal.copy() if hasattr(proposal, "copy") else proposal
            current_params = best_params
        else:
            set_property(scene, property_name, current_params)

        params_str = format_params(scene, property_name)
        print(f"{step:4d} {human_prob * 100:9.2f}%  {res['num_persons']:8d}  "
              f"{res['top_label']:<12}{params_str}")

        if best_prob <= SUCCESS_THRESHOLD:
            print("-" * 70)
            print(f"Success at step {step}: human prob {best_prob * 100:.2f}% <= "
                  f"{int(SUCCESS_THRESHOLD * 100)}%")
            break
    else:
        print("-" * 70)
        print(f"Did not reach <= {int(SUCCESS_THRESHOLD * 100)}% in {N_EPOCHS} epochs. "
              f"Best: step {best_step}, human prob {best_prob * 100:.2f}%")

    if best_img is not None:
        out_path = "test_least_human_result.png"
        Image.fromarray(best_img.astype(np.uint8)).save(out_path)
        print(f"Saved lowest-human-prob render (step {best_step}) -> {out_path}")

    set_property(scene, property_name, best_params)
    with torch.no_grad():
        final_render = scene.render()

    if final_render is not None:
        final_res = classifier.classify(final_render)
        print("\nFINAL BEST RENDER DETECTION:")
        print(f"  Human probability : {final_res['human_prob'] * 100:.2f}%")
        print(f"  Persons detected  : {final_res['num_persons']}")
        print(f"  Top label         : {final_res['top_label']}")


if __name__ == "__main__":
    main()