from __future__ import annotations
import numpy as np

import config
from renderer import DifferentiableScene, build_scene


def build_human_scene(backend: str, device: str) -> DifferentiableScene:
    scene = build_scene(backend, device=device, image_size=config.IMAGE_SIZE)

    for part_name, part_path in config.HUMAN_PARTS.items():
        scene.load_mesh(part_path, name=part_name)

    if config.BACKGROUND_PATH:
        scene.load_background(config.BACKGROUND_PATH)

    for part_name, color in config.DEFAULT_MATERIAL_COLORS.items():
        if part_name in config.HUMAN_PARTS:
            scene.set_material_color(part_name, color)

    scene.set_camera_orbit(**config.CAMERA)
    scene.set_lighting(config.LIGHT["intensity"])
    return scene


def subject_coverage(img_np: np.ndarray) -> float:
    """Rough fraction of the frame occupied by the subject, assuming a
    roughly uniform background in the top-left corner pixel."""
    bg_color = img_np[0, 0].astype(np.int16)
    diff = np.abs(img_np.astype(np.int16) - bg_color).sum(axis=-1)
    mask = diff > 20
    return float(mask.mean())


def auto_zoom_to_subject(scene: DifferentiableScene) -> float:
    lo, hi = config.SEARCH["target_coverage"]
    coverage = None
    for _ in range(config.SEARCH["max_zoom_iters"]):
        img_np = scene.render()
        if img_np is None:
            break
        coverage = subject_coverage(img_np)
        if lo <= coverage <= hi:
            break
        target_mid = (lo + hi) / 2
        scale = 0.5 if coverage <= 0 else (coverage / target_mid) ** 0.5
        scale = max(0.3, min(scale, 3.0))
        new_distance = max(config.SEARCH["min_cam_distance"], scene.cam_distance * scale)
        scene.set_camera_orbit(
            new_distance, scene.cam_elev, scene.cam_azim,
            scene.cam_target, fov=scene.cam_fov,
        )
    return coverage if coverage is not None else 0.0
