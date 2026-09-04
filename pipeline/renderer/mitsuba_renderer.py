from __future__ import annotations
from typing import Dict, Optional, Tuple
import os

import numpy as np

from .base import DifferentiableScene

try:
    import mitsuba as mi

    _MI_VARIANT = os.environ.get("MITSUBA_VARIANT", "scalar_rgb")
    mi.set_variant(_MI_VARIANT)
    _MITSUBA_AVAILABLE = True
except ImportError:
    _MITSUBA_AVAILABLE = False


class MitsubaScene(DifferentiableScene):
    def __init__(self, device: str = "cpu", image_size: int = 512):
        if not _MITSUBA_AVAILABLE:
            raise ImportError("mitsuba is not installed. Run: pip install mitsuba")
        self.image_size = image_size

        self._part_paths: Dict[str, str] = {}
        self._part_colors: Dict[str, Tuple[float, float, float]] = {}
        self._background_path: Optional[str] = None

        self.pos = np.zeros(3, dtype=np.float32)
        self.rot_deg = np.zeros(3, dtype=np.float32)  # yaw, pitch, roll
        self.ambient_intensity = 1.0

        self._cam_distance = 3.0
        self._cam_elev = 10.0
        self._cam_azim = 0.0
        self._cam_target = (0.0, 0.0, 0.0)
        self._cam_fov = 40.0

        self._scene = None  

    # ---- asset loading ---------------------------------------------
    def load_mesh(self, path: str, name: str = "mesh") -> None:
        self._part_paths[name] = path
        self._part_colors.setdefault(name, (0.8, 0.8, 0.8))

    def load_background(self, path: str) -> None:
        self._background_path = path

    # ---- position / rotation ----------------------------------------
    def set_position(self, x: float, y: float, z: float) -> None:
        self.pos = np.array([x, y, z], dtype=np.float32)

    def get_position(self) -> np.ndarray:
        return self.pos.copy()

    def set_rotation_deg(self, yaw: float, pitch: float = 0.0, roll: float = 0.0) -> None:
        self.rot_deg = np.array([yaw, pitch, roll], dtype=np.float32)

    def get_rotation_deg(self) -> np.ndarray:
        return self.rot_deg.copy()

    # ---- lighting -----------------------------------------------------
    def set_lighting(self, intensity: float) -> None:
        self.ambient_intensity = float(intensity)

    def get_lighting(self) -> float:
        return self.ambient_intensity

    # ---- material -------------------------------------------------------
    def set_material_color(self, part_name: str, rgb: Tuple[float, float, float]) -> None:
        self._part_colors[part_name] = tuple(float(c) for c in rgb)

    def get_material_color(self, part_name: str) -> np.ndarray:
        return np.array(self._part_colors.get(part_name, (0.8, 0.8, 0.8)))

    # ---- camera ----------------------------------------------------------
    def set_camera_orbit(
        self,
        distance: float,
        elevation: float,
        azimuth: float,
        target: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        fov: float = 40.0,
    ) -> None:
        self._cam_distance = distance
        self._cam_elev = elevation
        self._cam_azim = azimuth
        self._cam_target = target
        self._cam_fov = fov

    @property
    def cam_distance(self) -> float:
        return self._cam_distance

    @property
    def cam_elev(self) -> float:
        return self._cam_elev

    @property
    def cam_azim(self) -> float:
        return self._cam_azim

    @property
    def cam_target(self) -> Tuple[float, float, float]:
        return self._cam_target

    @property
    def cam_fov(self) -> float:
        return self._cam_fov

    # ---- scene assembly -------------------------------------------------
    def _camera_origin(self) -> Tuple[float, float, float]:
        elev_rad = np.radians(self._cam_elev)
        azim_rad = np.radians(self._cam_azim)
        d = self._cam_distance
        tx, ty, tz = self._cam_target
        x = tx + d * np.cos(elev_rad) * np.sin(azim_rad)
        y = ty + d * np.sin(elev_rad)
        z = tz + d * np.cos(elev_rad) * np.cos(azim_rad)
        return float(x), float(y), float(z)

    def _build_scene_dict(self) -> dict:
        origin = self._camera_origin()

        scene_dict = {
            "type": "scene",
            "integrator": {"type": "path", "max_depth": 6},
            "sensor": {
                "type": "perspective",
                "fov": self._cam_fov,
                "to_world": mi.ScalarTransform4f().look_at(
                    origin=origin, target=self._cam_target, up=(0, 1, 0)
                ),
                "film": {
                    "type": "hdrfilm",
                    "width": self.image_size,
                    "height": self.image_size,
                    "pixel_format": "rgb",
                },
                "sampler": {"type": "independent", "sample_count": 32},
            },
            "light": {
                "type": "point",
                "position": [2.0, 2.0, 2.0],
                "intensity": {"type": "spectrum", "value": self.ambient_intensity * 20.0},
            },
        }

        to_world = (
            mi.ScalarTransform4f()
            .translate(self.pos.tolist())
            .rotate([0, 0, 1], self.rot_deg[0])
            .rotate([1, 0, 0], self.rot_deg[1])
            .rotate([0, 1, 0], self.rot_deg[2])
        )

        for name, path in self._part_paths.items():
            r, g, b = self._part_colors.get(name, (0.8, 0.8, 0.8))
            scene_dict[f"part_{name}"] = {
                "type": "obj",
                "filename": path,
                "to_world": to_world,
                "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [r, g, b]}},
            }

        if self._background_path is not None:
            scene_dict["background"] = {
                "type": "obj",
                "filename": self._background_path,
                "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.5, 0.5, 0.5]}},
            }
        else:
            scene_dict["floor"] = {
                "type": "rectangle",
                "to_world": mi.ScalarTransform4f().translate([0, -1, 0])
                .rotate([1, 0, 0], -90)
                .scale(10),
                "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.6, 0.6, 0.6]}},
            }

        return scene_dict

    # ---- render -------------------------------------------------------
    def render(self) -> Optional[np.ndarray]:
        if not self._part_paths:
            return None
        try:
            scene_dict = self._build_scene_dict()
            scene = mi.load_dict(scene_dict)
            image = mi.render(scene, spp=32)
            img_np = np.array(mi.util.convert_to_bitmap(image))
            if img_np.shape[-1] == 4:
                img_np = img_np[..., :3]
            return img_np.astype(np.uint8)
        except Exception as exc:  # noqa: BLE001
            print(f"[mitsuba_renderer] render failed: {exc}")
            return None
