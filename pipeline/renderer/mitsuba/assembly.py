from __future__ import annotations

import copy
import math
from typing import Tuple

import mitsuba as mi
import numpy as np

import config


class SceneAssemblyMixin:
    # ---- scene assembly -------------------------------------------------
    def _camera_origin(self) -> Tuple[float, float, float]:
        """Convert the orbit camera settings into a world-space position."""
        elev_rad = np.radians(self._cam_elev)
        azim_rad = np.radians(self._cam_azim)
        d = self._cam_distance
        tx, ty, tz = self._cam_target
        x = tx + d * np.cos(elev_rad) * np.sin(azim_rad)
        y = ty + d * np.sin(elev_rad)
        z = tz + d * np.cos(elev_rad) * np.cos(azim_rad)
        return float(x), float(y), float(z)

    @staticmethod
    def _integrator_dict(differentiable: bool) -> dict:
        """`direct` takes one bounce and cannot pass rays through a `mask` BSDF, so the
        transparent parts of leaf cards render black. `path` (as in render_background.py)
        follows the null transmission. `prb` is the path-style integrator with a proper
        backward pass for the Torch bridge."""
        depth = int(getattr(config, "MITSUBA_MAX_DEPTH", 6))
        return {"type": "prb" if differentiable else "path", "max_depth": depth}

    def _build_scene_dict(self, differentiable: bool = False) -> dict:
        """Assemble Mitsuba's scene description from camera, lights, and meshes."""
        origin = self._camera_origin()

        scene_dict = {
            "type": "scene",
            "integrator": self._integrator_dict(differentiable),
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
                    "rfilter": {"type": "gaussian"},
                },
                "sampler": {"type": "independent", "sample_count": self._spp},
            },
            "environment": {
                "type": "constant",
                "radiance": {
                    "type": "rgb",
                    "value": [
                        math.pi * float(config.AMBIENT_LIGHT["intensity"]) * float(component)
                        for component in config.AMBIENT_LIGHT["color"]
                    ],
                },
            },
            "street_light": {
                "type": "point",
                "position": list(config.LIGHT["position"]),
                "intensity": {
                    "type": "rgb",
                    "value": [
                        20.0 * self.ambient_intensity * float(channel)
                        for channel in self.light_color
                    ],
                },
            },
            "fill_light": {
                "type": "point",
                "position": list(config.FILL_LIGHT["position"]),
                "intensity": {
                    "type": "rgb",
                    "value": [
                        20.0 * float(config.FILL_LIGHT["intensity"]) * float(channel)
                        for channel in config.FILL_LIGHT["color"]
                    ],
                },
            },
        }

        to_world = (
            mi.ScalarTransform4f()
            .translate(self.pos.tolist())
            .rotate([0, 0, 1], self.rot_deg[0])
            .rotate([1, 0, 0], self.rot_deg[1])
            .rotate([0, 1, 0], self.rot_deg[2])
        )

        for name, shapes in self._part_shapes.items():
            r, g, b = self._part_colors.get(name, (0.8, 0.8, 0.8))
            for key, mesh_path, bsdf in shapes:
                if name == "shirt" and self._shirt_color_override is not None:
                    bsdf = {
                        "type": "diffuse",
                        "reflectance": {
                            "type": "rgb",
                            "value": list(self._shirt_color_override),
                        },
                    }
                scene_dict[key] = {
                    "type": "obj",
                    "filename": mesh_path,
                    "to_world": to_world,
                    # untextured faces fall back to the part color
                    "bsdf": copy.deepcopy(bsdf) if bsdf is not None
                    else {"type": "diffuse", "reflectance": {"type": "rgb", "value": [r, g, b]}},
                }

        if self._background_shapes:
            scene_dict.update(copy.deepcopy(self._background_shapes))
        else:
            scene_dict["floor"] = {
                "type": "rectangle",
                "to_world": mi.ScalarTransform4f().translate([0, -1, 0])
                .rotate([1, 0, 0], -90)
                .scale(10),
                "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.6, 0.6, 0.6]}},
            }

        return scene_dict

