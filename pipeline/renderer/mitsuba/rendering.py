from __future__ import annotations

from typing import Optional

import mitsuba as mi
import numpy as np
import torch

import config
from .differentiable import _MitsubaTorchRender
from .utils import linear_to_srgb, srgb_output


class SceneRenderingMixin:
    # ---- render -------------------------------------------------------
    def render(self) -> Optional[np.ndarray]:
        """Render an 8-bit RGB image for display, returning None on failure."""
        if not self._part_paths:
            return None
        try:
            if getattr(self, "position_parameter", None) is not None:
                self.pos = self.position_parameter.detach().cpu().numpy().astype(np.float32)
                self.rot_deg = self.rotation_parameter.detach().cpu().numpy().astype(np.float32)
                self.ambient_intensity = float(self.lighting_parameter.detach().cpu())
                if self._active_differentiable_property == "CLOTHING":
                    self._shirt_color_override = tuple(
                        self.shirt_color_parameter.detach().cpu().numpy().tolist()
                    )
            scene_dict = self._build_scene_dict()
            scene = mi.load_dict(scene_dict)
            image = mi.render(scene, spp=self._spp)
            img_np = np.clip(np.array(image, copy=True)[..., :3], 0.0, 1.0)
            if srgb_output():
                img_np = linear_to_srgb(img_np)
            return (img_np * 255.0).astype(np.uint8)
        except Exception as exc:  # noqa: BLE001
            print(f"[mitsuba_renderer] render failed: {exc}")
            return None

    def render_differentiable(self):
        """Render CHW float RGB with a Torch autograd bridge to Dr.Jit."""
        if not self._part_paths:
            raise RuntimeError("load at least one mesh before rendering")
        if mi.variant() not in ("llvm_ad_rgb", "cuda_ad_rgb"):
            raise RuntimeError(
                f"Mitsuba differentiable rendering needs an AD variant, got {mi.variant()!r}. "
                "Set config.MITSUBA_VARIANT to 'llvm_ad_rgb' or 'cuda_ad_rgb' before importing Mitsuba."
            )

        if getattr(self, "position_parameter", None) is None:
            self.position_parameter = torch.tensor(
                self.pos.copy(), dtype=torch.float32, requires_grad=True
            )
        if getattr(self, "lighting_parameter", None) is None:
            self.lighting_parameter = torch.tensor(
                self.ambient_intensity, dtype=torch.float32, requires_grad=True
            )
        if getattr(self, "rotation_parameter", None) is None:
            self.rotation_parameter = torch.tensor(
                self.rot_deg.copy(), dtype=torch.float32, requires_grad=True
            )
        if getattr(self, "shirt_color_parameter", None) is None:
            self.shirt_color_parameter = torch.tensor(
                self.get_material_color("shirt").astype(np.float32),
                dtype=torch.float32,
                requires_grad=True,
            )
        shirt_color = None
        if self._active_differentiable_property == "CLOTHING":
            if not self._part_shapes.get("shirt"):
                raise ValueError("CLOTHING optimization requires a model part named 'shirt'")
            shirt_color = self.shirt_color_parameter
            self._shirt_color_override = tuple(
                shirt_color.detach().cpu().numpy().tolist()
            )
        image = _MitsubaTorchRender.apply(
            self, self.position_parameter, self.rotation_parameter,
            self.lighting_parameter, shirt_color,
        )
        return linear_to_srgb(image) if srgb_output() else image

    def differentiable_parameters(self):
        """Return optimizer-ready Torch leaves for position and light intensity."""
        if getattr(self, "position_parameter", None) is None:
            self.position_parameter = torch.tensor(
                self.pos.copy(), dtype=torch.float32, requires_grad=True
            )
        if getattr(self, "rotation_parameter", None) is None:
            self.rotation_parameter = torch.tensor(
                self.rot_deg.copy(), dtype=torch.float32, requires_grad=True
            )
        if getattr(self, "lighting_parameter", None) is None:
            self.lighting_parameter = torch.tensor(
                self.ambient_intensity, dtype=torch.float32, requires_grad=True
            )
        if getattr(self, "shirt_color_parameter", None) is None:
            self.shirt_color_parameter = torch.tensor(
                self.get_material_color("shirt").astype(np.float32),
                dtype=torch.float32,
                requires_grad=True,
            )
        return (
            self.position_parameter,
            self.rotation_parameter,
            self.lighting_parameter,
            self.shirt_color_parameter,
        )
