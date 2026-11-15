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
                self.key_light_intensity = float(self.lighting_parameter.detach().cpu())
                self.light_position = self.light_position_parameter.detach().cpu().numpy().astype(np.float32)
                if self._active_differentiable_property == "TEXTURE":
                    self._texture_image_override = self.texture_image_parameter.detach().cpu().numpy().copy()
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
                self.key_light_intensity, dtype=torch.float32, requires_grad=True
            )
        if getattr(self, "light_position_parameter", None) is None:
            self.light_position_parameter = torch.tensor(
                self.light_position.copy(), dtype=torch.float32, requires_grad=True
            )
        if getattr(self, "rotation_parameter", None) is None:
            self.rotation_parameter = torch.tensor(
                self.rot_deg.copy(), dtype=torch.float32, requires_grad=True
            )
        texture_image = None
        if self._active_differentiable_property == "TEXTURE":
            if self.texture_attack_target is None:
                self.set_texture_attack_target()
            if self.texture_image_parameter is None:
                self.texture_image_parameter = torch.tensor(
                    self.load_texture_attack_image(), dtype=torch.float32, requires_grad=True
                )
            texture_image = self.texture_image_parameter
            self._texture_image_override = texture_image.detach().cpu().numpy().copy()
        image = _MitsubaTorchRender.apply(
            self, self.position_parameter, self.rotation_parameter,
            self.lighting_parameter, texture_image, self.light_position_parameter,
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
                self.key_light_intensity, dtype=torch.float32, requires_grad=True
            )
        if getattr(self, "light_position_parameter", None) is None:
            self.light_position_parameter = torch.tensor(
                self.light_position.copy(), dtype=torch.float32, requires_grad=True
            )
        texture_parameter = self.texture_color_parameter
        if self._active_differentiable_property == "TEXTURE":
            if self.texture_attack_target is None:
                self.set_texture_attack_target()
            if self.texture_image_parameter is None:
                self.texture_image_parameter = torch.tensor(
                    self.load_texture_attack_image(), dtype=torch.float32, requires_grad=True
                )
            texture_parameter = self.texture_image_parameter
        elif getattr(self, "texture_color_parameter", None) is None:
            color_target = (
                "shirt" if "shirt" in self._part_shapes
                else next(iter(self._part_shapes), None)
            )
            self.texture_color_parameter = torch.tensor(
                self.get_material_color(color_target).astype(np.float32),
                dtype=torch.float32,
                requires_grad=True,
            )
            texture_parameter = self.texture_color_parameter
        return (
            self.position_parameter,
            self.rotation_parameter,
            self.lighting_parameter,
            texture_parameter,
            self.light_position_parameter,
        )
