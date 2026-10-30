from __future__ import annotations

from typing import Optional

import numpy as np
import torch

import config
from .dependencies import (
    FoVPerspectiveCameras, MeshRenderer, MeshRasterizer, Meshes, PointLights,
    TexturesVertex, join_meshes_as_scene, look_at_view_transform,
)
from .shaders import AlphaCutoutPhongShader, InverseSquarePointLights


class SceneRenderingMixin:
    # ---- standard render (numpy) --------------------------------------
    def render(self) -> Optional[np.ndarray]:
        if not self.parts:
            return None
        try:
            img_tensor = self.render_differentiable()
            img_hwc = img_tensor.permute(1, 2, 0).clamp(0, 1)
            img_np = (img_hwc.detach().cpu().numpy() * 255.0).astype(np.uint8)
            return img_np
        except Exception as exc:
            print(f"[pytorch3d_renderer] render failed: {exc}")
            return None

    def render_differentiable(self) -> torch.Tensor:

        # ============================================================
        # 1. Assemble HUMAN ONLY for debugging
        # ============================================================
        R_obj = self._rotation_matrix()

        meshes = []

        for name, mesh in self.parts.items():

            verts = mesh.verts_padded()[0]

            # Apply object transformation
            verts = torch.matmul(verts, R_obj.T) + self.pos

            m = mesh.clone()
            m = m.update_padded(verts.unsqueeze(0))
            if name == "shirt" and self._active_differentiable_property == "CLOTHING":
                if self.shirt_color_parameter is None:
                    raise RuntimeError("Initialize differentiable parameters before clothing rendering")
                vertex_count = m.verts_packed().shape[0]
                colors = self.shirt_color_parameter.view(1, 1, 3).expand(1, vertex_count, 3)
                m.textures = TexturesVertex(verts_features=colors)

            meshes.append(m)

        if len(meshes) == 1:
            scene_mesh = meshes[0]
        else:
            scene_mesh = join_meshes_as_scene(meshes)

        # Use the configured scene camera.  The old code recalculated both
        # target and distance from the human every frame, which guaranteed a
        # close-up and excluded the environment from composition.
        R, T = look_at_view_transform(
            dist=self._cam_distance,
            elev=self._cam_elev,
            azim=self._cam_azim,
            at=(self._cam_target,),
            device=self.device,
        )

        cameras = FoVPerspectiveCameras(
            device=self.device,
            R=R,
            T=T,
            fov=self._cam_fov,
        )

        # ============================================================
        # 6. Lighting
        # ============================================================
        # Match Mitsuba's point-emitter radiometry.  Its diffuse BSDF applies
        # a 1/pi term, so it belongs in this shader's light colour. Ambient
        # illumination is independent of both point-light intensities.
        ambient_intensity = float(config.AMBIENT_LIGHT["intensity"])
        ambient_rgb = tuple(
            float(channel) * ambient_intensity
            for channel in config.AMBIENT_LIGHT["color"]
        )
        street_lights = InverseSquarePointLights(
            device=self.device,
            location=self.light_position.unsqueeze(0),
            ambient_color=(ambient_rgb,),
            diffuse_color=(
                self.light_color
                * self.ambient_intensity
                * (20.0 / np.pi)
            ).unsqueeze(0),
            # Mitsuba uses a purely diffuse BSDF for this scene.
            specular_color=((0.0, 0.0, 0.0),),
        )
        fill_lights = InverseSquarePointLights(
            device=self.device,
            location=self.fill_light_position.unsqueeze(0),
            ambient_color=((0.0, 0.0, 0.0),),
            diffuse_color=(
                self.fill_light_color
                * config.FILL_LIGHT["intensity"]
                * (20.0 / np.pi)
            ).unsqueeze(0),
            specular_color=((0.0, 0.0, 0.0),),
        )

        # ============================================================
        # 7. Rasterizer
        # ============================================================
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=self._raster_settings,
        )

        # ============================================================
        # 8. Renderer
        # ============================================================
        def make_renderer(lights: PointLights) -> MeshRenderer:
            return MeshRenderer(
            rasterizer=rasterizer,
            shader=AlphaCutoutPhongShader(
                device=self.device,
                cameras=cameras,
                lights=lights,
            ),
            )

        street_renderer = make_renderer(street_lights)
        fill_renderer = make_renderer(fill_lights)

        def render_with_fixture_lights(mesh: Meshes) -> torch.Tensor:
            """Sum two direct point-light passes; neither pass has ambient light."""
            street = street_renderer(mesh)
            fill = fill_renderer(mesh)
            return torch.cat(
                (street[..., :3] + fill[..., :3], torch.maximum(street[..., 3:4], fill[..., 3:4])),
                dim=-1,
            )

        # ============================================================
        # 9. Render
        # ============================================================
        foreground = render_with_fixture_lights(scene_mesh)
        if self.background is not None and self._background_is_textured:
            background = render_with_fixture_lights(self.background)
            # SoftPhong's alpha can be fractional at a face even with no
            # intended transparency.  Use a hard rasterized silhouette for
            # compositing so the road texture cannot show through the human.
            alpha = (foreground[..., 3:4] > 0).to(foreground.dtype)
            images = torch.cat(
                (foreground[..., :3] * alpha + background[..., :3] * (1 - alpha), alpha),
                dim=-1,
            )
        else:
            images = foreground

        img = images[0, ..., :3].clamp(0.0, 1.0)

        return img.permute(2, 0, 1)

    def differentiable_parameters(self):
        """Return the optimizer leaves shared by differentiable searches."""
        if self.shirt_color_parameter is None:
            if "shirt" not in self.parts:
                color = torch.full((3,), 0.8, device=self.device, dtype=torch.float32)
            else:
                color = torch.as_tensor(
                    self.get_material_color("shirt"), device=self.device, dtype=torch.float32
                )
            self.shirt_color_parameter = torch.nn.Parameter(color.detach().clone())
        return self.pos, self.rot_deg, self.ambient_intensity, self.shirt_color_parameter
