from __future__ import annotations

import logging

import numpy as np
import torch
import pytorch3d.transforms as T
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    Materials,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    look_at_view_transform,
)

from config import CAMERA_AZIM, CAMERA_DIST, CAMERA_ELEV, IMAGE_SIZE, OBJ_PATH
from core.background import load_background
from core.mesh_loader import load_mesh_from_file

logger = logging.getLogger(__name__)


class MeshScene:
    """Wraps a PyTorch3D renderer and exposes differentiable scene parameters."""

    def __init__(self, meshes, device: torch.device):
        self.device = device
        self.mesh   = meshes

        self.renderer, self.cameras = self._build_renderer()

        # Differentiable scene parameters (initialised to defaults)
        self.pos               = torch.zeros(3, device=device, requires_grad=True)
        self.rot               = torch.zeros(3, device=device, requires_grad=True)
        self.light_pos         = torch.tensor([[0.0, 1.0, -2.0]], device=device, requires_grad=True)
        self.ambient_intensity = torch.tensor([0.5], device=device, requires_grad=True)

        if OBJ_PATH:
            self.load_mesh(OBJ_PATH)

    # ── Renderer ──────────────────────────────────────────────────────────────
    def _build_renderer(self):
        R, T_cam = look_at_view_transform(dist=CAMERA_DIST, elev=CAMERA_ELEV, azim=CAMERA_AZIM)
        cameras  = FoVPerspectiveCameras(device=self.device, R=R, T=T_cam)

        raster_settings = RasterizationSettings(
            image_size=IMAGE_SIZE,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        lights = PointLights(
            device=self.device,
            location=[[0.0, 2.0, -2.0]],
            ambient_color=[[0.5, 0.5, 0.5]],
            diffuse_color=[[0.8, 0.8, 0.8]],
            specular_color=[[0.3, 0.3, 0.3]],
        )
        materials = Materials(device=self.device, shininess=64.0)

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(
                device=self.device,
                cameras=cameras,
                lights=lights,
                materials=materials,
            ),
        )
        return renderer, cameras

    # ── Mesh loading ──────────────────────────────────────────────────────────
    def load_mesh(self, path: str) -> None:
        """Load a new OBJ and reset all scene parameters to defaults.

        Resetting pos/rot is essential: the previous model's adversarially
        optimised position would put the new model completely off-screen.
        """
        self.mesh = load_mesh_from_file(path, self.device)
        logger.info("Mesh loaded: %s", path)

        # FIX: reset pose/lighting so the new model appears centred
        with torch.no_grad():
            self.pos.zero_()
            self.rot.zero_()
            self.light_pos.copy_(
                torch.tensor([[0.0, 1.0, -2.0]], device=self.device)
            )
            self.ambient_intensity.fill_(0.5)

    # ── Lighting ──────────────────────────────────────────────────────────────
    def _make_lights(self) -> PointLights:
        ambient_color = self.ambient_intensity.expand(3).unsqueeze(0)
        return PointLights(
            device=self.device,
            location=self.light_pos,
            ambient_color=ambient_color,
        )

    # ── Blank fallback ────────────────────────────────────────────────────────
    def _blank(self):
        blank_np = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
        blank_t  = torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE))
        return blank_np, blank_t

    # ── UI render (no gradients) ──────────────────────────────────────────────
    def render(self):
        if self.mesh is None:
            return self._blank()

        bg = load_background()

        with torch.no_grad():
            angles = self.rot.unsqueeze(0)
            R      = T.euler_angles_to_matrix(angles, convention="XYZ")[0]
            verts  = self.mesh.verts_packed()
            verts  = verts @ R.T + self.pos
            mesh   = self.mesh.update_padded(verts.unsqueeze(0))
            images = self.renderer(mesh, lights=self._make_lights())

        rgba  = images[0].cpu().numpy()
        rgb   = rgba[..., :3]
        alpha = rgba[..., 3:4]
        mask  = (alpha > 0.0).astype(np.float32)

        img_np = rgb * mask + bg * (1 - mask)
        img_np = np.clip(img_np, 0, 1)
        img_t  = torch.from_numpy(img_np).permute(2, 0, 1).float()
        return img_np, img_t

    # ── Differentiable render (for adversarial optimisation) ──────────────────
    def render_differentiable(self):
        if self.mesh is None:
            return torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), device=self.device)

        bg_np = load_background()
        bg_t  = torch.from_numpy(bg_np).to(self.device).float()

        angles = self.rot.unsqueeze(0)
        R      = T.euler_angles_to_matrix(angles, convention="XYZ")[0]
        verts  = self.mesh.verts_packed()
        verts  = verts @ R.T + self.pos
        mesh   = self.mesh.update_padded(verts.unsqueeze(0))
        images = self.renderer(mesh, lights=self._make_lights())

        rgba  = images[0]
        rgb   = rgba[..., :3]
        alpha = rgba[..., 3:4]
        mask  = (alpha > 0.0).float()

        img_t = rgb * mask + bg_t * (1 - mask)
        return img_t.permute(2, 0, 1).unsqueeze(0)