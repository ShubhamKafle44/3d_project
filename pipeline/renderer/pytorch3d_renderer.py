"""
PyTorch3D backend.

Requires: torch, pytorch3d (https://github.com/facebookresearch/pytorch3d)
Install pytorch3d per its official instructions - it must be built against
the exact torch/CUDA version in your environment; there is no universal
pip wheel.

This scene loads one or more part meshes, keeps a single rigid transform
(position + rotation) applied to all of them, exposes per-part vertex-color
material control, a point light with adjustable intensity, and an
orbiting camera. Everything is a torch.nn.Parameter-free plain tensor by
default (attributes are not `requires_grad` unless you want to do
gradient-based attacks - see the note at the bottom of the file).
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .base import DifferentiableScene

try:
    from pytorch3d.io import load_objs_as_meshes
    from pytorch3d.structures import join_meshes_as_scene, Meshes
    from pytorch3d.renderer import (
        look_at_view_transform,
        FoVPerspectiveCameras,
        PointLights,
        RasterizationSettings,
        MeshRenderer,
        MeshRasterizer,
        SoftPhongShader,
        TexturesVertex,
    )
    _PYTORCH3D_AVAILABLE = True
except ImportError:
    _PYTORCH3D_AVAILABLE = False


class PyTorch3DScene(DifferentiableScene):
    def __init__(self, device: str = "cpu", image_size: int = 512):
        if not _PYTORCH3D_AVAILABLE:
            raise ImportError(
                "pytorch3d is not installed. Install it per "
                "https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md "
                "for your torch/CUDA version."
            )
        self.device = torch.device(device)
        self.image_size = image_size

        self.parts: Dict[str, Meshes] = {}
        self.background: Optional[Meshes] = None

        # Adversary-controlled parameters
        self.pos = torch.zeros(3, device=self.device)
        self.rot_deg = torch.zeros(3, device=self.device)  # yaw, pitch, roll
        self.ambient_intensity = torch.tensor(1.0, device=self.device)

        # Camera state
        self._cam_distance = 3.0
        self._cam_elev = 10.0
        self._cam_azim = 0.0
        self._cam_target = (0.0, 0.0, 0.0)
        self._cam_fov = 40.0

        self._raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

    # ---- asset loading ---------------------------------------------
    def load_mesh(self, path: str, name: str = "mesh") -> None:
        mesh = load_objs_as_meshes([path], device=self.device)
        if not mesh.textures:
            verts = mesh.verts_packed()
            white = torch.ones_like(verts)[None]
            mesh.textures = TexturesVertex(verts_features=white)
        self.parts[name] = mesh

    def load_background(self, path: str) -> None:
        self.background = load_objs_as_meshes([path], device=self.device)

    # ---- position / rotation ----------------------------------------
    def set_position(self, x: float, y: float, z: float) -> None:
        self.pos = torch.tensor([x, y, z], device=self.device, dtype=torch.float32)

    def get_position(self) -> np.ndarray:
        return self.pos.detach().cpu().numpy().copy()

    def set_rotation_deg(self, yaw: float, pitch: float = 0.0, roll: float = 0.0) -> None:
        self.rot_deg = torch.tensor([yaw, pitch, roll], device=self.device, dtype=torch.float32)

    def get_rotation_deg(self) -> np.ndarray:
        return self.rot_deg.detach().cpu().numpy().copy()

    # ---- lighting -----------------------------------------------------
    def set_lighting(self, intensity: float) -> None:
        self.ambient_intensity = torch.tensor(float(intensity), device=self.device)

    def get_lighting(self) -> float:
        return float(self.ambient_intensity.detach().cpu().item())

    # ---- material -------------------------------------------------------
    def set_material_color(self, part_name: str, rgb: Tuple[float, float, float]) -> None:
        mesh = self.parts.get(part_name)
        if mesh is None:
            return
        n_verts = mesh.verts_packed().shape[0]
        color = torch.tensor(rgb, device=self.device, dtype=torch.float32)
        colors = color.unsqueeze(0).expand(n_verts, 3).unsqueeze(0).clone()
        mesh.textures = TexturesVertex(verts_features=colors)

    def get_material_color(self, part_name: str) -> np.ndarray:
        mesh = self.parts.get(part_name)
        if mesh is not None and hasattr(mesh.textures, "verts_features_list"):
            return mesh.textures.verts_features_list()[0][0].detach().cpu().numpy().copy()
        return np.array([1.0, 1.0, 1.0])

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

    # ---- transform helpers -------------------------------------------
    def _rotation_matrix(self) -> torch.Tensor:
        yaw, pitch, roll = (self.rot_deg * np.pi / 180.0).unbind(0)

        cz, sz = torch.cos(yaw), torch.sin(yaw)
        Rz = torch.stack([
            torch.stack([cz, -sz, torch.zeros_like(cz)]),
            torch.stack([sz, cz, torch.zeros_like(cz)]),
            torch.stack([torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)]),
        ])

        cx, sx = torch.cos(pitch), torch.sin(pitch)
        Rx = torch.stack([
            torch.stack([torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx)]),
            torch.stack([torch.zeros_like(cx), cx, -sx]),
            torch.stack([torch.zeros_like(cx), sx, cx]),
        ])

        cy, sy = torch.cos(roll), torch.sin(roll)
        Ry = torch.stack([
            torch.stack([cy, torch.zeros_like(cy), sy]),
            torch.stack([torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy)]),
            torch.stack([-sy, torch.zeros_like(cy), cy]),
        ])
        return (Rz @ Rx @ Ry).to(self.device)

    def _assemble_scene_mesh(self) -> Meshes:
        R = self._rotation_matrix()
        meshes = []
        for mesh in self.parts.values():
            verts = mesh.verts_padded()[0]
            verts = verts @ R.T + self.pos
            m = mesh.clone()
            m = m.update_padded(verts.unsqueeze(0))
            meshes.append(m)
        if self.background is not None:
            meshes.append(self.background)
        return join_meshes_as_scene(meshes) if len(meshes) > 1 else meshes[0]

    # ---- render -------------------------------------------------------
    def render(self) -> Optional[np.ndarray]:
        if not self.parts:
            return None
        try:
            scene_mesh = self._assemble_scene_mesh()

            R, T = look_at_view_transform(
                dist=self._cam_distance,
                elev=self._cam_elev,
                azim=self._cam_azim,
                at=(self._cam_target,),
                device=self.device,
            )
            cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=self._cam_fov)

            lights = PointLights(
                device=self.device,
                location=[[2.0, 2.0, 2.0]],
                ambient_color=((self.ambient_intensity.item(),) * 3,),
                diffuse_color=((1.0,) * 3,),
                specular_color=((0.3,) * 3,),
            )

            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(cameras=cameras, raster_settings=self._raster_settings),
                shader=SoftPhongShader(device=self.device, cameras=cameras, lights=lights),
            )

            images = renderer(scene_mesh)
            img = images[0, ..., :3].clamp(0, 1)
            img_np = (img.detach().cpu().numpy() * 255.0).astype(np.uint8)
            return img_np
        except Exception as exc:  # noqa: BLE001
            print(f"[pytorch3d_renderer] render failed: {exc}")
            return None


# ----------------------------------------------------------------------
# Note on gradient-based attacks:
# Because pos/rot_deg/ambient_intensity/material colors are plain tensors
# above, this scene is used for black-box (random-search) attacks by
# default, matching the search strategy used with the Mitsuba backend so
# both renderers are attacked identically. To do a *gradient*-based attack
# instead, set e.g. `scene.pos.requires_grad_(True)` before rendering and
# backprop the detector's score through `scene.render()`'s tensor path
# (expose a `render_differentiable()` that skips the numpy conversion).
