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

        # Adversary-controlled parameters (Tensors for Autograd)
        self.pos = torch.zeros(3, device=self.device)
        self.rot_deg = torch.zeros(3, device=self.device)  # yaw, pitch, roll
        self.ambient_intensity = torch.tensor(1.0, device=self.device)

        # Camera state
        self._cam_distance = 14
        self._cam_elev = 10.0
        self._cam_azim = 0.0
        self._cam_target = (0.0, 1.0, 0.0)
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

            meshes.append(m)

        if len(meshes) == 1:
            scene_mesh = meshes[0]
        else:
            scene_mesh = join_meshes_as_scene(meshes)

        # ============================================================
        # 2. Find HUMAN bounding box
        # ============================================================
        verts = scene_mesh.verts_packed()

        bbox_min = verts.min(dim=0).values
        bbox_max = verts.max(dim=0).values

        center = (bbox_min + bbox_max) / 2.0
        size = bbox_max - bbox_min
        max_size = torch.max(size)

        print("\n========================================")
        print("PYTORCH3D HUMAN DEBUG")
        print("========================================")
        print("bbox min :", bbox_min.detach().cpu().numpy())
        print("bbox max :", bbox_max.detach().cpu().numpy())
        print("center   :", center.detach().cpu().numpy())
        print("size     :", size.detach().cpu().numpy())
        print("max size :", max_size.item())
        print("========================================\n")

        # ============================================================
        # 3. Automatically aim camera at human
        # ============================================================
        camera_target = center.detach()

        # Convert tensor to tuple
        camera_target_tuple = tuple(
            camera_target.cpu().numpy().tolist()
        )

        # ============================================================
        # 4. Automatically choose camera distance
        # ============================================================
        fov_rad = np.deg2rad(self._cam_fov)

        # Fit object vertically in the camera
        distance = (
            max_size.item()
            / (2.0 * np.tan(fov_rad / 2.0))
        )

        # Add some margin
        distance *= 1.5

        # Prevent absurdly small distances
        distance = max(distance, 0.1)

        print("Camera target :", camera_target_tuple)
        print("Camera dist   :", distance)

        # ============================================================
        # 5. Camera
        # ============================================================
        R, T = look_at_view_transform(
            dist=distance,
            elev=self._cam_elev,
            azim=self._cam_azim,
            at=(camera_target_tuple,),
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
        lights = PointLights(
            device=self.device,

            # Position of the point light
            location=[
                [-2.0, 5.0, 5.0]
            ],

            # No ambient light
            ambient_color=(
                (0.0, 0.0, 0.0),
            ),

            # Main illumination from point light
            diffuse_color=(
                (0.7, 0.7, 0.7),
            ),

            # Very small specular highlight
            specular_color=(
                (0.05, 0.05, 0.05),
            ),
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
        renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=SoftPhongShader(
                device=self.device,
                cameras=cameras,
                lights=lights,
            ),
        )

        # ============================================================
        # 9. Render
        # ============================================================
        images = renderer(scene_mesh)

        img = images[0, ..., :3].clamp(0.0, 1.0)

        return img.permute(2, 0, 1)