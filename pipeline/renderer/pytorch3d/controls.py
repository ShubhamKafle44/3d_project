from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

import config
from ..collision import boxes_overlap_any
from .dependencies import FoVPerspectiveCameras, TexturesVertex, look_at_view_transform


class SceneControlsMixin:
    # ---- position / rotation ----------------------------------------
    def set_position(self, x: float, y: float, z: float) -> None:
        candidate = np.array([x, y, z], dtype=np.float32)
        if self.is_position_valid(candidate):
            self.pos = torch.tensor(
                candidate, device=self.device, dtype=torch.float32, requires_grad=True
            )

    def get_position(self) -> np.ndarray:
        return self.pos.detach().cpu().numpy().copy()

    def is_position_valid(self, position: np.ndarray) -> bool:
        if not self._is_subject_in_frame(position):
            return False
        if self._background_face_lower is None or self._subject_lower is None:
            return True
        offset = np.asarray(position, dtype=np.float32)
        return not boxes_overlap_any(
            self._subject_lower + offset,
            self._subject_upper + offset,
            self._background_face_lower,
            self._background_face_upper,
            config.POSITION_COLLISION_CLEARANCE,
        )

    def _is_subject_in_frame(self, position: np.ndarray) -> bool:
        """Return whether the transformed subject bounding box fits the image."""
        if self._subject_lower is None or self._subject_upper is None:
            return True
        lower, upper = self._subject_lower, self._subject_upper
        corners = np.array(
            [[x, y, z] for x in (lower[0], upper[0])
             for y in (lower[1], upper[1])
             for z in (lower[2], upper[2])],
            dtype=np.float32,
        )
        with torch.no_grad():
            points = torch.as_tensor(corners, device=self.device)
            rotation = self._rotation_matrix()
            offset = torch.as_tensor(position, dtype=torch.float32, device=self.device)
            points = points @ rotation.T + offset
            R, T = look_at_view_transform(
                dist=self._cam_distance,
                elev=self._cam_elev,
                azim=self._cam_azim,
                at=(self._cam_target,),
                device=self.device,
            )
            cameras = FoVPerspectiveCameras(
                device=self.device, R=R, T=T, fov=self._cam_fov
            )
            screen_points = cameras.transform_points_screen(
                points.unsqueeze(0), image_size=((self.image_size, self.image_size),)
            )[0]
            margin = config.POSITION_FRAME_MARGIN_PX
            x, y, depth = screen_points.unbind(dim=1)
            return bool(
                (depth > 0).all()
                and (x >= margin).all()
                and (x <= self.image_size - 1 - margin).all()
                and (y >= margin).all()
                and (y <= self.image_size - 1 - margin).all()
            )

    def set_rotation_deg(self, yaw: float, pitch: float = 0.0, roll: float = 0.0) -> None:
        self.rot_deg = torch.tensor(
            [yaw, pitch, roll], device=self.device, dtype=torch.float32,
            requires_grad=True,
        )

    def get_rotation_deg(self) -> np.ndarray:
        return self.rot_deg.detach().cpu().numpy().copy()

    # ---- lighting -----------------------------------------------------
    def set_lighting(self, intensity: float) -> None:
        self.ambient_intensity = torch.tensor(
            float(intensity), device=self.device, requires_grad=True
        )

    def get_lighting(self) -> float:
        return float(self.ambient_intensity.detach().cpu().item())

    # ---- material -------------------------------------------------------
    def set_material_color(self, part_name: str, rgb: Tuple[float, float, float]) -> None:
        mesh = self.parts.get(part_name)
        if mesh is None:
            return
        if part_name == "shirt" and self.shirt_color_parameter is not None:
            with torch.no_grad():
                self.shirt_color_parameter.copy_(
                    torch.as_tensor(rgb, device=self.device, dtype=torch.float32)
                )
            self.shirt_color_parameter.grad = None
        n_verts = mesh.verts_packed().shape[0]
        color = torch.tensor(rgb, device=self.device, dtype=torch.float32)
        colors = color.unsqueeze(0).expand(n_verts, 3).unsqueeze(0).clone()
        mesh.textures = TexturesVertex(verts_features=colors)

    def get_material_color(self, part_name: str) -> np.ndarray:
        if part_name == "shirt" and self.shirt_color_parameter is not None:
            return self.shirt_color_parameter.detach().cpu().numpy().copy()
        mesh = self.parts.get(part_name)
        if mesh is not None and hasattr(mesh.textures, "verts_features_list"):
            return mesh.textures.verts_features_list()[0][0].detach().cpu().numpy().copy()
        return np.array([1.0, 1.0, 1.0])

    def set_vertex_colors(self, part_name: str, colors: torch.Tensor) -> None:
        """Assign differentiable per-vertex RGB values to a mesh part."""
        mesh = self.parts.get(part_name)
        if mesh is None:
            raise KeyError(f"Unknown mesh part: {part_name}")
        expected = mesh.verts_packed().shape[0]
        if colors.shape != (expected, 3):
            raise ValueError(f"expected colors shaped ({expected}, 3), got {tuple(colors.shape)}")
        mesh.textures = TexturesVertex(verts_features=colors.unsqueeze(0))

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

