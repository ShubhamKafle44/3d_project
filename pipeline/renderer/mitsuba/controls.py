from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch

import config
from ..collision import boxes_overlap_any


class SceneControlsMixin:
    # ---- position / rotation ----------------------------------------
    def set_position(self, x: float, y: float, z: float) -> None:
        """Set the subject position when it remains visible and collision-free."""
        candidate = np.array([x, y, z], dtype=np.float32)
        if not self.is_position_valid(candidate):
            return
        self.pos = candidate
        if getattr(self, "position_parameter", None) is not None:
            with torch.no_grad():
                self.position_parameter.copy_(torch.as_tensor(self.pos))
            self.position_parameter.grad = None

    def get_position(self) -> np.ndarray:
        """Return the current position, including any active Torch parameter."""
        if getattr(self, "position_parameter", None) is not None:
            return self.position_parameter.detach().cpu().numpy().copy()
        return self.pos.copy()

    def is_position_valid(self, position: np.ndarray) -> bool:
        """Keep the subject in the camera frame and clear of scene geometry."""
        if self.rotation_parameter is not None:
            self.rot_deg = self.rotation_parameter.detach().cpu().numpy().astype(np.float32)
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
        """Project the rotated subject bounds through Mitsuba's perspective camera."""
        if self._subject_lower is None or self._subject_upper is None:
            return True

        lower, upper = self._subject_lower, self._subject_upper
        corners = np.array(
            [[x, y, z] for x in (lower[0], upper[0])
             for y in (lower[1], upper[1])
             for z in (lower[2], upper[2])],
            dtype=np.float32,
        )
        yaw, pitch, roll = np.radians(self.rot_deg)
        cz, sz = np.cos(yaw), np.sin(yaw)
        cx, sx = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(roll), np.sin(roll)
        rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        world = corners @ (rz @ rx @ ry).T + np.asarray(position, dtype=np.float32)

        origin = np.asarray(self._camera_origin(), dtype=np.float32)
        target = np.asarray(self._cam_target, dtype=np.float32)
        forward = target - origin
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array([0, 1, 0], dtype=np.float32))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        relative = world - origin
        depth = relative @ forward
        if np.any(depth <= 0):
            return False

        half_extent = depth * math.tan(math.radians(self._cam_fov) / 2.0)
        ndc_x = (relative @ right) / half_extent
        ndc_y = (relative @ up) / half_extent
        margin = 2.0 * config.POSITION_FRAME_MARGIN_PX / self.image_size
        limit = 1.0 - margin
        return bool(
            np.all(np.abs(ndc_x) <= limit)
            and np.all(np.abs(ndc_y) <= limit)
        )

    def set_rotation_deg(self, yaw: float, pitch: float = 0.0, roll: float = 0.0) -> None:
        """Set yaw, pitch, and roll in degrees and sync the optimizer parameter."""
        self.rot_deg = np.array([yaw, pitch, roll], dtype=np.float32)
        if self.rotation_parameter is not None:
            with torch.no_grad():
                self.rotation_parameter.copy_(torch.as_tensor(self.rot_deg))
            self.rotation_parameter.grad = None

    def get_rotation_deg(self) -> np.ndarray:
        """Return the current yaw, pitch, and roll values in degrees."""
        if self.rotation_parameter is not None:
            return self.rotation_parameter.detach().cpu().numpy().copy()
        return self.rot_deg.copy()

    # ---- lighting -----------------------------------------------------
    def set_lighting(self, intensity: float) -> None:
        """Set the key-light intensity and sync any active Torch parameter."""
        self.ambient_intensity = float(intensity)
        if getattr(self, "lighting_parameter", None) is not None:
            with torch.no_grad():
                self.lighting_parameter.fill_(float(intensity))
            self.lighting_parameter.grad = None

    def get_lighting(self) -> float:
        """Return the current key-light intensity."""
        if getattr(self, "lighting_parameter", None) is not None:
            return float(self.lighting_parameter.detach().cpu())
        return self.ambient_intensity

    # ---- material -------------------------------------------------------
    def set_material_color(self, part_name: str, rgb: Tuple[float, float, float]) -> None:
        """Set a part's fallback color and update the differentiable shirt color."""
        self._part_colors[part_name] = tuple(float(c) for c in rgb)
        if part_name == "shirt" and self.shirt_color_parameter is not None:
            with torch.no_grad():
                self.shirt_color_parameter.copy_(torch.as_tensor(rgb))
            self.shirt_color_parameter.grad = None
            if self._active_differentiable_property == "CLOTHING":
                self._shirt_color_override = tuple(float(c) for c in rgb)

    def get_material_color(self, part_name: str) -> np.ndarray:
        """Return the current color for a part as an RGB NumPy array."""
        if part_name == "shirt" and self.shirt_color_parameter is not None:
            return self.shirt_color_parameter.detach().cpu().numpy().copy()
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
        """Configure the orbit camera's distance, angle, target, and field of view."""
        self._cam_distance = distance
        self._cam_elev = elevation
        self._cam_azim = azimuth
        self._cam_target = target
        self._cam_fov = fov

    @property
    def cam_distance(self) -> float:
        """Return the camera's orbit distance."""
        return self._cam_distance

    @property
    def cam_elev(self) -> float:
        """Return the camera's elevation angle in degrees."""
        return self._cam_elev

    @property
    def cam_azim(self) -> float:
        """Return the camera's azimuth angle in degrees."""
        return self._cam_azim

    @property
    def cam_target(self) -> Tuple[float, float, float]:
        """Return the point the camera looks at."""
        return self._cam_target

    @property
    def cam_fov(self) -> float:
        """Return the camera's perspective field of view in degrees."""
        return self._cam_fov

