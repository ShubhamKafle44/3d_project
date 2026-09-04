
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class DifferentiableScene(ABC):

    # ---- asset loading ---------------------------------------------
    @abstractmethod
    def load_mesh(self, path: str, name: str = "mesh") -> None:
        """Load an OBJ mesh and register it under `name`."""

    @abstractmethod
    def load_background(self, path: str) -> None:
        """Load optional static background/scene geometry."""

    # ---- parameters an attack is allowed to change ------------------
    @abstractmethod
    def set_position(self, x: float, y: float, z: float) -> None: ...

    @abstractmethod
    def get_position(self) -> np.ndarray: ...

    @abstractmethod
    def set_rotation_deg(self, yaw: float, pitch: float = 0.0, roll: float = 0.0) -> None: ...

    @abstractmethod
    def get_rotation_deg(self) -> np.ndarray: ...

    @abstractmethod
    def set_lighting(self, intensity: float) -> None: ...

    @abstractmethod
    def get_lighting(self) -> float: ...

    @abstractmethod
    def set_material_color(self, part_name: str, rgb: Tuple[float, float, float]) -> None: ...

    @abstractmethod
    def get_material_color(self, part_name: str) -> np.ndarray: ...

    @abstractmethod
    def set_camera_orbit(
        self,
        distance: float,
        elevation: float,
        azimuth: float,
        target: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        fov: float = 40.0,
    ) -> None: ...

    @property
    @abstractmethod
    def cam_distance(self) -> float: ...

    @property
    @abstractmethod
    def cam_elev(self) -> float: ...

    @property
    @abstractmethod
    def cam_azim(self) -> float: ...

    @property
    @abstractmethod
    def cam_target(self) -> Tuple[float, float, float]: ...

    @property
    @abstractmethod
    def cam_fov(self) -> float: ...

    # ---- render -------------------------------------------------------
    @abstractmethod
    def render(self) -> Optional[np.ndarray]:
        """Return an (H, W, 3) uint8 RGB numpy array, or None on failure."""
