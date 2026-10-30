from __future__ import annotations

import numpy as np
from tempfile import TemporaryDirectory
from typing import Dict, Optional, Tuple

import mitsuba as mi
import config

from ..base import DifferentiableScene
from .assets import AssetLoadingMixin
from .controls import SceneControlsMixin
from .assembly import SceneAssemblyMixin
from .rendering import SceneRenderingMixin

if mi.variant() is None:
    mi.set_variant(config.MITSUBA_VARIANT)


class MitsubaScene(
    AssetLoadingMixin,
    SceneControlsMixin,
    SceneAssemblyMixin,
    SceneRenderingMixin,
    DifferentiableScene,
):
    """Mitsuba scene facade, composed from focused renderer components."""

    def __init__(self, device: str = "cpu", image_size: int = 512):
        """Initialize scene state, render settings, and differentiable parameters."""
        self.image_size = image_size
        self._spp = config.MITSUBA_SPP

        self._part_paths: Dict[str, str] = {}
        self._part_colors: Dict[str, Tuple[float, float, float]] = {}
        self._part_dirs: Dict[str, TemporaryDirectory] = {}
        self._part_shapes: Dict[str, list] = {}  # name -> [(scene key, obj path, bsdf or None)]
        self._background_path: Optional[str] = None
        self._background_dir: Optional[TemporaryDirectory] = None
        self._background_shapes: Optional[dict] = None
        self._subject_lower: Optional[np.ndarray] = None
        self._subject_upper: Optional[np.ndarray] = None
        self._background_face_lower: Optional[np.ndarray] = None
        self._background_face_upper: Optional[np.ndarray] = None

        self.pos = np.zeros(3, dtype=np.float32)
        self.rot_deg = np.zeros(3, dtype=np.float32)  # yaw, pitch, roll
        self.position_parameter = None
        self.rotation_parameter = None
        self.lighting_parameter = None
        self.shirt_color_parameter = None
        self._shirt_color_override = None
        self._active_differentiable_property = None
        self.ambient_intensity = 1.0
        self.light_color = config.LIGHT.get("color", (1.0, 1.0, 1.0))

        self._cam_distance = 3.0
        self._cam_elev = 10.0
        self._cam_azim = 0.0
        self._cam_target = (0.0, 0.0, 0.0)
        self._cam_fov = 40.0

