from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

import config
from ..base import DifferentiableScene
from .dependencies import PYTORCH3D_AVAILABLE, Meshes, RasterizationSettings
from .assets import AssetLoadingMixin
from .controls import SceneControlsMixin
from .assembly import SceneAssemblyMixin
from .rendering import SceneRenderingMixin


class PyTorch3DScene(
    AssetLoadingMixin,
    SceneControlsMixin,
    SceneAssemblyMixin,
    SceneRenderingMixin,
    DifferentiableScene,
):
    def __init__(self, device: str = "cpu", image_size: int = 512):
        if not PYTORCH3D_AVAILABLE:
            raise ImportError(
                "pytorch3d is not installed. Install it per "
                "https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md "
                "for your torch/CUDA version."
            )
        self.device = torch.device(device)
        self.image_size = image_size

        self.parts: Dict[str, Meshes] = {}
        self.background: Optional[Meshes] = None
        self._background_is_textured = False
        self._subject_lower: Optional[np.ndarray] = None
        self._subject_upper: Optional[np.ndarray] = None
        self._background_face_lower: Optional[np.ndarray] = None
        self._background_face_upper: Optional[np.ndarray] = None

        # Adversary-controlled parameters (Tensors for Autograd)
        # Keep the two shared adversary controls as optimizer-ready leaves.
        self.pos = torch.zeros(3, device=self.device, requires_grad=True)
        self.rot_deg = torch.zeros(
            3, device=self.device, requires_grad=True
        )  # yaw, pitch, roll
        self.ambient_intensity = torch.tensor(
            1.0, device=self.device, requires_grad=True
        )
        self.shirt_color_parameter = None
        self._active_differentiable_property = None
        self.light_position = torch.tensor(
            config.LIGHT["position"], dtype=torch.float32, device=self.device
        )
        self.light_color = torch.tensor(
            config.LIGHT["color"], dtype=torch.float32, device=self.device
        )
        self.fill_light_position = torch.tensor(
            config.FILL_LIGHT["position"], dtype=torch.float32, device=self.device
        )
        self.fill_light_color = torch.tensor(
            config.FILL_LIGHT["color"], dtype=torch.float32, device=self.device
        )

        # Camera state
        self._cam_distance = 14
        self._cam_elev = 10.0
        self._cam_azim = 0.0
        self._cam_target = (0.0, 1.0, 0.0)
        self._cam_fov = 40.0

        self._raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            # Retain surfaces behind transparent leaf texels so cutouts reveal
            # the building/other foliage instead of a blank background.
            faces_per_pixel=4,
            # The road scene has many small faces in the same screen bins.
            # These settings prevent coarse-rasterization overflow and the
            # resulting incomplete geometry warnings.
            # The CUDA kernel needs fewer than 22 bins on either screen axis.
            # Scale this with resolution: 32 at 512px, 64 at 1024px, and 128
            # at 2048px. CPU rendering uses the supported naive path.
            bin_size=(
                2 ** max(int(np.ceil(np.log2(self.image_size))) - 4, 4)
                if self.device.type == "cuda"
                else 0
            ),
            max_faces_per_bin=200_000,
        )

