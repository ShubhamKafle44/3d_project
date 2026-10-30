"""Compatibility import for the modular PyTorch3D renderer."""

from .pytorch3d import PyTorch3DScene
from .pytorch3d.dependencies import PYTORCH3D_AVAILABLE as _PYTORCH3D_AVAILABLE
from .pytorch3d.shaders import AlphaCutoutPhongShader, InverseSquarePointLights
from .pytorch3d.assets import (
    _material_alpha_images, _pytorch3d_safe_obj, _resolve_mtl_texture,
)

__all__ = ["PyTorch3DScene"]
