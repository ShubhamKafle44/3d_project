"""Compatibility import for the modular Mitsuba renderer."""

from .mitsuba import MitsubaScene
from .mitsuba.materials import read_materials, split_materials, texture_bsdf
from .mitsuba.utils import linear_to_srgb, srgb_output as _srgb_output, torch_from_numpy

__all__ = [
    "MitsubaScene", "read_materials", "split_materials", "texture_bsdf",
    "linear_to_srgb", "torch_from_numpy",
]
