from __future__ import annotations

import numpy as np
import torch


def linear_to_srgb(x):
    """Convert linear RGB to sRGB for either NumPy arrays or Torch tensors."""
    if isinstance(x, torch.Tensor):
        x = x.clamp(0.0, 1.0)
        return torch.where(
            x <= 0.0031308,
            12.92 * x,
            1.055 * x.clamp_min(0.0031308) ** (1.0 / 2.4) - 0.055,
        )

    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def srgb_output() -> bool:
    """Report whether rendered images should be converted to display-ready sRGB."""
    import config
    return bool(getattr(config, "MITSUBA_SRGB_OUTPUT", True))


def torch_from_numpy(array):
    """Copy an array into a float32 Torch tensor for the autograd bridge."""
    return torch.from_numpy(np.array(array, copy=True).astype(np.float32))
