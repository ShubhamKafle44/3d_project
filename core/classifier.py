from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
)

import config

logger = logging.getLogger(__name__)

# ── Model factory ──────────────────────────────────────────────────────────────
_MODEL_FACTORY: dict[str, tuple] = {
    "vit_b_16":           (tvm.vit_b_16,           ViT_B_16_Weights.DEFAULT),
    "resnet50":           (tvm.resnet50,            ResNet50_Weights.DEFAULT),
    "mobilenet_v3_large": (tvm.mobilenet_v3_large,  MobileNet_V3_Large_Weights.DEFAULT),
}

# Standard ImageNet preprocessing — brings any image to 224×224
_PREPROCESS = T.Compose([
    T.Resize(256, antialias=True),
    T.CenterCrop(224),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Torch hub cache directory
_HUB_CACHE = Path(torch.hub.get_dir()) / "checkpoints"


def _clear_cached_weights(weights_enum) -> None:
    """Delete any cached .pth files whose name matches the weights URL filename.
    Called when a hash mismatch is detected so the next load re-downloads cleanly.
    """
    try:
        url      = weights_enum.url         
        filename = url.split("/")[-1]
        cached   = _HUB_CACHE / filename
        if cached.exists():
            cached.unlink()
            logger.warning("Deleted corrupt cache file: %s", cached)
    except Exception as exc:
        logger.warning("Could not clear cache: %s", exc)


def _load_weights_safe(model_fn, weights_enum, device: torch.device):
    """Load a torchvision model, retrying once after clearing a corrupt cache."""
    try:
        return model_fn(weights=weights_enum).to(device).eval()
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "hash" in msg or "invalid" in msg or "checksum" in msg:
            logger.warning(
                "Hash mismatch detected for %s — clearing cache and retrying.",
                weights_enum,
            )
            _clear_cached_weights(weights_enum)
            # Retry — this time downloads a fresh copy
            return model_fn(weights=weights_enum).to(device).eval()
        raise   # re-raise unrelated errors


class ImageClassifier:

    def __init__(self, model_name: str | None = None):
        self.device     = config.device
        self.model_name = model_name or config.MODEL_NAME
        self._load_model(self.model_name)

    # ------------------------------------------------------------------
    def _load_model(self, model_name: str) -> None:
        if model_name not in _MODEL_FACTORY:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(_MODEL_FACTORY)}"
            )
        model_fn, weights = _MODEL_FACTORY[model_name]
        self.model        = _load_weights_safe(model_fn, weights, self.device)
        self.model_name   = model_name
        self.categories: list[str] = weights.meta["categories"]

    # ------------------------------------------------------------------
    def swap_model(self, model_name: str) -> None:
        """Atomic hot-swap — self.model is always valid after this returns."""
        if model_name == self.model_name:
            return

        # Keep old state so we can roll back on any failure
        old_model      = self.model
        old_name       = self.model_name
        old_categories = self.categories

        try:
            self._load_model(model_name)      # sets self.model on success
        except Exception:
            # Restore — classifier stays usable even if the new model fails to load
            self.model      = old_model
            self.model_name = old_name
            self.categories = old_categories
            raise
        else:
            # Success — now safe to free the old model
            del old_model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def classify(self, image_tensor: torch.Tensor) -> list[tuple[str, float]]:
        x      = self._prepare(image_tensor)
        logits = self.model(x)
        probs  = torch.softmax(logits, dim=1)[0]
        top5_p, top5_i = probs.topk(5)
        return [(self.categories[i], p.item()) for i, p in zip(top5_i, top5_p)]

    # ------------------------------------------------------------------
    def classify_for_grad(self, image_tensor: torch.Tensor) -> torch.Tensor:
        x = self._prepare(image_tensor, keep_grad=True)
        return self.model(x)

    # ------------------------------------------------------------------
    def _prepare(
        self,
        image_tensor: torch.Tensor,
        keep_grad: bool = False,
    ) -> torch.Tensor:
        """Convert any supported layout → (1, 3, 224, 224) normalised float32."""
        x = image_tensor
        if x.ndim == 4:
            x = x.squeeze(0)
        if x.shape[0] != 3:
            if x.shape[-1] == 3:
                x = x.permute(2, 0, 1)
            else:
                raise ValueError(f"Invalid input shape: {x.shape}")
        x = x.float().clamp(0.0, 1.0)
        x = _PREPROCESS(x)
        x = x.unsqueeze(0).to(self.device)
        if not keep_grad:
            x = x.detach()
        return x


# ── Adversarial optimiser ─────────────────────────────────────────────────────

class AdversarialOptimiser:

    def __init__(
        self,
        scene,
        classifier: ImageClassifier,
        target_class: int,
        lr: float = 0.05,
    ):
        self.scene            = scene
        self.classifier       = classifier
        self.target_class_idx = target_class
        self._running         = True
        self.target           = torch.tensor([target_class], device=config.device)
        self._build_optimiser(lr)

    def _build_optimiser(self, lr: float) -> None:
        self.optimiser = torch.optim.Adam(
            [self.scene.pos, self.scene.rot,
             self.scene.light_pos, self.scene.ambient_intensity],
            lr=lr,
        )

    def stop(self) -> None:
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def step(self) -> tuple[float, float]:
        self.optimiser.zero_grad()
        img    = self.scene.render_differentiable().float()
        logits = self.classifier.classify_for_grad(img)
        loss   = nn.CrossEntropyLoss()(logits, self.target)
        loss.backward()
        self.optimiser.step()
        with torch.no_grad():
            prob = torch.softmax(logits, dim=1)[0, self.target_class_idx].item()
        return loss.item(), prob