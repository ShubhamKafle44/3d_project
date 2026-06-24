from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
from torchvision.models import (
    ViT_B_16_Weights,
    ResNet50_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
)

import config

# ── Model factory ──────────────────────────────────────────────────────────────
_MODEL_FACTORY: dict[str, tuple] = {
    "vit_b_16":           (tvm.vit_b_16,           ViT_B_16_Weights.DEFAULT),
    "resnet50":           (tvm.resnet50,            ResNet50_Weights.DEFAULT),
    "efficientnet_b0":    (tvm.efficientnet_b0,     EfficientNet_B0_Weights.DEFAULT),
    "mobilenet_v3_large": (tvm.mobilenet_v3_large,  MobileNet_V3_Large_Weights.DEFAULT),
}

# FIX: Resize and CenterCrop were missing — models require exactly 224×224
_PREPROCESS = T.Compose([
    T.Resize(256),          # scale short edge to 256
    T.CenterCrop(224),      # crop centre 224×224
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class ImageClassifier:

    def __init__(self, model_name: str | None = None):
        self.device     = config.device
        self.model_name = model_name or config.MODEL_NAME
        self._load_model(self.model_name)

    def _load_model(self, model_name: str) -> None:
        if model_name not in _MODEL_FACTORY:
            raise ValueError(
                f"Unknown model '{model_name}'. Available: {list(_MODEL_FACTORY)}"
            )
        model_fn, weights  = _MODEL_FACTORY[model_name]
        self.model         = model_fn(weights=weights).to(self.device).eval()
        self.model_name    = model_name
        self.categories: list[str] = weights.meta["categories"]

    def swap_model(self, model_name: str) -> None:
        if model_name == self.model_name:
            return
        del self.model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self._load_model(model_name)

    @torch.no_grad()
    def classify(self, image_tensor: torch.Tensor) -> list[tuple[str, float]]:
        x      = self._prepare(image_tensor)
        logits = self.model(x)
        probs  = torch.softmax(logits, dim=1)[0]
        top5_p, top5_i = probs.topk(5)
        return [(self.categories[i], p.item()) for i, p in zip(top5_i, top5_p)]

    def classify_for_grad(self, image_tensor: torch.Tensor) -> torch.Tensor:
        x = self._prepare(image_tensor, keep_grad=True)
        return self.model(x)

    def _prepare(
        self,
        image_tensor: torch.Tensor,
        keep_grad: bool = False,
    ) -> torch.Tensor:
        """Normalise (H, W, 3) or (1, 3, H, W) float32 [0,1] → (1, 3, 224, 224)."""
        x = image_tensor
        if x.ndim == 4:
            x = x.squeeze(0)          # (1, 3, H, W) → (3, H, W)
        if x.shape[0] != 3:
            if x.shape[-1] == 3:
                x = x.permute(2, 0, 1)   # (H, W, 3) → (3, H, W)
            else:
                raise ValueError(f"Invalid input shape: {x.shape}")
        x = x.float().clamp(0.0, 1.0)
        x = _PREPROCESS(x)            # → (3, 224, 224)  includes Resize+Crop+Normalize
        x = x.unsqueeze(0).to(self.device)   # → (1, 3, 224, 224)
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