from __future__ import annotations

from typing import Union

import numpy as np
import torch


class ObjectDetectionClassifier:
    """Faster R-CNN inference and a differentiable target-class objective."""

    def __init__(
        self,
        model_name: str = "fasterrcnn_resnet50_fpn_v2",
        target_class: str | int = "person",
        device: str = "cpu",
    ):
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_V2_Weights,
            fasterrcnn_resnet50_fpn_v2,
        )

        if model_name != "fasterrcnn_resnet50_fpn_v2":
            raise ValueError(f"Unsupported detector: {model_name!r}")
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        categories = weights.meta["categories"]
        if isinstance(target_class, str):
            try:
                target_class = categories.index(target_class)
            except ValueError as error:
                raise ValueError(
                    f"Unknown COCO class {target_class!r}; choose from {categories[1:]}"
                ) from error
        if not isinstance(target_class, int) or not 0 < target_class < len(categories):
            raise ValueError(
                "target_class must be a COCO class name or ID from "
                f"1 to {len(categories) - 1}"
            )

        self.model_name = model_name
        self.target_class = target_class
        self.target_name = categories[target_class]
        self.device = torch.device(device)
        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=weights,
            box_score_thresh=0.05,
        ).to(self.device).eval()

    def _to_image_tensor(self, image: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(np.ascontiguousarray(image))

        if image.ndim != 3:
            raise ValueError("expected an RGB image shaped [H, W, 3] or [3, H, W]")
        if image.shape[-1] in (3, 4):
            image = image[..., :3].permute(2, 0, 1)
        elif image.shape[0] == 4:
            image = image[:3]
        elif image.shape[0] != 3:
            raise ValueError("expected an RGB image shaped [H, W, 3] or [3, H, W]")

        image = image.to(device=self.device, dtype=torch.float32)

        #NORMALIZE THE TENSORS
        if image.detach().amax() > 1.5:
            image = image / 255.0
        return image.clamp(0.0, 1.0)

    @torch.no_grad()
    def classify(self, image: Union[torch.Tensor, np.ndarray]) -> dict:
        image = self._to_image_tensor(image)
        prediction = self.model([image])[0]
        target_scores = prediction["scores"][
            prediction["labels"] == self.target_class
        ]

        if target_scores.numel() == 0:
            probability = 0.0
            scores = []
        else:
            probability = float(target_scores.max().item())
            scores = target_scores.cpu().tolist()

        return {
            "target_prob": probability,
            "target_label": self.target_name if scores else "none",
            "target_scores": scores,
            # Retained for callers written for the original person-only API.
            "human_prob": probability,
            "top_label": self.target_name if scores else "none",
            "all_scores": scores,
        }

    def target_probability(self, image: Union[torch.Tensor, np.ndarray]) -> float:
        return self.classify(image)["target_prob"]

    def target_detection_loss(
        self,
        image: torch.Tensor,
        input_size: int = 384,
    ) -> torch.Tensor:
        """Return the strongest differentiable pre-NMS target-class probability.

        Proposal selection itself is discrete, but gradients flow through the
        selected RoI features, class logits, and rendered image.
        """
        image = self._to_image_tensor(image)
        if input_size < 1:
            raise ValueError("input_size must be positive")

        transform = self.model.transform
        original_min_size = transform.min_size
        original_max_size = transform.max_size
        transform.min_size = (input_size,)
        transform.max_size = input_size
        try:
            images, _ = transform([image], None)
        finally:
            transform.min_size = original_min_size
            transform.max_size = original_max_size

        features = self.model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = {"0": features}

        proposals, _ = self.model.rpn(images, features, None)
        roi_heads = self.model.roi_heads
        pooled_features = roi_heads.box_roi_pool(
            features,
            proposals,
            images.image_sizes,
        )
        box_features = roi_heads.box_head(pooled_features)
        class_logits, _ = roi_heads.box_predictor(box_features)
        if class_logits.shape[0] == 0:
            raise RuntimeError("Faster R-CNN produced no proposals for the image")

        target_probabilities = class_logits.softmax(dim=1)[:, self.target_class]
        return target_probabilities.max()

