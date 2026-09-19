
from __future__ import annotations
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

COCO_PERSON_IDX = 1


class HumanDetectionClassifier:
    def __init__(self, model_name: str = "fasterrcnn_resnet50_fpn_v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = torch.device(device)

        if model_name == "fasterrcnn_resnet50_fpn_v2":
            from torchvision.models.detection import (
                fasterrcnn_resnet50_fpn_v2,
                FasterRCNN_ResNet50_FPN_V2_Weights,
            )
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.05)
        elif model_name == "retinanet_resnet50_fpn_v2":
            from torchvision.models.detection import (
                retinanet_resnet50_fpn_v2,
                RetinaNet_ResNet50_FPN_V2_Weights,
            )
            weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
            self.model = retinanet_resnet50_fpn_v2(weights=weights, box_score_thresh=0.05)
        else:
            raise ValueError(f"Unknown detector: {model_name}")

        self.model.to(self.device).eval()
        self.transform = weights.transforms()

    @torch.no_grad()
    def classify(self, img: Union[torch.Tensor, np.ndarray]) -> dict:
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(np.ascontiguousarray(img))
        img = img.to(self.device).float()

        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]
        if img.max() > 1.5:
            img = img / 255.0
        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1)

        predictions = self.model([img])
        pred = predictions[0]
        labels, scores = pred["labels"], pred["scores"]

        person_mask = labels == COCO_PERSON_IDX
        person_scores = scores[person_mask]

        if len(person_scores) > 0:
            human_prob = person_scores.max().item()
            top_label = "person"
        else:
            human_prob = 0.0
            top_label = "none"

        return {
            "human_prob": human_prob,
            "num_persons": int(person_mask.sum().item()),
            "top_label": top_label,
            "all_scores": person_scores.cpu().tolist(),
        }

    def human_probability(self, img) -> float:
        return self.classify(img)["human_prob"]

    def person_detection_loss(
        self,
        image: torch.Tensor,
        topk: int = 256,
        input_size: int = 384,
        person_weight: float = 1.0,
        rpn_weight: float = 0.25,
    ) -> torch.Tensor:
        """Differentiable Faster R-CNN surrogate for reducing *person* scores.

        Final detector boxes pass through thresholding and NMS, so their
        scores are unsuitable as an autograd objective.  This loss instead
        uses the pre-NMS RPN logits and the RoI classifier logits for COCO's
        person class.  NMS only selects the RoIs; gradients still flow from
        their class logits through the detector and rendered RGB image.
        """
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("expected a differentiable RGB image shaped [3, H, W]")
        if not hasattr(self.model, "rpn") or not hasattr(self.model, "roi_heads"):
            raise TypeError(
                "person_detection_loss requires a Faster R-CNN-style detector"
            )
        image = image.to(self.device, dtype=torch.float32).clamp(0.0, 1.0)
        # The normal detector transform upsamples inputs to 800px, which is
        # wasteful during backpropagation on a small GPU.  Use a compact
        # surrogate pass; the unmodified detector is still used for validation.
        transform = self.model.transform
        original_min_size, original_max_size = transform.min_size, transform.max_size
        transform.min_size, transform.max_size = (input_size,), input_size
        try:
            images, _ = transform([image], None)
        finally:
            transform.min_size, transform.max_size = original_min_size, original_max_size
        features = self.model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = {"0": features}
        objectness, _ = self.model.rpn.head(list(features.values()))
        logits = torch.cat([level.reshape(-1) for level in objectness])
        strongest = torch.topk(logits, k=min(topk, logits.numel())).values
        rpn_loss = F.softplus(strongest).mean()

        # ``rpn`` performs non-differentiable proposal filtering, but that is
        # only used to choose pooling regions.  The pooled feature values and
        # class logits retain their gradient path to ``image``.
        proposals, _ = self.model.rpn(images, features, None)
        box_features = self.model.roi_heads.box_roi_pool(
            features, proposals, images.image_sizes
        )
        box_features = self.model.roi_heads.box_head(box_features)
        class_logits, _ = self.model.roi_heads.box_predictor(box_features)
        if class_logits.numel() == 0:
            return rpn_weight * rpn_loss

        person_logits = class_logits[:, COCO_PERSON_IDX]
        non_person_logits = torch.cat(
            (class_logits[:, :COCO_PERSON_IDX], class_logits[:, COCO_PERSON_IDX + 1 :]),
            dim=1,
        )
        # Margin to the strongest competing category is more stable than
        # optimizing a post-softmax probability (which can saturate early).
        person_margin = person_logits - non_person_logits.logsumexp(dim=1)
        strongest_person = torch.topk(
            person_margin, k=min(topk, person_margin.numel())
        ).values
        person_loss = F.softplus(strongest_person).mean()
        return person_weight * person_loss + rpn_weight * rpn_loss

    # Backward-compatible proposal-only objective for existing callers.
    def rpn_objectness_loss(
        self, image: torch.Tensor, topk: int = 256, input_size: int = 384
    ) -> torch.Tensor:
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("expected a differentiable RGB image shaped [3, H, W]")
        if not hasattr(self.model, "rpn"):
            raise TypeError("rpn_objectness_loss requires an RPN-style detector")
        image = image.to(self.device, dtype=torch.float32).clamp(0.0, 1.0)
        transform = self.model.transform
        original_min_size, original_max_size = transform.min_size, transform.max_size
        transform.min_size, transform.max_size = (input_size,), input_size
        try:
            images, _ = transform([image], None)
        finally:
            transform.min_size, transform.max_size = original_min_size, original_max_size
        features = self.model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = {"0": features}
        objectness, _ = self.model.rpn.head(list(features.values()))
        logits = torch.cat([level.reshape(-1) for level in objectness])
        strongest = torch.topk(logits, k=min(topk, logits.numel())).values
        return F.softplus(strongest).mean()
