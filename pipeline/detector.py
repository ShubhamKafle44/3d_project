
from __future__ import annotations
from typing import Union

import numpy as np
import torch

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
