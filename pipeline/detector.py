from __future__ import annotations

from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from config import (
    DETECTOR_CONFIDENCE_THRESHOLD,
    DETECTOR_NAMES,
    YOLO_INPUT_SIZE,
)


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
        self.categories = categories
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
            "target_boxes": prediction["boxes"][
                prediction["labels"] == self.target_class
            ].cpu().tolist(),
            "detections": [
                {
                    "label": self.categories[int(label)],
                    "class_id": int(label),
                    "confidence": float(score),
                    "box": box.cpu().tolist(),
                }
                for label, score, box in zip(
                    prediction["labels"].cpu(),
                    prediction["scores"].cpu(),
                    prediction["boxes"].cpu(),
                )
            ],
            # Retained for callers written for the original person-only API.
            "human_prob": probability,
            "top_label": self.target_name if scores else "none",
            "all_scores": scores,
        }

    def target_probability(self, image: Union[torch.Tensor, np.ndarray]) -> float:
        return self.classify(image)["target_prob"]

    def target_detection_loss(
        self, image: torch.Tensor, input_size: int | None = None
    ) -> torch.Tensor:
        """Return the strongest differentiable pre-NMS target-class probability.

        Proposal selection itself is discrete, but gradients flow through the
        selected RoI features, class logits, and rendered image.
        """
        image = self._to_image_tensor(image)
        transform = self.model.transform
        original_min_size = transform.min_size
        original_max_size = transform.max_size
        if input_size is not None:
            if input_size < 1:
                raise ValueError("input_size must be positive")
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


class YOLODetector:
    """Ultralytics YOLO adapter with inference and differentiable class score."""

    def __init__(
        self,
        model_name: str = "yolo26n",
        target_class: str = "person",
        device: str = "cpu",
    ):
        try:
            if model_name == "yoloe-26s-seg":
                from ultralytics import YOLOE
            else:
                from ultralytics import YOLO
        except ImportError as error:
            raise ImportError(
                "YOLO detectors require Ultralytics; install or upgrade it with: "
                "pip install -U ultralytics"
            ) from error

        if model_name not in DETECTOR_NAMES or model_name == "fasterrcnn_resnet50_fpn_v2":
            raise ValueError(f"Unsupported YOLO detector: {model_name!r}")
        self.model_name = model_name
        self.device = torch.device(device)
        if model_name == "yoloe-26s-seg":
            self.yolo = YOLOE(f"{model_name}.pt")
            self.yolo.set_classes([target_class])
            self.target_class = 0
            self.target_name = target_class
        else:
            self.yolo = YOLO(f"{model_name}.pt")
        self.model = self.yolo.model.to(self.device).eval()
        names = self.yolo.names
        if isinstance(names, list):
            names = dict(enumerate(names))
        if model_name != "yoloe-26s-seg":
            normalized_names = {
                str(name).casefold(): int(class_id)
                for class_id, name in names.items()
            }
            try:
                self.target_class = normalized_names[target_class.casefold()]
            except KeyError as error:
                examples = ", ".join(list(normalized_names)[:20])
                raise ValueError(
                    f"Unknown target class {target_class!r} for {model_name}; "
                    f"available classes include: {examples}"
                ) from error
            self.target_name = str(names[self.target_class])

    @torch.no_grad()
    def classify(self, image: Union[torch.Tensor, np.ndarray]) -> dict:
        image = self._to_numpy_image(image)
        result = self.yolo.predict(
            image[..., ::-1],
            device=str(self.device),
            conf=DETECTOR_CONFIDENCE_THRESHOLD,
            verbose=False,
        )[0]
        detections = []
        if result.boxes is not None:
            boxes = result.boxes
            for class_id, score, box in zip(
                boxes.cls.int().cpu().tolist(),
                boxes.conf.cpu().tolist(),
                boxes.xyxy.cpu().tolist(),
            ):
                detections.append({
                    "label": str(result.names[class_id]),
                    "class_id": class_id,
                    "confidence": float(score),
                    "box": box,
                })
        target_detections = [
            detection for detection in detections
            if detection["class_id"] == self.target_class
        ]
        scores = [detection["confidence"] for detection in target_detections]
        return {
            "target_prob": max(scores, default=0.0),
            "target_label": self.target_name if scores else "none",
            "target_scores": scores,
            "target_boxes": [detection["box"] for detection in target_detections],
            "detections": detections,
        }

    def _to_numpy_image(self, image: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        image = np.asarray(image)
        if image.ndim != 3:
            raise ValueError("expected an RGB image shaped [H, W, 3] or [3, H, W]")
        if image.shape[-1] in (3, 4):
            image = image[..., :3]
        elif image.shape[0] in (3, 4):
            image = np.moveaxis(image[:3], 0, -1)
        else:
            raise ValueError("expected an RGB image shaped [H, W, 3] or [3, H, W]")
        image = np.ascontiguousarray(image)
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating) and image.size and image.max() <= 1.5:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def target_probability(self, image: Union[torch.Tensor, np.ndarray]) -> float:
        return self.classify(image)["target_prob"]

    def target_detection_loss(
        self, image: torch.Tensor, input_size: int = YOLO_INPUT_SIZE
    ) -> torch.Tensor:
        """Max target-class score from YOLO's raw head, before box postprocessing."""
        if input_size < 1:
            raise ValueError("input_size must be positive")
        if image.ndim != 3:
            raise ValueError("expected an RGB image shaped [H, W, 3] or [3, H, W]")
        if image.shape[-1] in (3, 4):
            image = image[..., :3].permute(2, 0, 1)
        elif image.shape[0] == 4:
            image = image[:3]
        elif image.shape[0] != 3:
            raise ValueError("expected an RGB image shaped [H, W, 3] or [3, H, W]")
        image = image.to(device=self.device, dtype=torch.float32)
        if image.detach().amax() > 1.5:
            image = image / 255.0
        image = image.clamp(0.0, 1.0)

        height, width = image.shape[-2:]
        scale = input_size / max(height, width)
        resized_height = max(1, round(height * scale))
        resized_width = max(1, round(width * scale))
        image = F.interpolate(
            image.unsqueeze(0),
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )
        pad_height = input_size - resized_height
        pad_width = input_size - resized_width
        top = pad_height // 2
        left = pad_width // 2
        image = F.pad(
            image,
            (left, pad_width - left, top, pad_height - top),
            value=114.0 / 255.0,
        )

        output = self.model(image)
        raw = output[1] if isinstance(output, tuple) and len(output) > 1 else None
        if not isinstance(raw, dict):
            raise RuntimeError(
                "YOLO raw detection head output is unavailable; "
                "use a compatible Ultralytics YOLO26 release"
            )
        head = raw.get("one2one", raw.get("one2many", raw))
        if not isinstance(head, dict) or "scores" not in head:
            raise RuntimeError("YOLO raw head did not return class score logits")
        class_scores = head["scores"].sigmoid()
        return class_scores[:, self.target_class, :].amax()


def create_detector(
    model_name: str,
    target_class: str | int = "person",
    device: str = "cpu",
):
    """Construct one of the detector adapters supported by the pipeline."""
    if model_name not in DETECTOR_NAMES:
        raise ValueError(
            f"Unsupported detector {model_name!r}; choose one of: "
            f"{', '.join(DETECTOR_NAMES)}"
        )
    if model_name == "fasterrcnn_resnet50_fpn_v2":
        return ObjectDetectionClassifier(model_name, target_class, device)
    return YOLODetector(model_name, str(target_class), device)

