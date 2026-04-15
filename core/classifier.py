from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import get_model, get_model_weights

from config import MODEL_NAME, SUCCESS_THRESHOLD

# Set up logging so we can track errors or fallbacks in the console
logger = logging.getLogger(__name__)


# ── Label loading ─────────────────────────────────────────────────────────────

def _load_labels(model_name: str) -> list[str]:
    """Fetch ImageNet category labels from model weight metadata only (no model download)."""
    try:
        # Reach into the model's metadata to grab the human-readable names (e.g., 'Golden Retriever')
        return list(get_model_weights(model_name).DEFAULT.meta["categories"])
    except Exception:
        # If the internet is down or the model name is weird, we use numbers (0-999) as a backup
        logger.warning("Could not load labels for '%s'; using numeric placeholders.", model_name)
        return [str(i) for i in range(1000)]


# Initialize the global list of labels based on the model chosen in config
IMAGENET_LABELS: list[str] = _load_labels(MODEL_NAME)


# ── Classifier ────────────────────────────────────────────────────────────────

class ImageClassifier:
    def __init__(self, device: torch.device):
        self.device = device
        # Boot up the model, the preprocessing logic, and the label list
        self.model, self.preprocess, self.categories = self._load_model(MODEL_NAME)
        # Use standard CrossEntropy for measuring how 'wrong' the model is during optimization
        self.criterion = nn.CrossEntropyLoss()

    def _load_model(self, model_name: str):
        """Load a pre-trained model and extract its preprocessing transform."""
        try:
            # Grab the best available pre-trained weights for this specific model
            weights = get_model_weights(model_name).DEFAULT
            # Build the model architecture and load the weights onto our CPU/GPU
            model   = get_model(model_name, weights=weights).to(self.device)
            # Set to evaluation mode (turns off things like Dropout or Batch Norm updates)
            model.eval()
            return model, weights.transforms(), weights.meta["categories"]
        except Exception as exc:
            # If the specific model fails to load, we fall back to ResNet18 as a safety net
            logger.error("Error loading '%s' (%s). Falling back to resnet18.", model_name, exc)
            return self._load_model("resnet18")

    def top_predictions(self, img_t: torch.Tensor, k: int = 5) -> list[tuple[str, float]]:
        """Return top-k (label, probability) pairs for an image tensor."""
        with torch.no_grad(): # Disable gradient tracking to save memory/speed
            # If a single image is passed, wrap it in a batch of 1
            if img_t.dim() == 3:
                img_t = img_t.unsqueeze(0)

            # Resize/normalize the image, run it through the model, and get raw scores (logits)
            x = self.preprocess(img_t).to(self.device) 
            logits = self.model(x)
            # Convert raw scores into percentages (0.0 to 1.0)
            probs = F.softmax(logits, dim=1)
            # Pick out the highest 'k' confidence scores and their corresponding IDs
            top_probs, top_idxs = torch.topk(probs, k)
        
        # Map those IDs back to human-readable names and return as a list
        return [
            (self.categories[idx.item()], top_probs[0, i].item())
            for i, idx in enumerate(top_idxs[0])
        ]

    def top_class_index(self, img_t: torch.Tensor) -> int:
        """Return the predicted class index for an image tensor."""
        with torch.no_grad():
            if img_t.dim() == 3:
                img_t = img_t.unsqueeze(0)
            x = self.preprocess(img_t).to(self.device)
            logits = self.model(x)
        # Just tell us the ID of whatever the model thinks is the most likely object
        return int(logits.argmax(dim=1).item())


# ── Optimiser ─────────────────────────────────────────────────────────────────

class AdversarialOptimiser:
    """Iteratively perturbs 3-D scene parameters to fool the classifier."""

    def __init__(
        self,
        scene,
        classifier: ImageClassifier,
        target_class: int,
        lr: float = 0.01,
    ):
        self.scene      = scene
        self.classifier = classifier
        # The specific class ID we want to 'trick' the model into seeing
        self.target_idx = torch.tensor([target_class], device=classifier.device)
        self.lr         = lr
        self._running   = True
        self.on_step    = None # Optional callback function for UI updates

        # Define which parts of the 3D scene we are allowed to tweak
        self.params = [
            scene.pos,               # Where the object is
            scene.rot,               # How it's turned
            scene.ambient_intensity, # Overall brightness
            scene.light_pos,         # Where the light is coming from
        ]
        # Use Adam optimizer to nudge these 3D parameters based on the feedback from the AI
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)

    def step(self) -> dict:
        """Perform one gradient-descent step toward the target class."""
        # Reset the "memory" of previous gradients
        self.optimizer.zero_grad()

        # Render the 3D scene into a 2D image while keeping track of the math (differentiable)
        img_raw     = self.scene.render_differentiable()          
        # See what the model thinks of this specific render
        logits      = self.classifier.model(self.classifier.preprocess(img_raw))
        # Calculate how far away we are from the target class we want to achieve
        loss        = self.classifier.criterion(logits, self.target_idx)
        
        # Backpropagate: figure out how to change the scene to reduce that loss
        loss.backward()
        # Apply the changes to the scene's position, rotation, and lighting
        self.optimizer.step()

        # Package up the current progress to show the user
        probs  = F.softmax(logits, dim=1)
        prob   = probs[0, self.target_idx.item()].item()
        # Convert the tensor image back to a standard format for display (NumPy)
        img_np = img_raw[0].permute(1, 2, 0).detach().cpu().numpy()

        return {
            "loss":    loss.item(),
            "prob":    prob,
            "top5":    self.classifier.top_predictions(img_raw),
            "img_np":  img_np,
            "success": prob >= SUCCESS_THRESHOLD, # Stop if we've successfully fooled it
        }

    def run(self, max_steps: int) -> None:
        """Run up to *max_steps* iterations, firing *on_step* after each one."""
        param_names = ["pos", "rot", "ambient", "light"]

        for step_num in range(max_steps):
            # Check if the user clicked 'Stop' in the UI
            if not self._running:
                break

            result         = self.step()
            result["step"] = step_num + 1

            # If we won, capture the final gradients so we can see what was changing most
            if result["success"]:
                result["grad_info"] = {
                    name: param.grad.detach().cpu()
                    for name, param in zip(param_names, self.params)
                    if param.grad is not None
                }

            # Send the updates to whatever function is watching (like a GUI update)
            if self.on_step:
                self.on_step(result)

            # If we've successfully reached the threshold, mission accomplished!
            if result["success"]:
                break

    def stop(self) -> None:
        # Simple flag to kill the 'run' loop manually
        self._running = False