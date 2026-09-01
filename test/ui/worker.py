from __future__ import annotations

import numpy as np
import torch
from PyQt6.QtCore import QObject, pyqtSignal

import config as config
from core.classifier import AdversarialOptimiser


class OptimWorker(QObject):
    """Runs the adversarial optimisation loop in a background thread."""

    step_done = pyqtSignal(dict)
    finished  = pyqtSignal()

    def __init__(self, optimiser: AdversarialOptimiser, max_steps: int):
        super().__init__()
        self.optimiser = optimiser
        self.max_steps = max_steps

    def run(self) -> None:
        """Entry point called by QThread.started signal."""
        opt = self.optimiser

        for step in range(1, self.max_steps + 1):
            if not opt.is_running:
                break

            loss, prob = opt.step()

            # ── Grab the current rendered numpy image for display ──────────────
            try:
                img_np, _ = opt.scene.render()
            except Exception:
                img_np = None

            # ── Top-5 predictions from the current render ─────────────────────
            top5: list[tuple[str, float]] = []
            if img_np is not None:
                try:
                    img_t  = torch.from_numpy(img_np.astype(np.float32)).to(config.device)
                    top5   = opt.classifier.classify(img_t)
                except Exception:
                    pass

            # ── Gradient info for the success callback ────────────────────────
            grad_info: dict = {}
            for name, param in [
                ("pos",               opt.scene.pos),
                ("rot",               opt.scene.rot),
                ("light_pos",         opt.scene.light_pos),
                ("ambient_intensity", opt.scene.ambient_intensity),
            ]:
                if param.grad is not None:
                    grad_info[name] = param.grad.detach().cpu()

            success = prob >= config.SUCCESS_THRESHOLD

            self.step_done.emit({
                "step":      step,
                "loss":      loss,
                "prob":      prob,
                "img_np":    img_np,
                "top5":      top5,
                "grad_info": grad_info,
                "success":   success,
            })

            if success:
                break

        self.finished.emit()