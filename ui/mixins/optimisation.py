from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import QThread

from core.classifier import AdversarialOptimiser, IMAGENET_LABELS
from ui.worker import OptimWorker

# Get our logger ready so we can track what's happening in the background
logger = logging.getLogger(__name__)


class OptimisationMixin:
    """Manages the adversarial optimisation thread lifecycle and step callbacks."""

    def _on_start_optimise(self) -> None:
        # Don't start a second optimizer if one is already running
        if self.opt_thread and self.opt_thread.isRunning():
            return

        # Pull the settings directly from the UI sliders and input boxes
        target = self._target_spin.value()
        lr     = self._lr_spin.value()
        steps  = self._steps_spin.value()

        # Build the engine that calculates how to trick the AI
        self.optimiser = AdversarialOptimiser(
            scene=self.scene,
            classifier=self.classifier,
            target_class=target,
            lr=lr,
        )
        
        # We put the heavy math in a 'Worker' so it doesn't freeze the actual UI window
        worker = OptimWorker(self.optimiser, steps)
        self.opt_thread = QThread()
        worker.moveToThread(self.opt_thread)
        
        # Wire up the thread 'brain': start the work, update on every step, and clean up when done
        self.opt_thread.started.connect(worker.run)
        worker.step_done.connect(self._on_opt_step)
        worker.finished.connect(self._on_opt_finished)
        worker.finished.connect(self.opt_thread.quit)
        
        self._worker = worker
        self.opt_thread.start()

        # Update buttons so the user can't click 'Start' twice
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._success_label.setText("")
        self._saved_label.setText("")
        self.status_bar.showMessage(
            f"Optimising → target: [{target}] {IMAGENET_LABELS[target]}"
        )

    def _on_stop_optimise(self) -> None:
        # Simply flip the 'running' switch in the optimizer to stop the loop early
        if self.optimiser:
            self.optimiser.stop()

    def _on_opt_step(self, result: dict) -> None:
        """Update the UI after each optimisation iteration."""
        prob   = result["prob"]
        img_np = result.get("img_np")

        # Update the labels to show the current iteration and how confident the model is
        self._step_label.setText(f"Step: {result['step']}")
        self._prob_label.setText(
            f"Target prob: {prob * 100:.2f}%  ({'▲' if prob > 0.5 else '—'})"
        )

        # Update the picture in the UI to show the latest 3D render
        if img_np is not None:
            self._show_image(img_np)

        # Refresh the top 5 prediction list and move the 3D sliders to match the new math
        self._update_predictions(result.get("top5", []))
        self._sync_sliders()

        # If the optimizer successfully fooled the AI, trigger the success routine
        if result.get("success"):
            self._on_success(img_np, result.get("grad_info", {}), prob)

    def _on_success(
        self,
        img_np: np.ndarray | None,
        grad_info: dict,
        prob: float,
    ) -> None:
        """Called once target confidence reaches the success threshold."""
        target_idx  = self._target_spin.value()
        target_name = IMAGENET_LABELS[target_idx]

        # Celebrate in the UI
        self._success_label.setText(
            f"classifier!  '{target_name}'  {prob * 100:.1f}%"
        )
        logger.info(
            "ADVERSARIAL SUCCESS — target: [%d] %s  prob=%.2f%%",
            target_idx, target_name, prob * 100,
        )
        
        # Log the final 'gradient' values (how much each parameter had to change)
        for name, grad in grad_info.items():
            logger.info("  Final gradient  %s: %s", name, grad.numpy())

        # Automatically save a copy of the 'trick' image so we can look at it later
        if img_np is not None:
            save_path = f"adversarial_{target_name.replace(' ', '_')}.png"
            self._save_numpy_image(img_np, save_path)
            self._adv_save_path = save_path
            self._saved_label.setText(f"Saved adversarial image → {save_path}")
            self.status_bar.showMessage(
                f"Success!  Adversarial image saved to {save_path}"
            )

    def _on_opt_finished(self) -> None:
        """Runs when the worker finishes all steps or is stopped."""
        # Flip the buttons back to their original state
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        
        # If we didn't hit the success threshold, let the user know it just finished
        if not self._success_label.text():
            self.status_bar.showMessage(
                "Optimisation complete (target threshold not reached)."
            )
        logger.info("Optimisation routine finished.")