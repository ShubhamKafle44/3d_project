from __future__ import annotations

import logging

import numpy as np
from PIL import Image
from PyQt6.QtCore import QThread

from core.classifier import AdversarialOptimiser
from ui.worker import OptimWorker

logger = logging.getLogger(__name__)


class OptimisationMixin:

    def _on_start_optimise(self) -> None:
        if self.opt_thread and self.opt_thread.isRunning():
            return

        target = self.target_spin.value()          # FIX: was self._target_spin
        lr     = self._lr_spin.value()
        steps  = self._steps_spin.value()

        self.optimiser = AdversarialOptimiser(
            scene=self.scene,
            classifier=self.classifier,
            target_class=target,
            lr=lr,
        )

        worker = OptimWorker(self.optimiser, steps)
        self.opt_thread = QThread()
        worker.moveToThread(self.opt_thread)

        self.opt_thread.started.connect(worker.run)
        worker.step_done.connect(self._on_opt_step)
        worker.finished.connect(self._on_opt_finished)
        worker.finished.connect(self.opt_thread.quit)

        self._worker = worker
        self.opt_thread.start()

        self.btn_start.setEnabled(False)           # FIX: was self._start_btn
        self.btn_stop.setEnabled(True)             # FIX: was self._stop_btn
        self._success_label.setText("")
        self._saved_label.setText("")
        self.status_label.setText(               # FIX: was self.status_bar.showMessage
            f"Optimising → target: [{target}] {self.classifier.categories[target]}"
        )

    def _on_stop_optimise(self) -> None:
        if self.optimiser:
            self.optimiser.stop()

    def _on_opt_step(self, result: dict) -> None:
        prob   = result["prob"]
        img_np = result.get("img_np")

        self._step_label.setText(f"Step: {result['step']}")
        self._prob_label.setText(
            f"Target prob: {prob * 100:.2f}%  ({'▲' if prob > 0.5 else '—'})"
        )

        if img_np is not None:
            self._show_numpy(img_np)              # FIX: was self._show_image

        self._update_predictions(result.get("top5", []))
        self._sync_sliders()

        # Update target probability bar
        self.target_prob_label.setText(f"{prob * 100:.2f} %")
        self.target_prob_bar.setValue(int(prob * 100))

        if result.get("success"):
            self._on_success(img_np, result.get("grad_info", {}), prob)

    def _on_success(
        self,
        img_np: np.ndarray | None,
        grad_info: dict,
        prob: float,
    ) -> None:
        target_idx  = self.target_spin.value()    # FIX: was self._target_spin
        target_name = self.classifier.categories[target_idx]

        self._success_label.setText(
            f"✓ Success!  '{target_name}'  {prob * 100:.1f}%"
        )
        logger.info(
            "ADVERSARIAL SUCCESS — target: [%d] %s  prob=%.2f%%",
            target_idx, target_name, prob * 100,
        )

        for name, grad in grad_info.items():
            logger.info("  Final gradient  %s: %s", name, grad.numpy())

        if img_np is not None:
            save_path = f"adversarial_{target_name.replace(' ', '_')}.png"
            # FIX: was self._save_numpy_image() — inline the save directly
            Image.fromarray(
                (img_np * 255).clip(0, 255).astype(np.uint8)
            ).save(save_path)
            self._adv_save_path = save_path
            self._saved_label.setText(f"Saved → {save_path}")
            self.status_label.setText(           # FIX: was self.status_bar.showMessage
                f"Success! Adversarial image saved to {save_path}"
            )

    def _on_opt_finished(self) -> None:
        self.btn_start.setEnabled(True)           # FIX: was self._start_btn
        self.btn_stop.setEnabled(False)           # FIX: was self._stop_btn

        if not self._success_label.text():
            self.status_label.setText(           # FIX: was self.status_bar.showMessage
                "Optimisation complete (target threshold not reached)."
            )
        logger.info("Optimisation routine finished.")