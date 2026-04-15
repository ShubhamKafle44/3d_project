from PyQt6.QtCore import QObject, pyqtSignal

from core.classifier import AdversarialOptimiser


class OptimWorker(QObject):
    """Runs AdversarialOptimiser in a background thread, emitting a signal per step."""

    # These signals act like "phone calls" back to the main UI thread 
    # so we can update the screen without crashing the app.
    step_done = pyqtSignal(dict)
    finished  = pyqtSignal()

    def __init__(self, optimiser: AdversarialOptimiser, max_steps: int):
        super().__init__()
        # Store our math engine and the limit on how long it should run
        self.optimiser = optimiser
        self.max_steps = max_steps

    def run(self) -> None:
        """The main loop entry point when the thread starts."""
        # We 'hook' the optimizer's step event directly to our signal.
        # This means every time the math finishes one iteration, it automatically
        # triggers step_done.emit() with the latest data.
        self.optimiser.on_step = self.step_done.emit
        
        # Start the actual heavy lifting (the optimization loop)
        self.optimiser.run(self.max_steps)
        
        # Tell the UI we're all done so it can re-enable buttons or show a final message
        self.finished.emit()