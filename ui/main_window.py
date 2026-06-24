from PyQt6.QtWidgets import QMainWindow

import config
from core.renderer   import MeshScene
from core.classifier import ImageClassifier

from ui.mixins.panels           import PanelsMixin
from ui.mixins.actions          import ActionsMixin
from ui.mixins.slider_callbacks import SliderCallbacksMixin
from ui.mixins.render_view      import RenderViewMixin
from ui.mixins.optimisation     import OptimisationMixin


class MainWindow(
    PanelsMixin,
    ActionsMixin,
    SliderCallbacksMixin,
    RenderViewMixin,
    OptimisationMixin,
    QMainWindow,
):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.setWindowTitle("Adversarial 3D Mesh Renderer")

        # ── Dropdown / render state ────────────────────────────────────────────
        self._static_image_path:      str | None = None   # None = live render
        self._current_3d_model_index: int        = 0
        self._last_render:            object      = None  # numpy (H,W,3) or None

        # ── Optimiser state (expected by OptimisationMixin) ────────────────────
        self.optimiser  = None
        self.opt_thread = None

        # ── Core objects ───────────────────────────────────────────────────────
        # MeshScene.__init__ calls self.load_mesh(OBJ_PATH) internally —
        # do NOT load the mesh a second time here.
        self.scene      = MeshScene(meshes=None, device=device)
        self.classifier = ImageClassifier(model_name=config.MODEL_NAME)

        # ── Build the widget tree and wire signals ─────────────────────────────
        self._build_ui()
        self._connect_actions()

        # First render + initial classification
        self._render_and_display()