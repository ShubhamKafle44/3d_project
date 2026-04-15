import torch

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ── Paths ─────────────────────────────────────────────────────────────────────
OBJ_PATH        = "./models/Jeep/Jeep.obj"
BACKGROUND_PATH = "./background/background.png"
MODEL_NAME      = "vit_b_16"

# ── Renderer ──────────────────────────────────────────────────────────────────
IMAGE_SIZE   = 512
CAMERA_DIST  = 8.0
CAMERA_ELEV  = 10.0
CAMERA_AZIM  = 0.0

# ── Adversarial optimisation ──────────────────────────────────────────────────
SUCCESS_THRESHOLD = 0.80
