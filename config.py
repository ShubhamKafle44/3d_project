import torch

# ── Device ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ── Paths ──────────────────────────────────────────────────────────────────────
OBJ_PATH        = "./models/Jeep/Jeep.obj"
BACKGROUND_PATH = "./background/background1.jpeg"

# ── Renderer ───────────────────────────────────────────────────────────────────
IMAGE_SIZE   = 512
CAMERA_DIST  = 8.0
CAMERA_ELEV  = 10.0
CAMERA_AZIM  = 0.0

# ── Adversarial optimisation ───────────────────────────────────────────────────
SUCCESS_THRESHOLD = 0.80

# ── Default vision model ───────────────────────────────────────────────────────
MODEL_NAME = "vit_b_16"

# ── Registry: available 3-D models ────────────────────────────────────────────
# Each entry: (display_label, path_to_obj)
# The FIRST entry should match OBJ_PATH (it is selected by default at startup).
AVAILABLE_3D_MODELS = [
    ("Jeep",   "./models/Jeep/Jeep.obj"),
    ("Chair",  "./models/Chair/Chair.obj"),
    ("Plane",  "./models/Airbus/Airbus A310.obj"),
    ("Person", "./models/Person/Person.obj")
]

# ── Registry: available input images ──────────────────────────────────────────
AVAILABLE_IMAGES = [
    ("Background 1", "./background/background1.jpeg"),
    ("Background 2", "./background/background2.jpeg"),
]

# ── Registry: available vision / classifier models ────────────────────────────
# Each entry: (display_label, torchvision_model_key)
# Keys must match the _MODEL_FACTORY dict in core/classifier.py.
AVAILABLE_VISION_MODELS = [
    ("ViT-B/16 (default)",  "vit_b_16"),
    ("ResNet-50",           "resnet50"),
    ("MobileNet-V3-Large",  "mobilenet_v3_large"),
]