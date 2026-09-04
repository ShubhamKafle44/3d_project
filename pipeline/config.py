
import os

HUMAN_PARTS = {
    "body": "assets/human/body.obj",
    "shirt": "assets/human/shirt.obj",
    "pants": "assets/human/pants.obj",
}

# Optional background/scene geometry (room, floor, props). None = plain background.

# BACKGROUND_PATH = "assets/studio/Street environment_V01.obj"

# BACKGROUND_PATH = "assets/environment/scene.obj"

BACKGROUND_PATH = None

# Optional environment map used only by the Mitsuba backend for image-based lighting.
ENV_MAP_PATH = None  # e.g. "assets/scene/env.hdr"

# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
IMAGE_SIZE = 512
DEVICE = "cuda" if os.environ.get("FORCE_CPU") != "1" else "cpu"

CAMERA = {
    "distance": 10,
    "elevation": 10.0,
    "azimuth": 0.0,
    "target": (0.0, 0.0, 0.0),
    "fov": 10.0,
}

LIGHT = {
    "intensity": 1.0,
    "position": (2.0, 2.0, 2.0),
}

DEFAULT_MATERIAL_COLORS = {
    "body": (0.9, 0.75, 0.65),
    "shirt": (0.2, 0.4, 0.8),
    "pants": (0.15, 0.15, 0.15),
}

# --------------------------------------------------------------------------
# Adversarial search
# --------------------------------------------------------------------------
SEARCH = {
    "epochs": 100,
    "step_size": 0.1,
    "success_threshold": 0.05,      # stop once human_prob <= this
    "target_coverage": (0.20, 0.45),  # auto-zoom band, fraction of frame
    "max_zoom_iters": 10,
    "min_cam_distance": 1.0,
}

# Bounds for each perturbable property, used to clip random search proposals.
PROPERTY_BOUNDS = {
    "POSITION": (-1.5, 1.5),
    "ROTATION": (0.0, 360.0),
    "LIGHTING": (0.05, 3.0),
    "CLOTHING": (0.0, 1.0),
    "CAMERA": (-1.0, 1.0),  # relative nudge to elevation/azimuth in degrees*10
}
