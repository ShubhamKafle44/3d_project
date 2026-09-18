
HUMAN_PARTS = {
    "body": "assets/human/body.obj",
    "shirt": "assets/human/shirt.obj",
    "pants": "assets/human/pants.obj",
}

# Optional background/scene geometry (room, floor, props). None = plain background.
BACKGROUND_PATH = "assets/environment/scene.obj"

# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
IMAGE_SIZE = 2048
PYTORCH3D_IMAGE_SIZE = 512
MITSUBA_IMAGE_SIZE = 1024
MITSUBA_SPP = 8

CAMERA = {
    "distance": 8.5,
    "elevation": 10.0,
    "azimuth": -8.0,
    "target": (0.0, 0.9, 0.0),
    "fov": 45.0,
}

LIGHT = {

    "intensity": 20.0,
    "position": (-5.0, 4.2, -19.07),
    "color": (1.0, 0.72, 0.38),
}

FILL_LIGHT = {
    "intensity": 2.0,
    "position": (-2.0, 5.0, 5.0),
    "color": (1.0, 0.82, 0.65),
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
    "mode": "gradient_appearance",  # "gradient_appearance" (PyTorch3D) or "random"
    "epochs": 100,
    "step_size": 1.0,
    "success_threshold": 0.05,      # stop once human_prob <= this
    "gradient_learning_rate": 0.03,
    "gradient_validate_every": 10,
    "gradient_image_size": 512,
    "gradient_detector_size": 384,
}

# Bounds for each perturbable property, used to clip random search proposals.
PROPERTY_BOUNDS = {
    # Keep a recognisable subject in frame: do not let a POSITION search
    # "succeed" merely by moving the person below or behind the camera.
    "POSITION": ((-0.6, -0.3, -0.4), (0.6, 0.3, 0.4)),
    "ROTATION": (0.0, 360.0),
    "LIGHTING": (0.05, 3.0),
    "CLOTHING": (0.0, 1.0),
}

# Keep this much space between the subject bounding box and background meshes.
# Position proposals that would intersect the static environment are discarded.
POSITION_COLLISION_CLEARANCE = 0.03
