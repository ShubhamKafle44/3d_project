
HUMAN_PARTS = {
    "body": "assets/3D_MODELS/human/body.obj",
    "shirt": "assets/3D_MODELS/human/shirt.obj",
    "pants": "assets/3D_MODELS/human/pants.obj",
    "shoes_and_eyelash": "assets/3D_MODELS/human/shoes_and_eyelash.obj",
}

# Optional background/scene geometry (room, floor, props). None = plain background.
BACKGROUND_PATH = "assets/3D_ENVIRONMENTS/studio/environment.obj"


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
IMAGE_SIZE = 1024
MITSUBA_SPP = 2

# Mitsuba variant: use "llvm_ad_rgb" for CPU or "cuda_ad_rgb" for GPU
MITSUBA_VARIANT = "cuda_ad_rgb"

CAMERA = {
    "distance": 8.5,
    "elevation": 10.0,
    "azimuth": -25.0,
    "target": (0.0, 0.9, 0.0),
    "fov": 45.0,
}

LIGHT = {
    "intensity": 30.0,
    "position": (-2.0, 4.2, -19.07),
    "color": (1.0, 0.72, 0.38),
}

FILL_LIGHT = {
    "intensity": 2.0,
    "position": (-2.0, 5.0, 5.0),
    "color": (1.0, 0.82, 0.65),
}

# Uniform, direction-independent illumination, separate from the point lights.
AMBIENT_LIGHT = {
    "intensity": 0.25,
    "color": (1.0, 1.0, 1.0),
}

DEFAULT_MATERIAL_COLORS = {
    "body": (0.9, 0.75, 0.65),
    "shirt": (0.2, 0.4, 0.8),
    "pants": (0.15, 0.15, 0.15),
    "shoes_and_eyelash": (0.08, 0.06, 0.04),
}

# --------------------------------------------------------------------------
# Adversarial search
# --------------------------------------------------------------------------
SEARCH = {
    "epochs": 100,
    "success_threshold": 0.05,      
    "gradient_learning_rate": 0.06,
    "gradient_validate_every": 5,
    "gradient_image_size": 512,
    "gradient_detector_size": 384,
    "gradient_person_loss_weight": 1.0,
    "gradient_rpn_loss_weight": 0.25,
}

PROPERTY_BOUNDS = {
    "ROTATION": (0.0, 360.0),
    "LIGHTING": (0.05, 3.0),
    "CLOTHING": (0.0, 1.0),
}

POSITION_COLLISION_CLEARANCE = 0.03

POSITION_FRAME_MARGIN_PX = 32
