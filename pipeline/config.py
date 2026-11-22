DETECTOR_NAMES = (
    "fasterrcnn_resnet50_fpn_v2",
    "yolo26s",
    "yolo26m",
    "yoloe-26s-seg",
    "yolo26n",
    "yolo26x",
    "yolo26l",
)

# Square render/input size used by each detector. Faster R-CNN's standard
# transform resizes a square image to its 800 px minimum side.
DETECTOR_INPUT_SIZES = {
    "fasterrcnn_resnet50_fpn_v2": 800,
    "yolo26n": 640,
    "yolo26s": 640,
    "yolo26m": 640,
    "yolo26l": 640,
    "yolo26x": 640,
    "yoloe-26s-seg": 640,
}
DETECTOR_CONFIDENCE_THRESHOLD = 0.05

# Optional background/scene geometry (room, floor, props). None = plain background.
BACKGROUND_PATH = "assets/3D_ENVIRONMENTS/studio/environment.obj"


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
MITSUBA_SPP = 64

TEXTURE_ATTACK_RESOLUTION = 256
TEXTURE_ATTACK_NOISE_STD = 0.30
TEXTURE_ATTACK_NOISE_SEED = 7
# Penalize mean-squared pixel changes from the original texture during search.
TEXTURE_ORIGINAL_PENALTY_WEIGHT = 10.0

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
    "position": (-2.0, 4.2, 5.0),
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
    "gradient_learning_rate": 0.01,
    "gradient_validate_every": 1,
    "gradient_snapshot_every": 10,
}

PROPERTY_BOUNDS = {
    "ROTATION": (0.0, 360.0),
    "LIGHTING": (0.5, 3.0),
    "TEXTURE": (0.0, 1.0),
}

# Keep optimized point-light positions in a useful region around the subject.
LIGHT_POSITION_BOUNDS = ((-8.0, 8.0), (0.5, 10.0), (-8.0, 8.0))

POSITION_COLLISION_CLEARANCE = 0.03

POSITION_FRAME_MARGIN_PX = 1
