
import os

HUMAN_PARTS = {
    "body": "assets/human/body.obj",
    "shirt": "assets/human/shirt.obj",
    "pants": "assets/human/pants.obj",
}

# Optional background/scene geometry (room, floor, props). None = plain background.

# Multi-material road scene.  Keep this enabled so the render includes visual
# context instead of falling back to a human-only close-up.
BACKGROUND_PATH = "assets/environment/scene.obj"

# Optional environment map used only by the Mitsuba backend for image-based lighting.
ENV_MAP_PATH = None  # e.g. "assets/scene/env.hdr"

# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
IMAGE_SIZE = 2048
# Alpha-cutout foliage needs multiple depth layers in PyTorch3D.  Rendering
# those layers at 2048px exceeds the 6 GB GPU target, so use the same compact
# size as the gradient attack for normal PyTorch3D runs.
PYTORCH3D_IMAGE_SIZE = 512
# The attack runs at a smaller resolution to keep detector backpropagation in
# GPU memory. Its winning parameters are rendered once more at this size for
# the image written to disk.
FINAL_RENDER_IMAGE_SIZE = 2048
# Background maps are packed into an atlas. Keep the attack atlas compact,
# while preserving source detail in the final render.
FINAL_TEXTURE_ATLAS_HEIGHT = 2048
# Mitsuba's OptiX path is substantially more memory hungry than PyTorch3D.
# Keep its normal (non-gradient) renders within a 6 GB GPU budget.
MITSUBA_IMAGE_SIZE = 1024
MITSUBA_SPP = 8
DEVICE = "cuda" if os.environ.get("FORCE_CPU") != "1" else "cpu"

CAMERA = {
    "distance": 8.5,
    "elevation": 10.0,
    "azimuth": -8.0,
    "target": (0.0, 0.9, 0.0),
    "fov": 45.0,
}

LIGHT = {
    # OBJ/MTL has no light entities. These coordinates are the centre of the
    # exported street-lamp bulb mesh (Object_4.001 / Bola_lampu.001).
    # Clear the bulb mesh by more than half a scene unit.  A tiny offset can
    # still leave the mathematical point source inside its enclosing faces,
    # causing zero-distance light evaluations in Mitsuba.
    "intensity": 20.0,
    "position": (-5.0, 4.2, -19.07),
    "color": (1.0, 0.72, 0.38),
}

# This is a direct point-light fill, not a world/ambient illumination.  The
# exported lamp is occluded from the camera-facing part of this scene, so a
# small front fill prevents the render from becoming pure black.
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
    "target_coverage": (0.20, 0.45),  # auto-zoom band, fraction of frame
    "max_zoom_iters": 10,
    "min_cam_distance": 1.0,
    "gradient_learning_rate": 0.03,
    "gradient_validate_every": 10,
    # Compact optimization sizes keep 3D+detector backprop viable on a 6 GB
    # GPU.  The normal detector resolution is retained for validation.
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
    "CAMERA": (-1.0, 1.0),  # relative nudge to elevation/azimuth in degrees*10
}

# Keep this much space between the subject bounding box and background meshes.
# Position proposals that would intersect the static environment are discarded.
POSITION_COLLISION_CLEARANCE = 0.03
