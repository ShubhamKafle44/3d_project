from pathlib import Path

import numpy as np
import config
from renderer import DifferentiableScene, build_scene
from renderer.collision import obj_bounds


PIPELINE_DIR = Path(__file__).resolve().parent
MODEL_DIR = PIPELINE_DIR / "assets" / "3D_MODELS"
ENVIRONMENT_DIR = PIPELINE_DIR / "assets" / "3D_ENVIRONMENTS"


def _resolve_path(reference: str, asset_root: Path) -> Path:
    """Resolve an explicit path first, then a name within an asset library."""
    path = Path(reference).expanduser()
    if path.exists():
        return path.resolve()
    library_path = asset_root / reference
    if library_path.exists():
        return library_path.resolve()
    if not library_path.suffix:
        obj_path = library_path.with_suffix(".obj")
        if obj_path.is_file():
            return obj_path.resolve()
    raise FileNotFoundError(
        f"Asset {reference!r} was not found as a path or under {asset_root}"
    )


def _model_parts(model: str | None) -> dict[str, str]:
    """Return named OBJ parts for a built-in model, OBJ file, or model folder."""
    if model in (None, "human"):
        return {
            name: str((PIPELINE_DIR / path).resolve())
            for name, path in config.HUMAN_PARTS.items()
        }

    path = _resolve_path(model, MODEL_DIR)
    if path.is_file():
        if path.suffix.lower() != ".obj":
            raise ValueError(f"Only OBJ model files are supported: {path}")
        obj_files = [path]
    elif path.is_dir():
        obj_files = sorted(item for item in path.rglob("*") if item.suffix.lower() == ".obj")
    else:
        raise ValueError(f"Model must be an OBJ file or directory: {path}")

    if not obj_files:
        raise ValueError(f"No OBJ model files found in {path}")

    parts = {}
    for obj_path in obj_files:
        # Use the stem as a useful material/part name (so a part named
        # "shirt" continues to work with the optional appearance attack).
        name = obj_path.stem
        if name in parts:
            relative_name = obj_path.relative_to(path).with_suffix("") if path.is_dir() else obj_path
            name = "_".join(relative_name.parts)
        if name in parts:
            raise ValueError(f"Model contains duplicate part names: {name!r}")
        parts[name] = str(obj_path)
    return parts


def _environment_path(environment: str | None) -> str | None:
    """Resolve an optional single-OBJ environment, or None for a plain scene."""
    if environment is None or environment.lower() in {"none", "off"}:
        return None
    if environment == "studio":
        return str((PIPELINE_DIR / config.BACKGROUND_PATH).resolve())

    path = _resolve_path(environment, ENVIRONMENT_DIR)
    if path.is_dir():
        obj_files = sorted(item for item in path.rglob("*") if item.suffix.lower() == ".obj")
        preferred = path / "environment.obj"
        if preferred.is_file():
            return str(preferred)
        if len(obj_files) == 1:
            return str(obj_files[0])
        if len(obj_files) > 1:
            raise ValueError(
                f"Environment folder {path} contains multiple OBJ files; "
                "select the environment OBJ file directly."
            )
        raise ValueError(f"No OBJ environment file found in {path}")
    if path.is_file() and path.suffix.lower() == ".obj":
        return str(path)
    raise ValueError(f"Environment must be an OBJ file, folder, or 'none': {path}")


def available_assets() -> tuple[list[str], list[str]]:
    """List built-in names and top-level entries selectable from the CLI."""
    def entries(root: Path) -> list[str]:
        if not root.is_dir():
            return []
        return sorted(
            path.name for path in root.iterdir()
            if path.is_dir() or (path.is_file() and path.suffix.lower() == ".obj")
        )

    models = list(dict.fromkeys(["human", *entries(MODEL_DIR)]))
    environments = list(dict.fromkeys(["studio", "none", *entries(ENVIRONMENT_DIR)]))
    return models, environments


def _camera_for_model(parts: dict[str, str], is_default_model: bool) -> dict:
    """Keep the tuned default-human view, and fit custom model bounds to view."""
    camera = dict(config.CAMERA)
    if is_default_model:
        return camera

    bounds = [obj_bounds(path) for path in parts.values()]
    lower = np.min([bound[0] for bound in bounds], axis=0)
    upper = np.max([bound[1] for bound in bounds], axis=0)
    center = (lower + upper) / 2.0
    radius = float(np.linalg.norm(upper - lower) / 2.0)
    half_fov = np.radians(float(camera["fov"]) / 2.0)
    camera["target"] = tuple(float(value) for value in center)
    camera["distance"] = max(radius / np.sin(half_fov) * 1.15, 0.1)
    return camera


def build_human_scene(
    backend: str,
    device: str,
    image_size: int | None = None,
    model: str | None = "human",
    environment: str | None = "studio",
) -> DifferentiableScene:
    scene = build_scene(
        backend, device=device, image_size=image_size or config.IMAGE_SIZE
    )

    parts = _model_parts(model)
    for part_name, part_path in parts.items():
        scene.load_mesh(part_path, name=part_name)

    background_path = _environment_path(environment)
    if background_path:
        scene.load_background(background_path)

    for part_name, color in config.DEFAULT_MATERIAL_COLORS.items():
        if part_name in parts:
            scene.set_material_color(part_name, color)

    scene.set_camera_orbit(**_camera_for_model(parts, model in (None, "human")))

    light_intensity = (
        config.AMBIENT_LIGHT["intensity"]
        if backend == "mitsuba"
        else config.LIGHT["intensity"]
    )
    scene.set_lighting(light_intensity)
    return scene
