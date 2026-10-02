import config
from renderer import DifferentiableScene, build_scene


def build_human_scene(
    backend: str, device: str, image_size: int | None = None
) -> DifferentiableScene:
    scene = build_scene(
        backend, device=device, image_size=image_size or config.IMAGE_SIZE
    )

    for part_name, part_path in config.HUMAN_PARTS.items():
        scene.load_mesh(part_path, name=part_name)

    if config.BACKGROUND_PATH:
        scene.load_background(config.BACKGROUND_PATH)

    for part_name, color in config.DEFAULT_MATERIAL_COLORS.items():
        if part_name in config.HUMAN_PARTS:
            scene.set_material_color(part_name, color)

    scene.set_camera_orbit(**config.CAMERA)
    # Mitsuba exposes the uniform environment intensity as its lighting
    # parameter; PyTorch3D's shared lighting control scales the key point light.
    light_intensity = (
        config.AMBIENT_LIGHT["intensity"]
        if backend == "mitsuba"
        else config.LIGHT["intensity"]
    )
    scene.set_lighting(light_intensity)
    return scene
