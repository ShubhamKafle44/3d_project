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
    scene.set_lighting(config.LIGHT["intensity"])
    return scene
