import numpy as np

from renderer import DifferentiableScene


def get_property(scene: DifferentiableScene, prop: str):
    if prop == "POSITION":
        return scene.get_position()
    if prop == "ROTATION":
        return scene.get_rotation_deg()
    if prop == "POSE":
        return scene.get_position(), scene.get_rotation_deg()
    if prop == "LIGHTING":
        return scene.get_lighting()
    if prop == "CLOTHING":
        return scene.get_material_color("shirt")
    raise ValueError(prop)


def set_property(scene: DifferentiableScene, prop: str, value) -> None:
    if prop == "POSITION":
        scene.set_position(*value)
    elif prop == "ROTATION":
        scene.set_rotation_deg(*value)
    elif prop == "POSE":
        position, rotation = value
        # PyTorch3D projects bounds using the current orientation.
        scene.set_rotation_deg(*rotation)
        scene.set_position(*position)
    elif prop == "LIGHTING":
        scene.set_lighting(float(value))
    elif prop == "CLOTHING":
        scene.set_material_color("shirt", tuple(value))
    else:
        raise ValueError(prop)


def format_property(prop: str, value) -> str:
    if prop == "POSE":
        position, rotation = value
        return (
            f"position=({position[0]:+.3f}, {position[1]:+.3f}, {position[2]:+.3f}) "
            f"rotation=({rotation[0]:+.1f}°, {rotation[1]:+.1f}°, {rotation[2]:+.1f}°)"
        )
    if prop in ("POSITION", "ROTATION", "CLOTHING"):
        a, b, c = np.asarray(value).tolist()
        return f"{prop.lower()}=({a:+.3f}, {b:+.3f}, {c:+.3f})"
    return f"{prop.lower()}=({float(value):+.3f})"
