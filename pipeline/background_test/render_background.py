from pathlib import Path
import math
import os
from tempfile import TemporaryDirectory
from contextlib import ExitStack
import sys

import mitsuba as mi
import numpy as np
from PIL import Image

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import config

mi.set_variant(os.environ.get("MITSUBA_VARIANT", "cuda_ad_rgb"))

CAMERA = config.CAMERA


def compute_camera(cam_config):
    """Same orbit-camera math as MitsubaScene._camera_origin, driven by config.CAMERA."""
    elev_rad = np.radians(cam_config["elevation"])
    azim_rad = np.radians(cam_config["azimuth"])
    d = cam_config["distance"]
    tx, ty, tz = cam_config["target"]

    x = tx + d * np.cos(elev_rad) * np.sin(azim_rad)
    y = ty + d * np.sin(elev_rad)
    z = tz + d * np.cos(elev_rad) * np.cos(azim_rad)

    origin = [float(x), float(y), float(z)]
    return origin, list(cam_config["target"])


def read_textures(obj_path):
    with obj_path.open(encoding="utf-8", errors="replace") as obj_file:
        mtl_files = [line.split(maxsplit=1)[1].strip() for line in obj_file if line.startswith("mtllib ")]
    textures = {}
    for mtl_file in mtl_files:
        material = None
        mtl_path = obj_path.parent / mtl_file
        if not mtl_path.is_file():
            continue
        for line in mtl_path.open(encoding="utf-8", errors="replace"):
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            if parts[0] == "newmtl":
                material = parts[1]
            elif parts[0] == "map_Kd" and material:
                texture = (obj_path.parent / parts[1]).resolve()
                if texture.is_file():
                    textures[material] = texture
    return textures


def split_materials(obj_path, textures, output_dir):
    paths = {name: output_dir / f"material_{index}.obj" for index, name in enumerate([None, *textures])}
    counts = dict.fromkeys(paths, 0)
    current_material = None

    with ExitStack() as stack:
        outputs = {name: stack.enter_context(path.open("w", encoding="utf-8")) for name, path in paths.items()}
        with obj_path.open(encoding="utf-8", errors="replace") as source:
            for line in source:
                parts = line.split()
                if len(parts) > 1 and parts[0] == "usemtl":
                    current_material = parts[1]
                    continue
                if parts and parts[0] == "mtllib":
                    continue
                if parts and parts[0] == "f":
                    group = current_material if current_material in textures else None
                    outputs[group].write(line)
                    counts[group] += 1
                else:
                    for output in outputs.values():
                        output.write(line)
    return [(name, paths[name]) for name, count in counts.items() if count]

def read_materials(obj_path):
    """material -> {"kd": diffuse texture, "alpha": opacity map or None}, from the .mtl."""
    with obj_path.open(encoding="utf-8", errors="replace") as f:
        mtl_files = [l.split(maxsplit=1)[1].strip() for l in f if l.startswith("mtllib ")]

    def resolve(rest, base):
        rest = rest.strip()
        p = (base / rest).resolve()
        if p.is_file():                      # handles spaces in filenames
            return p
        p = (base / rest.split()[-1]).resolve()   # handles "-s 1 1 1 tex.png" options
        return p if p.is_file() else None

    materials, current = {}, None
    for mtl_file in mtl_files:
        mtl_path = obj_path.parent / mtl_file
        if not mtl_path.is_file():
            continue
        for line in mtl_path.open(encoding="utf-8", errors="replace"):
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            key, rest = parts
            if key == "newmtl":
                current = rest.strip()
                materials[current] = {"kd": None, "alpha": None}
            elif key == "map_Kd" and current:
                materials[current]["kd"] = resolve(rest, obj_path.parent)
            elif key == "map_d" and current:
                materials[current]["alpha"] = resolve(rest, obj_path.parent)

    for name, m in materials.items():
        print(f"material {name!r}: kd={m['kd'] and m['kd'].name}  alpha={m['alpha'] and m['alpha'].name}")
    return {n: m for n, m in materials.items() if m["kd"]}


def texture_bsdf(info, temp_dir):
    bsdf = {
        "type": "twosided",
        "nested_bsdf": {
            "type": "diffuse",
            "reflectance": {"type": "bitmap", "filename": str(info["kd"])},
        },
    }
    if info["alpha"]:
        img = Image.open(info["alpha"])
        ch = img.convert("RGBA").getchannel("A") if "A" in img.getbands() else img.convert("L")
        alpha_path = temp_dir / f"{info['alpha'].parent.name}_{info['alpha'].stem}_alpha.png"
        ch.save(alpha_path)
        bsdf = {
            "type": "mask",
            "opacity": {"type": "bitmap", "filename": str(alpha_path), "raw": True},
            "nested_bsdf": bsdf,
        }
    return bsdf


def render_scene(obj_path, output_image_path="render.png"):
    obj_path = Path(obj_path).resolve()
    cam_origin, cam_target = compute_camera(CAMERA)

    print(f"Calculated Camera Origin: {cam_origin}")
    print(f"Camera Target: {cam_target}")

    scene_dict = {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 6},
        "sensor": {
            "type": "perspective",
            "fov": CAMERA["fov"],
            "to_world": mi.ScalarTransform4f.look_at(
                origin=cam_origin,
                target=cam_target,
                up=[0, 1, 0],
            ),
            "sampler": {"type": "independent", "sample_count": 64},
            "film": {
                "type": "hdrfilm",
                "width": 1280,
                "height": 720,
                "rfilter": {"type": "gaussian"},
            },
        },
        # Lights below are taken from config, built the same way as in MitsubaScene
        "light": {
            "type": "constant",
            "radiance": {
                "type": "rgb",
                "value": [
                    math.pi * float(config.AMBIENT_LIGHT["intensity"]) * float(c)
                    for c in config.AMBIENT_LIGHT["color"]
                ],
            },
        },
        "street_light": {
            "type": "point",
            "position": list(config.LIGHT["position"]),
            "intensity": {
                "type": "rgb",
                "value": [
                    20.0 * float(config.LIGHT["intensity"]) * float(c)
                    for c in config.LIGHT["color"]
                ],
            },
        },
        "fill_light": {
            "type": "point",
            "position": list(config.FILL_LIGHT["position"]),
            "intensity": {
                "type": "rgb",
                "value": [
                    20.0 * float(config.FILL_LIGHT["intensity"]) * float(c)
                    for c in config.FILL_LIGHT["color"]
                ],
            },
        },
    }

    with TemporaryDirectory(prefix="mitsuba_background_") as temp:
        temp_dir = Path(temp)
        textures = read_materials(obj_path)
        for index, (material, mesh_path) in enumerate(split_materials(obj_path, textures, temp_dir)):
            shape = {
                "type": "obj",
                "filename": str(mesh_path),
                "face_normals": True,
                "bsdf": (
                    texture_bsdf(textures[material], temp_dir)
                    if material is not None
                    else {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.5] * 3}}
                ),
            }
            scene_dict[f"background_{index}"] = shape
        image = mi.render(mi.load_dict(scene_dict))

    # Convert linear HDR values to sRGB UInt8 bitmap
    bitmap = mi.Bitmap(image).convert(
        pixel_format=mi.Bitmap.PixelFormat.RGB,
        component_format=mi.Struct.Type.UInt8,
        srgb_gamma=True,
    )
    bitmap.write(output_image_path)
    print(f"Successfully rendered to {output_image_path}")

if __name__ == "__main__":
    OBJ_FILE = PIPELINE_DIR / "assets/studio/environment.obj"
    render_scene(OBJ_FILE)
