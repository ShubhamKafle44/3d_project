from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import shlex

from PIL import Image, ImageChops


# ---- textured background helpers --------------------------------------
def read_materials(obj_path: Path) -> dict:
    """Read diffuse and opacity properties for every material in an OBJ's MTL."""
    with obj_path.open(encoding="utf-8", errors="replace") as f:
        mtl_files = [l.split(maxsplit=1)[1].strip() for l in f if l.startswith("mtllib ")]

    def resolve(rest, base):
        """Resolve a material texture path, including paths with map options."""
        words = shlex.split(rest.strip())
        if not words:
            return None
        # MTL map directives may include options before the filename. Prefer
        # the complete path, then the final token for common -s/-o options.
        p = (base / " ".join(words)).resolve()
        if p.is_file():
            return p
        p = (base / words[-1]).resolve()
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
                materials[current] = {
                    "kd": None, "alpha": None, "opacity": 1.0,
                    "color": [0.8, 0.8, 0.8],
                }
            elif key == "map_Kd" and current:
                materials[current]["kd"] = resolve(rest, obj_path.parent)
            elif key == "map_d" and current:
                materials[current]["alpha"] = resolve(rest, obj_path.parent)
            elif key == "Kd" and current:
                try:
                    materials[current]["color"] = [
                        max(0.0, min(1.0, float(value))) for value in rest.split()[:3]
                    ]
                except ValueError:
                    pass
            elif key in {"d", "Tr"} and current:
                try:
                    value = max(0.0, min(1.0, float(rest.split()[0])))
                    materials[current]["opacity"] = value if key == "d" else 1.0 - value
                except (ValueError, IndexError):
                    pass
    return materials


def split_materials(obj_path: Path, textures: dict, output_dir: Path):
    """Write one OBJ per material so each texture can use its own BSDF."""
    paths = {name: output_dir / f"material_{i}.obj" for i, name in enumerate([None, *textures])}
    counts = dict.fromkeys(paths, 0)
    current_material = None

    with ExitStack() as stack:
        outputs = {n: stack.enter_context(p.open("w", encoding="utf-8")) for n, p in paths.items()}
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


def texture_bsdf(info: dict, temp_dir: Path) -> dict:
    """Build a diffuse BSDF and combine MTL, map_d, and diffuse-image alpha."""
    if info["kd"]:
        diffuse = {"type": "bitmap", "filename": str(info["kd"])}
        with Image.open(info["kd"]) as image:
            alpha = image.convert("RGBA").getchannel("A") if "A" in image.getbands() else None
            size = image.size
    else:
        diffuse = {"type": "rgb", "value": info.get("color", [0.8, 0.8, 0.8])}
        alpha = None
        size = None

    opacity = float(info.get("opacity", 1.0))
    if info.get("alpha"):
        with Image.open(info["alpha"]) as image:
            opacity_map = image.convert("L")
        if size and opacity_map.size != size:
            opacity_map = opacity_map.resize(size, Image.Resampling.BILINEAR)
        alpha = opacity_map if alpha is None else ImageChops.multiply(alpha, opacity_map)

    bsdf = {
        "type": "twosided",
        "nested_bsdf": {
            "type": "diffuse",
            "reflectance": diffuse,
        },
    }
    if opacity < 1.0 or alpha is not None:
        if alpha is None:
            opacity_value = opacity
        else:
            if opacity < 1.0:
                alpha = alpha.point(lambda value: round(value * opacity))
            alpha_path = temp_dir / f"opacity_{len(list(temp_dir.glob('opacity_*.png')))}.png"
            alpha.save(alpha_path)
            opacity_value = {"type": "bitmap", "filename": str(alpha_path), "raw": True}
        bsdf = {"type": "mask", "opacity": opacity_value, "nested_bsdf": bsdf}
    return bsdf


def glass_bsdf(info: dict) -> dict:
    """Build a thin glass BSDF, using the material's diffuse map as its tint."""
    transmittance = (
        {"type": "bitmap", "filename": str(info["kd"])}
        if info.get("kd")
        else {"type": "rgb", "value": info.get("color", [1.0, 1.0, 1.0])}
    )
    return {
        "type": "thindielectric",
        "int_ior": 1.52,
        "ext_ior": 1.000277,
        "specular_transmittance": transmittance,
    }


def material_bsdf(name: str, info: dict, temp_dir: Path) -> dict:
    """Choose a renderer-native glass material for OBJ materials named as glass."""
    if "glass" in name.casefold():
        return glass_bsdf(info)
    return texture_bsdf(info, temp_dir)


