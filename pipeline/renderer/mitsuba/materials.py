from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

from PIL import Image


# ---- textured background helpers --------------------------------------
def read_materials(obj_path: Path) -> dict:
    """material -> {"kd": diffuse texture, "alpha": opacity map or None}, from the .mtl."""
    with obj_path.open(encoding="utf-8", errors="replace") as f:
        mtl_files = [l.split(maxsplit=1)[1].strip() for l in f if l.startswith("mtllib ")]

    def resolve(rest, base):
        """Resolve a material texture path, including paths with map options."""
        rest = rest.strip()
        p = (base / rest).resolve()
        if p.is_file():
            return p
        p = (base / rest.split()[-1]).resolve()  # handles "-s 1 1 1 tex.png"
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
    return {n: m for n, m in materials.items() if m["kd"]}


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
    """Build a Mitsuba material for a diffuse texture and optional alpha mask."""
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


