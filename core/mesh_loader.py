import torch
from pathlib import Path
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesAtlas, TexturesVertex


def normalize_mesh(verts: torch.Tensor):
    """Scale a single mesh's verts to fit within a unit-ish cube centered
    at the origin. NOTE: only use this on ONE mesh at a time (e.g. a
    single standalone prop). Do NOT call this separately on multiple
    parts of the same model (body/shirt/pants) — it will rescale each
    part independently and destroy their relative proportions. For
    multi-part models, normalize all parts together using a single
    shared bounding box instead.
    """
    vmin = verts.min(dim=0).values
    vmax = verts.max(dim=0).values
    center = (vmin + vmax) / 2
    verts = verts - center
    scale = (vmax - vmin).max()
    verts = verts / scale
    verts = verts * 4
    return verts


def load_mesh_from_file(obj_path: str, device: torch.device) -> Meshes:
    """Load an OBJ with multiple textures from a .mtl file."""
    obj_path_obj = Path(obj_path)

    verts, faces_idx, aux = load_obj(
        str(obj_path_obj),
        load_textures=True,
        create_texture_atlas=True,
        texture_atlas_size=16,  # more 'pixels' so textures don't look blurry
        device=device,
    )

    verts = normalize_mesh(verts)

    if aux.texture_atlas is not None:
        textures = TexturesAtlas(atlas=[aux.texture_atlas.to(device)])
        print(f"Successfully loaded textures for: {obj_path_obj.name}")
    else:
        print(f"WARNING: No textures found for {obj_path_obj.name}.")
        print(f"Ensure .mtl and .png files are in {obj_path_obj.parent}")
        verts_rgb = torch.ones_like(verts)[None]
        textures = TexturesVertex(verts_features=verts_rgb.to(device))

    return Meshes(
        verts=[verts.to(device)],
        faces=[faces_idx.verts_idx.to(device)],
        textures=textures,
    )