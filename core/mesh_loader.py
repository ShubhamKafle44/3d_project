import torch
from pathlib import Path
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesAtlas
from pytorch3d.renderer import TexturesVertex


def load_mesh_from_file(obj_path: str, device: torch.device) -> Meshes:
    """Load an OBJ with multiple textures from a .mtl file."""
    
    # Convert the string path into a Path object to make it easier to handle file extensions and parents
    obj_path_obj = Path(obj_path)
    
    # 1. Load the OBJ 
    # This reads the 3D coordinates (verts) and how they connect to form surfaces (faces).
    # 'aux' holds extra info like textures defined in the companion .mtl file.
    # We increase texture_atlas_size to 16 for better fidelity on the Jeep
    verts, faces_idx, aux = load_obj(
        str(obj_path_obj), 
        load_textures=True, 
        create_texture_atlas=True, 
        texture_atlas_size=16, # Gives more 'pixels' to the textures so they don't look blurry
        device=device
    )
    
    # Check if textures were actually loaded from the .mtl
    # PyTorch3D stores complex textures in an 'atlas' (like a giant sticker sheet for the 3D model)
    if aux.texture_atlas is not None:
        # If textures exist, wrap them in a TexturesAtlas object so the renderer can use them
        textures = TexturesAtlas(atlas=[aux.texture_atlas.to(device)])
        print(f"Successfully loaded textures for: {obj_path_obj.name}")
    else:
        # If the textures are missing, the model would usually just look invisible or black
        # This fallback prevents a crash and gives you a hint on how to fix the file paths
        print(f"WARNING: No textures found for {obj_path_obj.name}.")
        print(f"Ensure .mtl and .png files are in {obj_path_obj.parent}")
        
        # Create a plain white fallback texture so the model is at least visible as a solid shape
        # torch.ones_like makes a tensor of the same size as the vertices, but filled with 1s (white)
        verts_rgb = torch.ones_like(verts)[None]  
        textures = TexturesVertex(verts_features=verts_rgb.to(device))

    # Finally, package everything—the points, the triangles, and the colors—into a Meshes object
    return Meshes(
        verts=[verts.to(device)], 
        faces=[faces_idx.verts_idx.to(device)], 
        textures=textures
    )