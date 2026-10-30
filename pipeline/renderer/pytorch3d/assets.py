from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..collision import obj_bounds, obj_face_bounds
from .dependencies import (
    Meshes, TexturesUV, TexturesVertex, load_obj, load_objs_as_meshes,
)

def _resolve_mtl_texture(mtl_path: Path, value: str) -> Optional[Path]:
    """Resolve an MTL map value containing options and/or a spaced filename."""
    words = value.split()
    for index in range(len(words)):
        candidate = mtl_path.parent / " ".join(words[index:])
        if candidate.is_file():
            return candidate
    return None


def _pytorch3d_safe_obj(path: str) -> str:
    """Cache an OBJ/MTL pair with unsupported MTL texture options removed."""
    source = Path(path)
    obj_lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    mtl_name = next(
        (line.split(maxsplit=1)[1] for line in obj_lines if line.startswith("mtllib ")),
        None,
    )
    if mtl_name is None:
        return str(source)
    mtl_source = source.parent / mtl_name
    if not mtl_source.is_file():
        return str(source)
    cached_obj = source.with_name(f"{source.stem}.pytorch3d_safe{source.suffix}")
    cached_mtl = source.with_name(f"{source.stem}.pytorch3d_safe.mtl")
    if (
        cached_obj.is_file()
        and cached_mtl.is_file()
        and cached_obj.stat().st_mtime >= source.stat().st_mtime
        and cached_mtl.stat().st_mtime >= mtl_source.stat().st_mtime
    ):
        return str(cached_obj)

    clean_mtl = []
    for line in mtl_source.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("map_Kd "):
            texture_path = _resolve_mtl_texture(mtl_source, line[7:])
            if texture_path is not None:
                # PyTorch3D treats MTL options such as ``-s`` as part of the
                # filename. Its loader already supports spaces in filenames.
                line = f"map_Kd {texture_path.name}"
        clean_mtl.append(line)
    cached_mtl.write_text("\n".join(clean_mtl) + "\n", encoding="utf-8")
    cached_obj.write_text(
        "\n".join(
            f"mtllib {cached_mtl.name}" if line.startswith("mtllib ") else line
            for line in obj_lines
        )
        + "\n",
        encoding="utf-8",
    )
    return str(cached_obj)


def _material_alpha_images(obj_path: Path) -> Dict[str, np.ndarray]:
    """Load embedded/map_d alpha masks keyed by OBJ material name."""
    mtl_name = next(
        (
            line.split(maxsplit=1)[1]
            for line in obj_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.startswith("mtllib ")
        ),
        None,
    )
    if mtl_name is None:
        return {}
    mtl_path = obj_path.parent / mtl_name
    if not mtl_path.is_file():
        return {}

    bindings: Dict[str, Dict[str, Path]] = {}
    material = None
    for line in mtl_path.read_text(encoding="utf-8", errors="replace").splitlines():
        tokens = line.split(maxsplit=1)
        if len(tokens) != 2:
            continue
        if tokens[0] == "newmtl":
            material = tokens[1]
        elif material is not None and tokens[0] in {"map_Kd", "map_d"}:
            texture_path = _resolve_mtl_texture(mtl_path, tokens[1])
            if texture_path is not None:
                key = "opacity" if tokens[0] == "map_d" else "diffuse"
                bindings.setdefault(material, {})[key] = texture_path

    alpha_images = {}
    for material, binding in bindings.items():
        for image_path in (binding.get("opacity"), binding.get("diffuse")):
            if image_path is None:
                continue
            try:
                image = Image.open(image_path)
                if "A" not in image.getbands() and "transparency" not in image.info:
                    continue
                alpha_images[material] = (
                    np.asarray(image.convert("RGBA").getchannel("A"), dtype=np.float32)
                    / 255.0
                )
                break
            except OSError:
                continue
    return alpha_images




class AssetLoadingMixin:
    # ---- asset loading ---------------------------------------------
    def load_mesh(self, path: str, name: str = "mesh") -> None:
        # Human parts receive the configured solid colours below, so avoid
        # loading absent source texture files solely to discard them later.
        mesh = load_objs_as_meshes([path], device=self.device, load_textures=False)
        if not mesh.textures:
            verts = mesh.verts_packed()
            white = torch.ones_like(verts)[None]
            mesh.textures = TexturesVertex(verts_features=white)
        self.parts[name] = mesh
        lower, upper = obj_bounds(path)
        self._subject_lower = lower if self._subject_lower is None else np.minimum(self._subject_lower, lower)
        self._subject_upper = upper if self._subject_upper is None else np.maximum(self._subject_upper, upper)

    def load_background(self, path: str) -> None:
        """Load background materials into a padded UV atlas.

        ``TexturesUV`` uses one map for a scene.  Each source map is therefore
        placed in its own atlas tile with replicated-pixel gutters.  The
        gutters keep bilinear sampling inside the correct material, while UVs
        are repeated within their own tile instead of leaking into a neighbor.
        """
        self._background_face_lower, self._background_face_upper = obj_face_bounds(path)
        render_path = _pytorch3d_safe_obj(path)
        verts, faces, aux = load_obj(
            render_path,
            load_textures=True,
            device=self.device,
        )
        texture_images = aux.texture_images or {}
        if not texture_images or faces.textures_idx is None:
            self.background = Meshes(verts=[verts], faces=[faces.verts_idx])
            self._background_is_textured = False
            return

        # Preserve source-map aspect ratios and reserve a neutral first tile
        # for faces with no ``usemtl`` binding (the enclosure in road_pack).
        # The source road map is 8192x4096.  Keeping a 2048px-high atlas for
        # a 768px optimization render needlessly consumes hundreds of MB on
        # CUDA; 512px retains more texture detail than the attack render can
        # resolve.  Use a larger atlas only for full-resolution final renders.
        atlas_height = 512 if self.image_size <= 1024 else 1024
        gutter = 4
        neutral_tile = torch.full(
            (4, atlas_height + 2 * gutter, atlas_height + 2 * gutter),
            0.18,
            dtype=torch.float32,
        )
        neutral_tile[3] = 1.0
        tiles = [neutral_tile]
        content_widths = [atlas_height]
        tile_widths = [atlas_height + 2 * gutter]
        # ``faces.materials_idx`` indexes the distinct ``usemtl`` names in
        # the OBJ, not just the subset whose ``map_Kd`` images loaded.  Keep
        # the name associated with each atlas tile so sparse material indices
        # can be resolved correctly below.
        texture_tile_by_name = {}
        alpha_images = _material_alpha_images(Path(path))
        for material_name, texture in texture_images.items():
            # Resize the large source maps on CPU, then move only the compact
            # atlas to CUDA.  Moving the original 8K road map first causes an
            # unnecessary ~400 MB transient allocation.
            image = texture[..., :3].to(
                device="cpu", dtype=torch.float32
            ).permute(2, 0, 1).unsqueeze(0)
            height, width = texture.shape[:2]
            tile_width = max(1, round(width / height * atlas_height))
            image = F.interpolate(
                image,
                size=(atlas_height, tile_width),
                mode="bilinear",
                align_corners=False,
            )
            alpha = alpha_images.get(material_name)
            if alpha is None:
                alpha_image = torch.ones((1, 1, height, width), dtype=torch.float32)
            else:
                alpha_image = torch.from_numpy(alpha).reshape(1, 1, height, width)
            alpha_image = F.interpolate(
                alpha_image,
                size=(atlas_height, tile_width),
                mode="bilinear",
                align_corners=False,
            )
            image = torch.cat((image, alpha_image), dim=1)
            # Replicate the edge texels around a tile.  Without this gutter,
            # bilinear filtering blends at an atlas boundary with the next
            # material and produces visible road/prop seams.
            tiles.append(F.pad(image[0], (gutter, gutter, gutter, gutter), mode="replicate"))
            content_widths.append(tile_width)
            tile_widths.append(tile_width + 2 * gutter)
            texture_tile_by_name[material_name] = len(content_widths) - 1

        atlas_width = sum(tile_widths)
        texture_atlas = torch.cat(tiles, dim=2).permute(1, 2, 0).to(self.device)

        source_material_ids = faces.materials_idx.to(self.device)
        # PyTorch3D assigns material indices in first-appearance order of the
        # OBJ's ``usemtl`` directives.  A scene commonly has untextured
        # materials interspersed with textured ones, so indexing atlas tiles
        # directly by this ID is invalid (and caused the previous exception).
        material_names = []
        seen_material_names = set()
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            tokens = line.split()
            if len(tokens) >= 2 and tokens[0] == "usemtl":
                material_name = tokens[1]
                if material_name not in seen_material_names:
                    seen_material_names.add(material_name)
                    material_names.append(material_name)
        material_to_tile = torch.zeros(
            len(material_names), dtype=torch.int64, device=self.device
        )
        for index, material_name in enumerate(material_names):
            material_to_tile[index] = texture_tile_by_name.get(material_name, 0)
        valid_material = (source_material_ids >= 0) & (
            source_material_ids < len(material_names)
        )
        # -1 (and any unrecognised material) means no usable texture: render
        # it with the neutral atlas tile rather than failing the whole scene.
        material_ids = torch.zeros_like(source_material_ids)
        material_ids[valid_material] = material_to_tile[source_material_ids[valid_material]]
        widths = torch.tensor(content_widths, dtype=torch.float32, device=self.device)
        padded_widths = torch.tensor(tile_widths, dtype=torch.float32, device=self.device)
        offsets = torch.cat((
            torch.zeros(1, device=self.device),
            padded_widths.cumsum(dim=0)[:-1],
        ))

        # UV indices may be shared by faces that use different materials.
        # Duplicate them per face before applying the material-tile transform.
        source_face_uvs = faces.textures_idx.to(self.device)
        valid_uvs = source_face_uvs >= 0
        # Some untextured OBJ faces have no ``vt`` reference.  ``-1`` would
        # otherwise index the final UV entry; give those faces the neutral
        # tile's centre instead.
        face_uvs = aux.verts_uvs.to(self.device)[source_face_uvs.clamp_min(0)].clone()
        face_uvs[~valid_uvs] = 0.5

        # Do not map an exact 1.0 UV endpoint back to 0.0.  The road is a
        # broad quad with UVs (0, 0) through (1, 1); plain modulo collapses
        # all four corners to one texel and makes the road appear untextured.
        def wrap_uv(uv: torch.Tensor) -> torch.Tensor:
            wrapped = torch.remainder(uv, 1.0)
            is_positive_integer = (uv > 0) & torch.isclose(
                wrapped, torch.zeros_like(wrapped), atol=1e-6, rtol=0.0
            )
            return torch.where(is_positive_integer, torch.ones_like(wrapped), wrapped)

        face_uvs[..., 0] = wrap_uv(face_uvs[..., 0])
        face_uvs[..., 1] = wrap_uv(face_uvs[..., 1])
        face_uvs[..., 0] = (
            face_uvs[..., 0] * widths[material_ids, None]
            + offsets[material_ids, None]
            + gutter
        ) / atlas_width
        face_uvs[..., 1] = (
            face_uvs[..., 1] * atlas_height + gutter
        ) / (atlas_height + 2 * gutter)
        atlas_uvs = face_uvs.reshape(-1, 2)
        atlas_faces_uvs = torch.arange(
            atlas_uvs.shape[0], device=self.device, dtype=torch.int64
        ).reshape_as(faces.textures_idx)

        self.background = Meshes(
            verts=[verts],
            faces=[faces.verts_idx],
            textures=TexturesUV(
                maps=[texture_atlas],
                verts_uvs=[atlas_uvs],
                faces_uvs=[atlas_faces_uvs],
            ),
        )
        self._background_is_textured = True

