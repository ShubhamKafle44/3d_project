from __future__ import annotations
from typing import Dict, Optional, Tuple
import os
from pathlib import Path

import numpy as np
from PIL import Image

import config
from .base import DifferentiableScene
from .collision import boxes_overlap_any, obj_bounds, obj_face_bounds

try:
    import mitsuba as mi

    _MI_VARIANT = os.environ.get("MITSUBA_VARIANT", "cuda_ad_rgb")
    mi.set_variant(_MI_VARIANT)
    _MITSUBA_AVAILABLE = True
except ImportError:
    _MITSUBA_AVAILABLE = False


class MitsubaScene(DifferentiableScene):
    def __init__(self, device: str = "cpu", image_size: int = 512):
        if not _MITSUBA_AVAILABLE:
            raise ImportError("mitsuba is not installed. Run: pip install mitsuba")
        self.image_size = image_size

        self._part_paths: Dict[str, str] = {}
        self._part_colors: Dict[str, Tuple[float, float, float]] = {}
        self._background_path: Optional[str] = None
        self._subject_lower: Optional[np.ndarray] = None
        self._subject_upper: Optional[np.ndarray] = None
        self._background_face_lower: Optional[np.ndarray] = None
        self._background_face_upper: Optional[np.ndarray] = None

        self.pos = np.zeros(3, dtype=np.float32)
        self.rot_deg = np.zeros(3, dtype=np.float32)  # yaw, pitch, roll
        self.ambient_intensity = 1.0

        self._cam_distance = 3.0
        self._cam_elev = 10.0
        self._cam_azim = 0.0
        self._cam_target = (0.0, 0.0, 0.0)
        self._cam_fov = 40.0

        self._scene = None  

    # ---- asset loading ---------------------------------------------
    def load_mesh(self, path: str, name: str = "mesh") -> None:
        self._part_paths[name] = path
        self._part_colors.setdefault(name, (0.8, 0.8, 0.8))
        lower, upper = obj_bounds(path)
        self._subject_lower = lower if self._subject_lower is None else np.minimum(self._subject_lower, lower)
        self._subject_upper = upper if self._subject_upper is None else np.maximum(self._subject_upper, upper)

    def load_background(self, path: str) -> None:
        self._background_path = path
        self._background_face_lower, self._background_face_upper = obj_face_bounds(path)

    # ---- position / rotation ----------------------------------------
    def set_position(self, x: float, y: float, z: float) -> None:
        candidate = np.array([x, y, z], dtype=np.float32)
        if self.is_position_valid(candidate):
            self.pos = candidate

    def get_position(self) -> np.ndarray:
        return self.pos.copy()

    def is_position_valid(self, position: np.ndarray) -> bool:
        if self._background_face_lower is None or self._subject_lower is None:
            return True
        offset = np.asarray(position, dtype=np.float32)
        return not boxes_overlap_any(
            self._subject_lower + offset,
            self._subject_upper + offset,
            self._background_face_lower,
            self._background_face_upper,
            config.POSITION_COLLISION_CLEARANCE,
        )

    def set_rotation_deg(self, yaw: float, pitch: float = 0.0, roll: float = 0.0) -> None:
        self.rot_deg = np.array([yaw, pitch, roll], dtype=np.float32)

    def get_rotation_deg(self) -> np.ndarray:
        return self.rot_deg.copy()

    # ---- lighting -----------------------------------------------------
    def set_lighting(self, intensity: float) -> None:
        self.ambient_intensity = float(intensity)

    def get_lighting(self) -> float:
        return self.ambient_intensity

    # ---- material -------------------------------------------------------
    def set_material_color(self, part_name: str, rgb: Tuple[float, float, float]) -> None:
        self._part_colors[part_name] = tuple(float(c) for c in rgb)

    def get_material_color(self, part_name: str) -> np.ndarray:
        return np.array(self._part_colors.get(part_name, (0.8, 0.8, 0.8)))

    # ---- camera ----------------------------------------------------------
    def set_camera_orbit(
        self,
        distance: float,
        elevation: float,
        azimuth: float,
        target: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        fov: float = 40.0,
    ) -> None:
        self._cam_distance = distance
        self._cam_elev = elevation
        self._cam_azim = azimuth
        self._cam_target = target
        self._cam_fov = fov

    @property
    def cam_distance(self) -> float:
        return self._cam_distance

    @property
    def cam_elev(self) -> float:
        return self._cam_elev

    @property
    def cam_azim(self) -> float:
        return self._cam_azim

    @property
    def cam_target(self) -> Tuple[float, float, float]:
        return self._cam_target

    @property
    def cam_fov(self) -> float:
        return self._cam_fov

    # ---- scene assembly -------------------------------------------------
    def _camera_origin(self) -> Tuple[float, float, float]:
        elev_rad = np.radians(self._cam_elev)
        azim_rad = np.radians(self._cam_azim)
        d = self._cam_distance
        tx, ty, tz = self._cam_target
        x = tx + d * np.cos(elev_rad) * np.sin(azim_rad)
        y = ty + d * np.sin(elev_rad)
        z = tz + d * np.cos(elev_rad) * np.cos(azim_rad)
        return float(x), float(y), float(z)

    @staticmethod
    def _build_textured_background_asset(obj_path: str) -> tuple[str, str]:
        """Convert a multi-material OBJ to one OBJ + one bitmap atlas.

        Mitsuba's ``obj`` shape plugin does not import MTL texture bindings;
        it loads our road pack as a single uniform-grey BSDF.  This converter
        packs the source ``map_Kd`` images into one atlas and duplicates UVs
        per face so every original material still samples its own tile.
        Cached files are regenerated when the source OBJ changes.
        """
        source = Path(obj_path).resolve()
        cache_dir = source.parent / ".mitsuba_cache"
        mesh_out = cache_dir / f"{source.stem}_textured.obj"
        atlas_out = cache_dir / f"{source.stem}_atlas.png"
        if (
            mesh_out.exists()
            and atlas_out.exists()
            and mesh_out.stat().st_mtime >= source.stat().st_mtime
        ):
            return str(mesh_out), str(atlas_out)

        material_files: dict[str, Path] = {}
        mtl_path = source.with_suffix(".mtl")
        current_material = None
        for raw_line in mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            tokens = raw_line.split(maxsplit=1)
            if len(tokens) != 2:
                continue
            if tokens[0] == "newmtl":
                current_material = tokens[1]
            elif tokens[0] == "map_Kd" and current_material:
                material_files[current_material] = source.parent / tokens[1]
        if not material_files:
            raise ValueError(f"No diffuse texture maps found in {mtl_path}")

        # Preserve texture aspect ratios: the road remains 2:1 while signs
        # and barriers retain their square textures.
        atlas_height = 2048
        material_tiles = {}
        tile_images = []
        x_offset = 0
        for name, texture_path in material_files.items():
            with Image.open(texture_path) as image:
                image = image.convert("RGB")
                tile_width = max(1, round(image.width / image.height * atlas_height))
                tile = image.resize((tile_width, atlas_height), Image.Resampling.LANCZOS)
            material_tiles[name] = (x_offset, tile_width)
            tile_images.append(tile)
            x_offset += tile_width

        atlas = Image.new("RGB", (x_offset, atlas_height))
        cursor = 0
        for tile in tile_images:
            atlas.paste(tile, (cursor, 0))
            cursor += tile.width

        cache_dir.mkdir(parents=True, exist_ok=True)
        atlas.save(atlas_out, quality=95)

        lines = source.read_text(encoding="utf-8", errors="ignore").splitlines()
        first_material = next(iter(material_tiles))
        active_material = first_material
        remapped_uvs = []
        remapped_faces = []

        for raw_line in lines:
            if raw_line.startswith("usemtl "):
                active_material = raw_line.split(maxsplit=1)[1]
                continue
            if not raw_line.startswith("f "):
                continue
            x_start, tile_width = material_tiles.get(
                active_material, material_tiles[first_material]
            )
            face = []
            for vertex_ref in raw_line.split()[1:]:
                indices = vertex_ref.split("/")
                if len(indices) < 2 or not indices[1]:
                    raise ValueError("Textured background face has no UV coordinate")
                uv_line = lines[0]  # placate type checkers; overwritten below
                # OBJ UV indices are one-based and support negative indexing.
                uv_index = int(indices[1])
                uv_lines = [line for line in lines if line.startswith("vt ")]
                if uv_index < 0:
                    uv_index = len(uv_lines) + uv_index + 1
                uv_tokens = uv_lines[uv_index - 1].split()
                u, v = float(uv_tokens[1]) % 1.0, float(uv_tokens[2])
                remapped_uvs.append(((x_start + u * tile_width) / atlas.width, v))
                uv_id = len(remapped_uvs)
                normal = indices[2] if len(indices) > 2 and indices[2] else ""
                face.append(f"{indices[0]}/{uv_id}/{normal}" if normal else f"{indices[0]}/{uv_id}")
            remapped_faces.append("f " + " ".join(face))

        with mesh_out.open("w", encoding="utf-8") as output:
            for line in lines:
                if line.startswith(("v ", "vn ")):
                    output.write(line + "\n")
            for u, v in remapped_uvs:
                output.write(f"vt {u:.9f} {v:.9f}\n")
            output.write("\n".join(remapped_faces) + "\n")
        return str(mesh_out), str(atlas_out)

    @staticmethod
    def _build_background_material_assets(obj_path: str) -> list[tuple[str, Optional[str]]]:
        """Split a multi-material OBJ into material-specific meshes.

        A bitmap atlas cannot faithfully preserve repeated UVs: wrapping at a
        tile edge starts sampling the next material.  Keeping each source
        texture on its own shape retains the OBJ's original UV coordinates
        and eliminates atlas bleed/seams on the road and props.
        """
        source = Path(obj_path).resolve()
        cache_dir = source.parent / ".mitsuba_cache" / "material_meshes"
        mtl_path = source.with_suffix(".mtl")
        texture_files: dict[str, str] = {}
        material = None
        for raw_line in mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            tokens = raw_line.split(maxsplit=1)
            if len(tokens) != 2:
                continue
            if tokens[0] == "newmtl":
                material = tokens[1]
            elif tokens[0] == "map_Kd" and material:
                texture_files[material] = str((source.parent / tokens[1]).resolve())

        lines = source.read_text(encoding="utf-8", errors="ignore").splitlines()
        geometry = [line for line in lines if line.startswith(("v ", "vt ", "vn "))]
        faces_by_material: dict[Optional[str], list[str]] = {}
        material = None
        for line in lines:
            if line.startswith("usemtl "):
                material = line.split(maxsplit=1)[1]
            elif line.startswith("f "):
                faces_by_material.setdefault(material, []).append(line)

        cache_dir.mkdir(parents=True, exist_ok=True)
        assets = []
        source_mtime = max(source.stat().st_mtime, mtl_path.stat().st_mtime)
        for index, (material, faces) in enumerate(faces_by_material.items()):
            mesh_path = cache_dir / f"{source.stem}_{index}.obj"
            if not mesh_path.exists() or mesh_path.stat().st_mtime < source_mtime:
                with mesh_path.open("w", encoding="utf-8") as output:
                    output.write("\n".join(geometry) + "\n")
                    output.write("\n".join(faces) + "\n")
            assets.append((str(mesh_path), texture_files.get(material)))
        return assets

    def _build_scene_dict(self) -> dict:
        origin = self._camera_origin()

        scene_dict = {
            "type": "scene",
            "integrator": {"type": "path", "max_depth": 6},
            "sensor": {
                "type": "perspective",
                "fov": self._cam_fov,
                "to_world": mi.ScalarTransform4f().look_at(
                    origin=origin, target=self._cam_target, up=(0, 1, 0)
                ),
                "film": {
                    "type": "hdrfilm",
                    "width": self.image_size,
                    "height": self.image_size,
                    "pixel_format": "rgb",
                },
                "sampler": {"type": "independent", "sample_count": 32},
            },
            "light": {
                "type": "point",
                "position": [2.0, 2.0, 2.0],
                "intensity": {"type": "spectrum", "value": self.ambient_intensity * 20.0},
            },
        }

        to_world = (
            mi.ScalarTransform4f()
            .translate(self.pos.tolist())
            .rotate([0, 0, 1], self.rot_deg[0])
            .rotate([1, 0, 0], self.rot_deg[1])
            .rotate([0, 1, 0], self.rot_deg[2])
        )

        for name, path in self._part_paths.items():
            r, g, b = self._part_colors.get(name, (0.8, 0.8, 0.8))
            scene_dict[f"part_{name}"] = {
                "type": "obj",
                "filename": path,
                "to_world": to_world,
                "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [r, g, b]}},
            }

        if self._background_path is not None:
            for index, (mesh_path, texture_path) in enumerate(
                self._build_background_material_assets(self._background_path)
            ):
                reflectance = (
                    {
                        "type": "bitmap",
                        "filename": texture_path,
                        "filter_type": "bilinear",
                        "wrap_mode": "repeat",
                    }
                    if texture_path
                    # These enclosure faces have no material binding in the
                    # source OBJ, so retain them as neutral geometry instead
                    # of incorrectly assigning the first texture map to them.
                    else {"type": "rgb", "value": [0.18, 0.18, 0.18]}
                )
                scene_dict[f"background_{index}"] = {
                    "type": "obj",
                    "filename": mesh_path,
                    "bsdf": {"type": "diffuse", "reflectance": reflectance},
                }
        else:
            scene_dict["floor"] = {
                "type": "rectangle",
                "to_world": mi.ScalarTransform4f().translate([0, -1, 0])
                .rotate([1, 0, 0], -90)
                .scale(10),
                "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.6, 0.6, 0.6]}},
            }

        return scene_dict

    # ---- render -------------------------------------------------------
    def render(self) -> Optional[np.ndarray]:
        if not self._part_paths:
            return None
        try:
            scene_dict = self._build_scene_dict()
            scene = mi.load_dict(scene_dict)
            image = mi.render(scene, spp=32)
            img_np = np.array(mi.util.convert_to_bitmap(image))
            if img_np.shape[-1] == 4:
                img_np = img_np[..., :3]
            return img_np.astype(np.uint8)
        except Exception as exc:  # noqa: BLE001
            print(f"[mitsuba_renderer] render failed: {exc}")
            return None
