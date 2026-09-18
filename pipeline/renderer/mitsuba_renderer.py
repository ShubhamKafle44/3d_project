from __future__ import annotations
import math
from pathlib import Path
import re
from typing import Dict, Optional, Tuple
import os

import numpy as np
from PIL import Image

import config
from .base import DifferentiableScene
from .collision import boxes_overlap_any, obj_bounds, obj_face_bounds

try:
    import mitsuba as mi

    # CPU is the safe default for the standard random-search path.  Some
    # installations provide ``cuda_ad_rgb`` but not ``cuda_rgb``, and OptiX
    # can exceed a 6 GB GPU budget on this scene.  Opt into a CUDA variant
    # explicitly via MITSUBA_VARIANT when sufficient VRAM is available.
    _MI_VARIANT = os.environ.get("MITSUBA_VARIANT", "scalar_rgb")
    mi.set_variant(_MI_VARIANT)
    _MITSUBA_AVAILABLE = True
except ImportError:
    _MITSUBA_AVAILABLE = False


def _mitsuba_safe_obj(path: str) -> str:
    """Return an OBJ Mitsuba can load without modifying the source export.

    Blender can export a zero-length ``vn`` value for degenerate geometry.
    PyTorch3D accepts it, but Mitsuba correctly rejects it as invalid normal
    data.  Preserve every valid record and replace only malformed or zero
    normals in a sibling cache file; face indices therefore remain unchanged.
    """
    source = Path(path)
    cached = source.with_name(f"{source.stem}.mitsuba_safe{source.suffix}")
    cache_version = "# mitsuba_safe_version=2"
    if cached.exists() and cached.stat().st_mtime >= source.stat().st_mtime:
        with cached.open("r", encoding="utf-8", errors="replace") as handle:
            if handle.readline().strip() == cache_version:
                return str(cached)

    source_lines = source.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    vertices = []
    for line in source_lines:
        tokens = line.split()
        if tokens and tokens[0] == "v" and len(tokens) >= 4:
            try:
                vertices.append(np.asarray([float(value) for value in tokens[1:4]]))
            except ValueError:
                vertices.append(np.full(3, np.nan))

    invalid_normals = 0
    invalid_faces = 0
    lines = [cache_version + "\n"]
    for line in source_lines:
        if line.startswith("vn "):
            values = line.split()[1:4]
            try:
                normal = [float(value) for value in values]
                length = math.sqrt(sum(value * value for value in normal))
                valid = len(normal) == 3 and math.isfinite(length) and length > 1e-12
            except ValueError:
                valid = False
            if not valid:
                line = "vn 0.000000 1.000000 0.000000\n"
                invalid_normals += 1
        elif line.startswith("f "):
            try:
                indices = [int(reference.split("/")[0]) for reference in line.split()[1:]]
                indices = [index - 1 if index > 0 else len(vertices) + index for index in indices]
                points = [vertices[index] for index in indices]
                # Mitsuba triangulates polygons internally.  The fan-area
                # test catches a collapsed first triangle and non-finite OBJ
                # coordinates without changing valid faces or material IDs.
                valid = (
                    len(points) >= 3
                    and all(np.all(np.isfinite(point)) for point in points)
                    and all(
                        np.linalg.norm(np.cross(points[i] - points[0], points[i + 1] - points[0])) > 1e-12
                        for i in range(1, len(points) - 1)
                    )
                )
            except (IndexError, ValueError):
                valid = False
            if not valid:
                invalid_faces += 1
                continue
        lines.append(line)

    if not invalid_normals and not invalid_faces:
        return str(source)
    cached.write_text("".join(lines), encoding="utf-8")
    print(
        f"[mitsuba_renderer] repaired {invalid_normals} invalid normal(s) and "
        f"removed {invalid_faces} degenerate face(s) in {cached.name}"
    )
    return str(cached)


def _obj_diffuse_maps(obj_path: Path) -> Dict[str, tuple[Path, Optional[Path]]]:
    """Read usable diffuse and opacity bindings from an OBJ's MTL."""
    mtl_name = None
    for line in obj_path.read_text(encoding="utf-8", errors="replace").splitlines():
        tokens = line.split(maxsplit=1)
        if len(tokens) == 2 and tokens[0] == "mtllib":
            mtl_name = tokens[1]
            break
    if mtl_name is None:
        return {}
    mtl_path = obj_path.parent / mtl_name
    if not mtl_path.is_file():
        return {}

    maps: Dict[str, Dict[str, Path]] = {}
    material_name = None
    for line in mtl_path.read_text(encoding="utf-8", errors="replace").splitlines():
        tokens = line.split(maxsplit=1)
        if len(tokens) == 2 and tokens[0] == "newmtl":
            material_name = tokens[1]
        elif len(tokens) == 2 and tokens[0] in {"map_Kd", "map_d"} and material_name:
            image_path = mtl_path.parent / tokens[1]
            if image_path.is_file():
                key = "diffuse" if tokens[0] == "map_Kd" else "opacity"
                maps.setdefault(material_name, {})[key] = image_path
    result = {}
    for material, binding in maps.items():
        if "diffuse" not in binding:
            continue
        opacity = binding.get("opacity")
        # Mitsuba's bitmap texture evaluates an RGBA image's RGB luminance for
        # a scalar opacity request.  Extract the actual alpha channel instead.
        alpha_source = opacity or binding["diffuse"]
        alpha_mask = _alpha_mask_file(alpha_source)
        result[material] = (binding["diffuse"], alpha_mask or opacity)
    return result


def _alpha_mask_file(source: Path) -> Optional[Path]:
    """Return a cached grayscale copy of an image's embedded alpha channel."""
    try:
        image = Image.open(source)
        has_alpha = "A" in image.getbands() or "transparency" in image.info
        if not has_alpha:
            return None
        output = source.with_name(f"{source.stem}.mitsuba_alpha.png")
        if not output.is_file() or output.stat().st_mtime < source.stat().st_mtime:
            image.convert("RGBA").getchannel("A").save(output)
        return output
    except OSError:
        return None


def _write_textured_obj(
    output: Path,
    faces: list[list[str]],
    vertices: list[str],
    texcoords: list[str],
    normals: list[str],
) -> None:
    """Write a compact OBJ with independently remapped vertex indices."""
    vertex_ids: Dict[int, int] = {}
    texcoord_ids: Dict[int, int] = {}
    normal_ids: Dict[int, int] = {}
    remapped_faces = []

    def remap(index: int, mapping: Dict[int, int]) -> int:
        if index not in mapping:
            mapping[index] = len(mapping) + 1
        return mapping[index]

    for face in faces:
        remapped = []
        for reference in face:
            fields = reference.split("/")
            vertex = remap(int(fields[0]), vertex_ids)
            texture = remap(int(fields[1]), texcoord_ids) if len(fields) > 1 and fields[1] else None
            normal = remap(int(fields[2]), normal_ids) if len(fields) > 2 and fields[2] else None
            if normal is not None:
                remapped.append(f"{vertex}/{texture or ''}/{normal}")
            elif texture is not None:
                remapped.append(f"{vertex}/{texture}")
            else:
                remapped.append(str(vertex))
        remapped_faces.append(remapped)

    with output.open("w", encoding="utf-8") as handle:
        for index in vertex_ids:
            handle.write(vertices[index - 1])
        for index in texcoord_ids:
            handle.write(texcoords[index - 1])
        for index in normal_ids:
            handle.write(normals[index - 1])
        for face in remapped_faces:
            handle.write("f " + " ".join(face) + "\n")


def _mitsuba_background_shapes(path: str) -> list[tuple[str, str, Optional[str], Optional[str]]]:
    """Split OBJ material groups because Mitsuba's OBJ plugin ignores MTLs.

    The base mesh contains all untextured faces.  Every ``map_Kd`` material is
    emitted as a small independent OBJ so a bitmap BSDF can be attached to it
    directly by Mitsuba.
    """
    safe_obj = Path(_mitsuba_safe_obj(path))
    diffuse_maps = _obj_diffuse_maps(safe_obj)
    if not diffuse_maps:
        return [("background", str(safe_obj), None, None)]

    cache_dir = safe_obj.parent / f"{safe_obj.stem}_materials"
    base_obj = cache_dir / "untextured.obj"
    cache_files = {
        material: cache_dir / f"{index:02d}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', material)}.obj"
        for index, material in enumerate(diffuse_maps)
    }
    cache_is_current = (
        base_obj.is_file()
        and all(file.is_file() for file in cache_files.values())
        and all(file.stat().st_mtime >= safe_obj.stat().st_mtime for file in [base_obj, *cache_files.values()])
    )
    if not cache_is_current:
        cache_dir.mkdir(exist_ok=True)
        vertices, texcoords, normals = [], [], []
        texture_faces = {material: [] for material in diffuse_maps}
        current_material = None
        with base_obj.open("w", encoding="utf-8") as base_handle:
            for line in safe_obj.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True):
                tokens = line.split()
                if len(tokens) >= 2 and tokens[0] == "usemtl":
                    current_material = tokens[1]
                elif tokens and tokens[0] == "v":
                    vertices.append(line)
                elif tokens and tokens[0] == "vt":
                    texcoords.append(line)
                elif tokens and tokens[0] == "vn":
                    normals.append(line)

                if tokens and tokens[0] == "f" and current_material in texture_faces:
                    texture_faces[current_material].append(tokens[1:])
                else:
                    base_handle.write(line)

        for material, faces in texture_faces.items():
            _write_textured_obj(cache_files[material], faces, vertices, texcoords, normals)
        print(f"[mitsuba_renderer] prepared {len(diffuse_maps)} textured material mesh(es)")

    shapes = [("background_base", str(base_obj), None, None)]
    shapes.extend(
        (
            f"background_texture_{index}",
            str(cache_files[material]),
            str(image_path),
            str(opacity_path) if opacity_path is not None else None,
        )
        for index, (material, (image_path, opacity_path)) in enumerate(diffuse_maps.items())
    )
    return shapes


class MitsubaScene(DifferentiableScene):
    def __init__(self, device: str = "cpu", image_size: int = 512):
        if not _MITSUBA_AVAILABLE:
            raise ImportError("mitsuba is not installed. Run: pip install mitsuba")
        self.image_size = image_size
        self._spp = config.MITSUBA_SPP

        self._part_paths: Dict[str, str] = {}
        self._part_colors: Dict[str, Tuple[float, float, float]] = {}
        self._background_path: Optional[str] = None
        self._background_shapes: list[tuple[str, str, Optional[str], Optional[str]]] = []
        self._subject_lower: Optional[np.ndarray] = None
        self._subject_upper: Optional[np.ndarray] = None
        self._background_face_lower: Optional[np.ndarray] = None
        self._background_face_upper: Optional[np.ndarray] = None

        self.pos = np.zeros(3, dtype=np.float32)
        self.rot_deg = np.zeros(3, dtype=np.float32)  # yaw, pitch, roll
        self.ambient_intensity = 1.0
        self.light_color = config.LIGHT.get("color", (1.0, 1.0, 1.0))

        self._cam_distance = 3.0
        self._cam_elev = 10.0
        self._cam_azim = 0.0
        self._cam_target = (0.0, 0.0, 0.0)
        self._cam_fov = 40.0

    # ---- asset loading ---------------------------------------------
    def load_mesh(self, path: str, name: str = "mesh") -> None:
        self._part_paths[name] = path
        self._part_colors.setdefault(name, (0.8, 0.8, 0.8))
        lower, upper = obj_bounds(path)
        self._subject_lower = lower if self._subject_lower is None else np.minimum(self._subject_lower, lower)
        self._subject_upper = upper if self._subject_upper is None else np.maximum(self._subject_upper, upper)

    def load_background(self, path: str) -> None:
        self._background_path = _mitsuba_safe_obj(path)
        self._background_shapes = _mitsuba_background_shapes(path)
        self._background_face_lower, self._background_face_upper = obj_face_bounds(path)

    # ---- position / rotation ----------------------------------------
    def set_position(self, x: float, y: float, z: float) -> None:
        self.pos = np.array([x, y, z], dtype=np.float32)

    def get_position(self) -> np.ndarray:
        return self.pos.copy()

    def is_position_valid(self, position: np.ndarray) -> bool:
        """Match PyTorch3D's static-scene collision guard for POSITION search."""
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

    def _build_scene_dict(self) -> dict:
        origin = self._camera_origin()

        scene_dict = {
            "type": "scene",
            # Every material constructed below is diffuse and illumination
            # comes from explicit point emitters.  Indirect path bounces add
            # no useful transport here, while the non-manifold source OBJ can
            # make them numerically unstable.  Direct lighting retains the
            # visible lighting and shadows without invalid sample values.
            "integrator": {"type": "direct"},
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
                "sampler": {"type": "independent", "sample_count": self._spp},
            },
            # The exported lamp mesh is non-manifold.  Point emitters near
            # it can create zero-distance 1/r^2 evaluations, which lead to
            # the ImageBlock NaNs. A constant environment is finite at every
            # surface point and gives this diffuse-only scene stable lighting.
            "environment": {
                "type": "constant",
                "radiance": {
                    "type": "rgb",
                    "value": [
                        component * self.ambient_intensity * 0.15
                        for component in self.light_color
                    ],
                },
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
            for shape_name, shape_path, texture_path, opacity_path in self._background_shapes:
                shape = {"type": "obj", "filename": shape_path}
                if texture_path is not None:
                    diffuse = {
                        "type": "diffuse",
                        "reflectance": {"type": "bitmap", "filename": texture_path},
                    }
                    shape["bsdf"] = (
                        {
                            "type": "mask",
                            "opacity": {"type": "bitmap", "filename": opacity_path},
                            "nested_bsdf": diffuse,
                        }
                        if opacity_path is not None
                        else diffuse
                    )
                else:
                    shape["bsdf"] = {
                        "type": "diffuse",
                        "reflectance": {"type": "rgb", "value": [0.5, 0.5, 0.5]},
                    }
                scene_dict[shape_name] = shape
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
            image = mi.render(scene, spp=self._spp)
            img_np = np.array(mi.util.convert_to_bitmap(image))
            if img_np.shape[-1] == 4:
                img_np = img_np[..., :3]
            return img_np.astype(np.uint8)
        except Exception as exc:  # noqa: BLE001
            print(f"[mitsuba_renderer] render failed: {exc}")
            return None
