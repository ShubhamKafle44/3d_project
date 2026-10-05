from __future__ import annotations
import copy
import math
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Optional, Tuple
import os

import numpy as np
from PIL import Image

import config
from .base import DifferentiableScene
from .collision import boxes_overlap_any, obj_bounds, obj_face_bounds
import mitsuba as mi
import drjit as dr
import torch as torch

# Set the variant here so this module works no matter which file imports it first.
# If something else already chose a variant, leave it alone.
if mi.variant() is None:
    # llvm_ad_rgb supports Dr.Jit reverse-mode AD on CPU; users with a
    # compatible NVIDIA setup can opt into cuda_ad_rgb via the environment.
    mi.set_variant(os.environ.get("MITSUBA_VARIANT", "llvm_ad_rgb"))


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def linear_to_srgb_torch(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp_min(0.0031308) ** (1.0 / 2.4) - 0.055)


def _srgb_output() -> bool:
    # Same as render_background.py (srgb_gamma=True). Set config.MITSUBA_SRGB_OUTPUT = False to disable.
    return bool(getattr(config, "MITSUBA_SRGB_OUTPUT", True))


def torch_from_numpy(array):
    return torch.from_numpy(np.array(array, copy=True).astype(np.float32))


# ---- textured background helpers --------------------------------------
def read_materials(obj_path: Path) -> dict:
    """material -> {"kd": diffuse texture, "alpha": opacity map or None}, from the .mtl."""
    with obj_path.open(encoding="utf-8", errors="replace") as f:
        mtl_files = [l.split(maxsplit=1)[1].strip() for l in f if l.startswith("mtllib ")]

    def resolve(rest, base):
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
    """Split one OBJ into one OBJ per textured material (+ None for untextured faces)."""
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


class _MitsubaTorchRender(torch.autograd.Function):
    """Bridge Torch cotangents to the Dr.Jit graph retained by mi.render."""

    @staticmethod
    def forward(ctx, scene: "MitsubaScene", position, lighting):
        pos = position.detach().cpu().numpy().astype(np.float32)
        intensity = float(lighting.detach().cpu())
        scene_dict = scene._build_scene_dict(differentiable=True)
        transform = (
            mi.ScalarTransform4f()
            .translate(pos.tolist())
            .rotate([0, 0, 1], scene.rot_deg[0])
            .rotate([1, 0, 0], scene.rot_deg[1])
            .rotate([0, 1, 0], scene.rot_deg[2])
        )
        part_keys = scene._part_shape_keys()
        for key in part_keys:
            scene_dict[key]["to_world"] = transform
        scene_dict["street_light"]["intensity"]["value"] = [
            20.0 * intensity * float(channel) for channel in scene.light_color
        ]
        mitsuba_scene = mi.load_dict(scene_dict)
        params = mi.traverse(mitsuba_scene)
        vertex_keys = [f"{key}.vertex_positions" for key in part_keys]
        for key in vertex_keys:
            dr.enable_grad(params[key])
        light_key = "street_light.intensity.value"
        dr.enable_grad(params[light_key])
        params.update()
        image = mi.render(mitsuba_scene, params=params, spp=scene._spp)
        image_np = np.array(image, copy=True)
        ctx.mitsuba_image = image
        ctx.params = params
        ctx.vertex_keys = vertex_keys
        ctx.light_key = light_key
        ctx.light_rgb = np.asarray(scene.light_color, dtype=np.float32)
        ctx.rot_deg = scene.rot_deg.copy()
        ctx.save_for_backward(torch_from_numpy(image_np))
        return torch_from_numpy(image_np).permute(2, 0, 1).clamp(0.0, 1.0)

    @staticmethod
    def backward(ctx, grad_output):
        image_t, = ctx.saved_tensors
        grad = grad_output.permute(1, 2, 0).contiguous()
        grad = grad * ((image_t >= 0.0) & (image_t <= 1.0)).to(grad.dtype)
        grad_np = grad.detach().cpu().numpy().astype(np.float32)
        dr.set_grad(ctx.mitsuba_image, mi.TensorXf(grad_np))
        dr.backward(ctx.mitsuba_image)

        position_grad = np.zeros(3, dtype=np.float32)
        for key in ctx.vertex_keys:
            vertex_grad = np.array(dr.grad(ctx.params[key]), copy=False).reshape(-1, 3)
            position_grad += vertex_grad.sum(axis=0)
        # Mesh vertex parameters are local-space values, but the optimized
        # position translates the rotated object in world space.
        yaw, pitch, roll = np.radians(ctx.rot_deg)
        cz, sz = np.cos(yaw), np.sin(yaw)
        cx, sx = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(roll), np.sin(roll)
        rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        position_grad = (rz @ rx @ ry) @ position_grad
        radiance_grad = np.array(dr.grad(ctx.params[ctx.light_key]), copy=False).reshape(-1, 3)
        radiance_grad = radiance_grad.sum(axis=0)
        scale = 20.0 * ctx.light_rgb
        lighting_grad = float(np.dot(radiance_grad, scale))
        return None, torch_from_numpy(position_grad).to(grad_output.device), torch.tensor(
            lighting_grad, dtype=grad_output.dtype, device=grad_output.device
        )


class MitsubaScene(DifferentiableScene):
    def __init__(self, device: str = "cpu", image_size: int = 512):
        self.image_size = image_size
        self._spp = config.MITSUBA_SPP

        self._part_paths: Dict[str, str] = {}
        self._part_colors: Dict[str, Tuple[float, float, float]] = {}
        self._part_dirs: Dict[str, TemporaryDirectory] = {}
        self._part_shapes: Dict[str, list] = {}  # name -> [(scene key, obj path, bsdf or None)]
        self._background_path: Optional[str] = None
        self._background_dir: Optional[TemporaryDirectory] = None
        self._background_shapes: Optional[dict] = None
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
        self._build_part_shapes(name, Path(path).resolve())

    def _build_part_shapes(self, name: str, obj_path: Path) -> None:
        """Split the subject OBJ per textured material (same logic as the background)."""
        old = self._part_dirs.pop(name, None)
        if old is not None:
            old.cleanup()
        tmp = TemporaryDirectory(prefix=f"mitsuba_part_{name}_")
        self._part_dirs[name] = tmp  # keep alive: split OBJs + alpha PNGs live here
        temp_dir = Path(tmp.name)

        materials = read_materials(obj_path)
        shapes = []
        for index, (material, mesh_path) in enumerate(split_materials(obj_path, materials, temp_dir)):
            bsdf = texture_bsdf(materials[material], temp_dir) if material is not None else None
            shapes.append((f"part_{name}_{index}", str(mesh_path), bsdf))
        self._part_shapes[name] = shapes

    def _part_shape_keys(self) -> list:
        return [key for shapes in self._part_shapes.values() for key, _, _ in shapes]

    def load_background(self, path: str) -> None:
        self._background_path = path
        self._background_face_lower, self._background_face_upper = obj_face_bounds(path)
        self._build_background_shapes(Path(path).resolve())

    def _build_background_shapes(self, obj_path: Path) -> None:
        """Parse the .mtl and split the OBJ once; the result is reused by every render."""
        if self._background_dir is not None:
            self._background_dir.cleanup()
        # Kept on self so the split OBJs / alpha PNGs outlive scene construction.
        self._background_dir = TemporaryDirectory(prefix="mitsuba_background_")
        temp_dir = Path(self._background_dir.name)

        materials = read_materials(obj_path)
        shapes = {}
        for index, (material, mesh_path) in enumerate(split_materials(obj_path, materials, temp_dir)):
            shapes[f"background_{index}"] = {
                "type": "obj",
                "filename": str(mesh_path),
                "face_normals": True,
                "bsdf": (
                    texture_bsdf(materials[material], temp_dir)
                    if material is not None
                    else {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.5] * 3}}
                ),
            }
        self._background_shapes = shapes

    # ---- position / rotation ----------------------------------------
    def set_position(self, x: float, y: float, z: float) -> None:
        candidate = np.array([x, y, z], dtype=np.float32)
        if not self.is_position_valid(candidate):
            return
        self.pos = candidate
        if getattr(self, "position_parameter", None) is not None:
            with torch.no_grad():
                self.position_parameter.copy_(torch.as_tensor(self.pos))
            self.position_parameter.grad = None

    def get_position(self) -> np.ndarray:
        if getattr(self, "position_parameter", None) is not None:
            return self.position_parameter.detach().cpu().numpy().copy()
        return self.pos.copy()

    def is_position_valid(self, position: np.ndarray) -> bool:
        """Keep the subject in the camera frame and clear of scene geometry."""
        if not self._is_subject_in_frame(position):
            return False
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

    def _is_subject_in_frame(self, position: np.ndarray) -> bool:
        """Project the rotated subject bounds through Mitsuba's perspective camera."""
        if self._subject_lower is None or self._subject_upper is None:
            return True

        lower, upper = self._subject_lower, self._subject_upper
        corners = np.array(
            [[x, y, z] for x in (lower[0], upper[0])
             for y in (lower[1], upper[1])
             for z in (lower[2], upper[2])],
            dtype=np.float32,
        )
        yaw, pitch, roll = np.radians(self.rot_deg)
        cz, sz = np.cos(yaw), np.sin(yaw)
        cx, sx = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(roll), np.sin(roll)
        rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        world = corners @ (rz @ rx @ ry).T + np.asarray(position, dtype=np.float32)

        origin = np.asarray(self._camera_origin(), dtype=np.float32)
        target = np.asarray(self._cam_target, dtype=np.float32)
        forward = target - origin
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array([0, 1, 0], dtype=np.float32))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        relative = world - origin
        depth = relative @ forward
        if np.any(depth <= 0):
            return False

        half_extent = depth * math.tan(math.radians(self._cam_fov) / 2.0)
        ndc_x = (relative @ right) / half_extent
        ndc_y = (relative @ up) / half_extent
        margin = 2.0 * config.POSITION_FRAME_MARGIN_PX / self.image_size
        limit = 1.0 - margin
        return bool(
            np.all(np.abs(ndc_x) <= limit)
            and np.all(np.abs(ndc_y) <= limit)
        )

    def set_rotation_deg(self, yaw: float, pitch: float = 0.0, roll: float = 0.0) -> None:
        self.rot_deg = np.array([yaw, pitch, roll], dtype=np.float32)

    def get_rotation_deg(self) -> np.ndarray:
        return self.rot_deg.copy()

    # ---- lighting -----------------------------------------------------
    def set_lighting(self, intensity: float) -> None:
        self.ambient_intensity = float(intensity)
        if getattr(self, "lighting_parameter", None) is not None:
            with torch.no_grad():
                self.lighting_parameter.fill_(float(intensity))
            self.lighting_parameter.grad = None

    def get_lighting(self) -> float:
        if getattr(self, "lighting_parameter", None) is not None:
            return float(self.lighting_parameter.detach().cpu())
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
    def _integrator_dict(differentiable: bool) -> dict:
        """`direct` takes one bounce and cannot pass rays through a `mask` BSDF, so the
        transparent parts of leaf cards render black. `path` (as in render_background.py)
        follows the null transmission. `prb` is the path-style integrator with a proper
        backward pass for the Torch bridge."""
        depth = int(getattr(config, "MITSUBA_MAX_DEPTH", 6))
        return {"type": "prb" if differentiable else "path", "max_depth": depth}

    def _build_scene_dict(self, differentiable: bool = False) -> dict:
        origin = self._camera_origin()

        scene_dict = {
            "type": "scene",
            "integrator": self._integrator_dict(differentiable),
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
                    "rfilter": {"type": "gaussian"},
                },
                "sampler": {"type": "independent", "sample_count": self._spp},
            },
            "environment": {
                "type": "constant",
                "radiance": {
                    "type": "rgb",
                    "value": [
                        math.pi * float(config.AMBIENT_LIGHT["intensity"]) * float(component)
                        for component in config.AMBIENT_LIGHT["color"]
                    ],
                },
            },
            "street_light": {
                "type": "point",
                "position": list(config.LIGHT["position"]),
                "intensity": {
                    "type": "rgb",
                    "value": [
                        20.0 * self.ambient_intensity * float(channel)
                        for channel in self.light_color
                    ],
                },
            },
            "fill_light": {
                "type": "point",
                "position": list(config.FILL_LIGHT["position"]),
                "intensity": {
                    "type": "rgb",
                    "value": [
                        20.0 * float(config.FILL_LIGHT["intensity"]) * float(channel)
                        for channel in config.FILL_LIGHT["color"]
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

        for name, shapes in self._part_shapes.items():
            r, g, b = self._part_colors.get(name, (0.8, 0.8, 0.8))
            for key, mesh_path, bsdf in shapes:
                scene_dict[key] = {
                    "type": "obj",
                    "filename": mesh_path,
                    "to_world": to_world,
                    # untextured faces fall back to the part color
                    "bsdf": copy.deepcopy(bsdf) if bsdf is not None
                    else {"type": "diffuse", "reflectance": {"type": "rgb", "value": [r, g, b]}},
                }

        if self._background_shapes:
            scene_dict.update(copy.deepcopy(self._background_shapes))
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
            if getattr(self, "position_parameter", None) is not None:
                self.pos = self.position_parameter.detach().cpu().numpy().astype(np.float32)
                self.ambient_intensity = float(self.lighting_parameter.detach().cpu())
            scene_dict = self._build_scene_dict()
            scene = mi.load_dict(scene_dict)
            image = mi.render(scene, spp=self._spp)
            img_np = np.clip(np.array(image, copy=True)[..., :3], 0.0, 1.0)
            if _srgb_output():
                img_np = linear_to_srgb(img_np)
            return (img_np * 255.0).astype(np.uint8)
        except Exception as exc:  # noqa: BLE001
            print(f"[mitsuba_renderer] render failed: {exc}")
            return None

    def render_differentiable(self):
        """Render CHW float RGB with a Torch autograd bridge to Dr.Jit."""
        if not self._part_paths:
            raise RuntimeError("load at least one mesh before rendering")
        if mi.variant() not in ("llvm_ad_rgb", "cuda_ad_rgb"):
            raise RuntimeError(
                f"Mitsuba differentiable rendering needs an AD variant, got {mi.variant()!r}. "
                "Set MITSUBA_VARIANT=llvm_ad_rgb or cuda_ad_rgb before importing Mitsuba."
            )

        if getattr(self, "position_parameter", None) is None:
            self.position_parameter = torch.tensor(
                self.pos.copy(), dtype=torch.float32, requires_grad=True
            )
        if getattr(self, "lighting_parameter", None) is None:
            self.lighting_parameter = torch.tensor(
                self.ambient_intensity, dtype=torch.float32, requires_grad=True
            )
        image = _MitsubaTorchRender.apply(
            self, self.position_parameter, self.lighting_parameter
        )
        return linear_to_srgb_torch(image) if _srgb_output() else image

    def differentiable_parameters(self):
        """Return optimizer-ready Torch leaves for position and light intensity."""
        if getattr(self, "position_parameter", None) is None:
            self.position_parameter = torch.tensor(
                self.pos.copy(), dtype=torch.float32, requires_grad=True
            )
        if getattr(self, "lighting_parameter", None) is None:
            self.lighting_parameter = torch.tensor(
                self.ambient_intensity, dtype=torch.float32, requires_grad=True
            )
        return self.position_parameter, self.lighting_parameter
