from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import config

from ..collision import obj_bounds, obj_face_bounds
from .materials import read_materials, split_materials, texture_bsdf


class AssetLoadingMixin:
    # ---- asset loading ---------------------------------------------
    def load_mesh(self, path: str, name: str = "mesh") -> None:
        """Load a subject part and update the bounds used for placement checks."""
        self._part_paths[name] = path
        self._part_colors.setdefault(name, (0.8, 0.8, 0.8))
        lower, upper = obj_bounds(path)
        self._subject_lower = lower if self._subject_lower is None else np.minimum(self._subject_lower, lower)
        self._subject_upper = upper if self._subject_upper is None else np.maximum(self._subject_upper, upper)
        self._build_part_shapes(name, Path(path).resolve())

    def _build_part_shapes(self, name: str, obj_path: Path) -> None:
        """Split the subject OBJ per textured material (same logic as the background)."""
        for old_key, _, _ in self._part_shapes.get(name, []):
            self._shape_texture_paths.pop(old_key, None)
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
            shape_key = f"part_{name}_{index}"
            shapes.append((shape_key, str(mesh_path), bsdf))
            if material is not None:
                self._shape_texture_paths[shape_key] = str(materials[material]["kd"].resolve())
        self._part_shapes[name] = shapes

    def available_texture_targets(self) -> list[str]:
        """Return loaded OBJ part names that have at least one diffuse texture."""
        textured_parts = set()
        for part_name, shapes in self._part_shapes.items():
            if any(key in self._shape_texture_paths for key, _, _ in shapes):
                textured_parts.add(part_name)
        return sorted(textured_parts)

    def set_texture_attack_target(self, target: str | None = None) -> str:
        """Select an OBJ part and prepare all of its diffuse maps for optimization."""
        if target is None:
            targets = self.available_texture_targets()
            if len(targets) != 1:
                choices = ", ".join(targets) or "(no textured OBJ parts found)"
                raise ValueError(f"Choose texture_attack_target from loaded OBJ parts: {choices}")
            target = targets[0]
        if target not in self._part_shapes:
            choices = ", ".join(self.available_texture_targets()) or "(no textured OBJ parts found)"
            raise ValueError(f"OBJ texture target {target!r} is not loaded. Choose from: {choices}")

        shape_keys = [
            key for key, _, _ in self._part_shapes[target]
            if key in self._shape_texture_paths
        ]
        if not shape_keys:
            raise ValueError(f"OBJ part {target!r} has no diffuse texture maps to optimize")
        paths = sorted({self._shape_texture_paths[key] for key in shape_keys})
        self._texture_attack_assets = []
        from PIL import Image

        resolution = int(getattr(config, "TEXTURE_ATTACK_RESOLUTION", 256))
        noise_std = float(getattr(config, "TEXTURE_ATTACK_NOISE_STD", 0.30))
        noise_seed = int(getattr(config, "TEXTURE_ATTACK_NOISE_SEED", 7))
        if resolution < 1:
            raise ValueError("TEXTURE_ATTACK_RESOLUTION must be positive")
        if noise_std < 0:
            raise ValueError("TEXTURE_ATTACK_NOISE_STD must be non-negative")

        # Preserve each default material map and its UV layout. Add fixed-seed
        # pixel noise in linear RGB before optimizing, so runs start from a
        # reproducible noisy version of the model's actual texture.
        rng = np.random.default_rng(noise_seed)
        offset = 0
        for path in paths:
            with Image.open(path) as image:
                srgb = np.asarray(
                    image.convert("RGB").resize(
                        (resolution, resolution), Image.Resampling.LANCZOS
                    ),
                    dtype=np.float32,
                ) / 255.0
            linear = np.where(
                srgb <= 0.04045,
                srgb / 12.92,
                ((srgb + 0.055) / 1.055) ** 2.4,
            ).astype(np.float32)
            if noise_std:
                linear = np.clip(
                    linear + rng.normal(0.0, noise_std, linear.shape).astype(np.float32),
                    0.0,
                    1.0,
                )
            mapped_shapes = [
                key for key in shape_keys
                if self._shape_texture_paths[key] == path
            ]
            count = int(linear.size)
            self._texture_attack_assets.append({
                "path": path,
                "source_paths": [path],
                "shape_keys": mapped_shapes,
                "shape": (resolution, resolution, 3),
                "offset": offset,
                "count": count,
                "initial": linear.reshape(-1),
            })
            offset += count

        self.texture_attack_target = target
        self._texture_image_override = None
        self.texture_image_parameter = None
        return target

    def load_texture_attack_image(self) -> np.ndarray:
        """Load the shared 256x256 linear RGB attack image as a flat vector."""
        if not self._texture_attack_assets:
            self.set_texture_attack_target(self.texture_attack_target)
        return np.concatenate([asset["initial"] for asset in self._texture_attack_assets])

    def _texture_images_from_vector(self, vector: np.ndarray) -> list[np.ndarray]:
        values = np.asarray(vector, dtype=np.float32).reshape(-1)
        return [
            values[asset["offset"]:asset["offset"] + asset["count"]].reshape(asset["shape"])
            for asset in self._texture_attack_assets
        ]

    def get_texture_image(self) -> np.ndarray:
        """Return selected-part texture pixels in linear RGB."""
        if self.texture_image_parameter is not None:
            return self.texture_image_parameter.detach().cpu().numpy().copy()
        if self._texture_image_override is not None:
            return self._texture_image_override.copy()
        return self.load_texture_attack_image()

    def set_texture_image(self, image: np.ndarray) -> None:
        """Set selected-part texture pixels and synchronize the optimizer parameter."""
        value = np.asarray(image, dtype=np.float32)
        expected = self.load_texture_attack_image().shape
        if value.shape != expected:
            raise ValueError(f"Expected texture shaped {expected}, got {value.shape}")
        value = np.clip(value, 0.0, 1.0)
        self._texture_image_override = value.copy()
        if self.texture_image_parameter is not None:
            import torch
            with torch.no_grad():
                self.texture_image_parameter.copy_(torch.as_tensor(value))
            self.texture_image_parameter.grad = None

    def save_texture_images(self, output_dir: str, vector: np.ndarray | None = None) -> list[str]:
        """Save the optimized 256x256 RGB attack image as an 8-bit PNG."""
        from PIL import Image
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        images = self._texture_images_from_vector(self.get_texture_image() if vector is None else vector)
        saved = []
        for asset, linear in zip(self._texture_attack_assets, images):
            srgb = np.where(
                linear <= 0.0031308,
                12.92 * linear,
                1.055 * np.maximum(linear, 0.0) ** (1.0 / 2.4) - 0.055,
            )
            rgb = (np.clip(srgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            suffix = f"_{Path(asset['path']).stem}"
            destination = output_path / (
                f"optimized_{self.texture_attack_target}{suffix}_texture.png"
            )
            Image.fromarray(rgb).save(destination)
            saved.append(str(destination))
        return saved

    def _part_shape_keys(self) -> list:
        """Return the Mitsuba scene keys for all loaded subject shapes."""
        return [key for shapes in self._part_shapes.values() for key, _, _ in shapes]

    def load_background(self, path: str) -> None:
        """Load the background mesh and cache its face bounds and materials."""
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


￼
Latest response
