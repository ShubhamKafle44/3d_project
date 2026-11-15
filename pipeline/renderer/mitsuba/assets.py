from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

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
        """Return diffuse texture filenames available on loaded subject meshes."""
        return sorted({Path(path).name for path in self._shape_texture_paths.values()})

    def set_texture_attack_target(self, target: str | None = None) -> str:
        """Resolve a texture filename/path and select its material for optimization."""
        paths = set(self._shape_texture_paths.values())
        if target is None:
            if len(paths) != 1:
                choices = ", ".join(self.available_texture_targets()) or "(no diffuse maps found)"
                raise ValueError(f"Choose texture_attack_target from: {choices}")
            resolved = next(iter(paths))
        else:
            candidate = Path(target).expanduser()
            matches = [
                path for path in paths
                if path == str(candidate.resolve()) or Path(path).name == candidate.name
            ]
            if len(matches) != 1:
                choices = ", ".join(self.available_texture_targets()) or "(no diffuse maps found)"
                raise ValueError(
                    f"Texture target {target!r} did not identify one loaded diffuse map. "
                    f"Choose from: {choices}"
                )
            resolved = matches[0]
        self.texture_attack_target = resolved
        self._texture_image_override = None
        self.texture_image_parameter = None
        return resolved

    def load_texture_attack_image(self) -> np.ndarray:
        """Load the selected sRGB texture as a linear RGB optimization tensor."""
        if self.texture_attack_target is None:
            self.set_texture_attack_target()
        from PIL import Image

        image = np.asarray(Image.open(self.texture_attack_target).convert("RGB"), dtype=np.float32) / 255.0
        return np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4).astype(np.float32)

    def get_texture_image(self) -> np.ndarray:
        """Return the selected optimized texture in linear RGB."""
        if self.texture_image_parameter is not None:
            return self.texture_image_parameter.detach().cpu().numpy().copy()
        if self._texture_image_override is not None:
            return self._texture_image_override.copy()
        return self.load_texture_attack_image()

    def set_texture_image(self, image: np.ndarray) -> None:
        """Set the selected texture image and synchronize its optimizer parameter."""
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

    def save_texture_image(self, path: str, image: np.ndarray | None = None) -> None:
        """Save a linear RGB texture as an 8-bit sRGB PNG."""
        from PIL import Image

        linear = self.get_texture_image() if image is None else np.asarray(image, dtype=np.float32)
        srgb = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * np.maximum(linear, 0.0) ** (1.0 / 2.4) - 0.055)
        rgb = (np.clip(srgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        with Image.open(self.texture_attack_target) as source:
            if "A" in source.getbands():
                alpha = np.asarray(source.convert("RGBA"), dtype=np.uint8)[..., 3:4]
                rgb = np.concatenate((rgb, alpha), axis=2)
        Image.fromarray(rgb).save(path)

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

