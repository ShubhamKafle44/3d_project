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

