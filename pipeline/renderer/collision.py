"""Lightweight, dependency-free collision checks for static OBJ scenes."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def obj_bounds(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return axis-aligned bounds for the vertices in an OBJ file."""
    vertices = _obj_vertices(Path(path))
    if not vertices:
        raise ValueError(f"No vertices found in OBJ: {path}")
    points = np.asarray(vertices, dtype=np.float32)
    return points.min(axis=0), points.max(axis=0)


def obj_face_bounds(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return one axis-aligned bound per OBJ face.

    Face AABBs are deliberately conservative: rejecting a proposal whose
    subject box touches one is preferable to allowing the subject to clip
    through a static scene object.  They also avoid a heavy collision-library
    dependency for the search loop.
    """
    source = Path(path)
    vertices = _obj_vertices(source)
    lower, upper = [], []
    for line in source.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("f "):
            continue
        ids = []
        for reference in line.split()[1:]:
            index = int(reference.split("/", 1)[0])
            ids.append(index - 1 if index > 0 else len(vertices) + index)
        points = np.asarray([vertices[index] for index in ids], dtype=np.float32)
        lower.append(points.min(axis=0))
        upper.append(points.max(axis=0))
    if not lower:
        raise ValueError(f"No faces found in OBJ: {path}")
    return np.asarray(lower), np.asarray(upper)


def boxes_overlap_any(
    subject_lower: np.ndarray,
    subject_upper: np.ndarray,
    static_lower: np.ndarray,
    static_upper: np.ndarray,
    clearance: float,
) -> bool:
    """Whether a clearance-expanded subject AABB overlaps any static face."""
    lower = np.asarray(subject_lower, dtype=np.float32) - clearance
    upper = np.asarray(subject_upper, dtype=np.float32) + clearance
    overlaps = np.all((static_upper >= lower) & (static_lower <= upper), axis=1)
    return bool(np.any(overlaps))


def _obj_vertices(path: Path) -> list[tuple[float, float, float]]:
    vertices = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            _, x, y, z, *_ = line.split()
            vertices.append((float(x), float(y), float(z)))
    return vertices
