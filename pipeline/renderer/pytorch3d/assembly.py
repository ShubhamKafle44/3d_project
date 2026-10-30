from __future__ import annotations

import numpy as np
import torch

from .dependencies import Meshes, join_meshes_as_scene


class SceneAssemblyMixin:
    # ---- transform helpers -------------------------------------------
    def _rotation_matrix(self) -> torch.Tensor:
        yaw, pitch, roll = (self.rot_deg * np.pi / 180.0).unbind(0)

        cz, sz = torch.cos(yaw), torch.sin(yaw)
        Rz = torch.stack([
            torch.stack([cz, -sz, torch.zeros_like(cz)]),
            torch.stack([sz, cz, torch.zeros_like(cz)]),
            torch.stack([torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)]),
        ])

        cx, sx = torch.cos(pitch), torch.sin(pitch)
        Rx = torch.stack([
            torch.stack([torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx)]),
            torch.stack([torch.zeros_like(cx), cx, -sx]),
            torch.stack([torch.zeros_like(cx), sx, cx]),
        ])

        cy, sy = torch.cos(roll), torch.sin(roll)
        Ry = torch.stack([
            torch.stack([cy, torch.zeros_like(cy), sy]),
            torch.stack([torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy)]),
            torch.stack([-sy, torch.zeros_like(cy), cy]),
        ])
        return (Rz @ Rx @ Ry).to(self.device)

    def _assemble_scene_mesh(self) -> Meshes:
        R = self._rotation_matrix()
        meshes = []
        for mesh in self.parts.values():
            verts = mesh.verts_padded()[0]
            verts = verts @ R.T + self.pos
            m = mesh.clone()
            m = m.update_padded(verts.unsqueeze(0))
            meshes.append(m)
        if self.background is not None and not self._background_is_textured:
            meshes.append(self.background)
        return join_meshes_as_scene(meshes) if len(meshes) > 1 else meshes[0]

