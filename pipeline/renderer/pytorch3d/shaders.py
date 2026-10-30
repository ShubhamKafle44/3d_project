from __future__ import annotations

import torch

from .dependencies import (
    PYTORCH3D_AVAILABLE, PointLights, SoftPhongShader, phong_shading,
    softmax_rgb_blend,
)

class InverseSquarePointLights(PointLights if PYTORCH3D_AVAILABLE else object):
    """Point lights with the distance falloff used by Mitsuba emitters.

    PyTorch3D's built-in ``PointLights`` normalizes the light direction but
    intentionally does not attenuate its brightness by distance.  Mitsuba's
    point emitter does, so matching its scene requires that attenuation here.
    """

    def _attenuation(self, points: torch.Tensor) -> torch.Tensor:
        location = self.reshape_location(points)
        distance_squared = (location - points).square().sum(dim=-1, keepdim=True)
        return distance_squared.clamp_min(1e-4).reciprocal()

    def diffuse(self, normals: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        return super().diffuse(normals, points) * self._attenuation(points)

    def specular(
        self,
        normals: torch.Tensor,
        points: torch.Tensor,
        camera_position: torch.Tensor,
        shininess: torch.Tensor,
    ) -> torch.Tensor:
        return (
            super().specular(normals, points, camera_position, shininess)
            * self._attenuation(points)
        )


class AlphaCutoutPhongShader(SoftPhongShader if PYTORCH3D_AVAILABLE else object):
    """Soft Phong shading with binary texture-alpha cutouts for foliage."""

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = super()._get_cameras(**kwargs)
        texels = meshes.sample_textures(fragments)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)
        colors = phong_shading(
            meshes=meshes,
            fragments=fragments,
            texels=texels[..., :3],
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        if texels.shape[-1] > 3:
            # ``Fragments`` is a frozen dataclass in current PyTorch3D, not
            # a NamedTuple.  Reconstruct it instead of using ``_replace``.
            fragments = type(fragments)(
                pix_to_face=fragments.pix_to_face.masked_fill(
                    texels[..., 3] < 0.5, -1
                ),
                zbuf=fragments.zbuf,
                bary_coords=fragments.bary_coords,
                dists=fragments.dists,
            )
        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        return softmax_rgb_blend(colors, fragments, blend_params, znear=znear, zfar=zfar)


