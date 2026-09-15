from __future__ import annotations
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import config
from .base import DifferentiableScene
from .collision import boxes_overlap_any, obj_bounds, obj_face_bounds

try:
    from pytorch3d.io import load_obj, load_objs_as_meshes
    from pytorch3d.structures import join_meshes_as_scene, Meshes
    from pytorch3d.renderer import (
        look_at_view_transform,
        FoVPerspectiveCameras,
        PointLights,
        RasterizationSettings,
        MeshRenderer,
        MeshRasterizer,
        SoftPhongShader,
        TexturesUV,
        TexturesVertex,
    )
    _PYTORCH3D_AVAILABLE = True
except ImportError:
    _PYTORCH3D_AVAILABLE = False


class PyTorch3DScene(DifferentiableScene):
    def __init__(self, device: str = "cpu", image_size: int = 512):
        if not _PYTORCH3D_AVAILABLE:
            raise ImportError(
                "pytorch3d is not installed. Install it per "
                "https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md "
                "for your torch/CUDA version."
            )
        self.device = torch.device(device)
        self.image_size = image_size

        self.parts: Dict[str, Meshes] = {}
        self.background: Optional[Meshes] = None
        self._background_is_textured = False
        self._subject_lower: Optional[np.ndarray] = None
        self._subject_upper: Optional[np.ndarray] = None
        self._background_face_lower: Optional[np.ndarray] = None
        self._background_face_upper: Optional[np.ndarray] = None

        # Adversary-controlled parameters (Tensors for Autograd)
        self.pos = torch.zeros(3, device=self.device)
        self.rot_deg = torch.zeros(3, device=self.device)  # yaw, pitch, roll
        self.ambient_intensity = torch.tensor(1.0, device=self.device)

        # Camera state
        self._cam_distance = 14
        self._cam_elev = 10.0
        self._cam_azim = 0.0
        self._cam_target = (0.0, 1.0, 0.0)
        self._cam_fov = 40.0

        self._raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            # The road scene has many small faces in the same screen bins.
            # These settings prevent coarse-rasterization overflow and the
            # resulting incomplete geometry warnings.
            # The CUDA kernel needs fewer than 22 bins on either screen axis.
            # Scale this with resolution: 32 at 512px, 64 at 1024px, and 128
            # at 2048px. CPU rendering uses the supported naive path.
            bin_size=(
                2 ** max(int(np.ceil(np.log2(self.image_size))) - 4, 4)
                if self.device.type == "cuda"
                else 0
            ),
            max_faces_per_bin=200_000,
        )

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
        verts, faces, aux = load_obj(
            path,
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
        tiles = [torch.full(
            (3, atlas_height + 2 * gutter, atlas_height + 2 * gutter),
            0.18,
            dtype=torch.float32,
        )]
        content_widths = [atlas_height]
        tile_widths = [atlas_height + 2 * gutter]
        for texture in texture_images.values():
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
            # Replicate the edge texels around a tile.  Without this gutter,
            # bilinear filtering blends at an atlas boundary with the next
            # material and produces visible road/prop seams.
            tiles.append(F.pad(image[0], (gutter, gutter, gutter, gutter), mode="replicate"))
            content_widths.append(tile_width)
            tile_widths.append(tile_width + 2 * gutter)

        atlas_width = sum(tile_widths)
        texture_atlas = torch.cat(tiles, dim=2).permute(1, 2, 0).to(self.device)

        source_material_ids = faces.materials_idx.to(self.device)
        if source_material_ids.max().item() >= len(texture_images):
            raise ValueError("Background material indices do not match loaded textures")
        # -1 means no material: map it to the neutral tile at index 0.
        material_ids = torch.where(
            source_material_ids >= 0,
            source_material_ids + 1,
            torch.zeros_like(source_material_ids),
        )
        widths = torch.tensor(content_widths, dtype=torch.float32, device=self.device)
        padded_widths = torch.tensor(tile_widths, dtype=torch.float32, device=self.device)
        offsets = torch.cat((
            torch.zeros(1, device=self.device),
            padded_widths.cumsum(dim=0)[:-1],
        ))

        # UV indices may be shared by faces that use different materials.
        # Duplicate them per face before applying the material-tile transform.
        face_uvs = aux.verts_uvs.to(self.device)[faces.textures_idx.to(self.device)].clone()

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

    # ---- position / rotation ----------------------------------------
    def set_position(self, x: float, y: float, z: float) -> None:
        candidate = np.array([x, y, z], dtype=np.float32)
        if self.is_position_valid(candidate):
            self.pos = torch.tensor(candidate, device=self.device, dtype=torch.float32)

    def get_position(self) -> np.ndarray:
        return self.pos.detach().cpu().numpy().copy()

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
        self.rot_deg = torch.tensor([yaw, pitch, roll], device=self.device, dtype=torch.float32)

    def get_rotation_deg(self) -> np.ndarray:
        return self.rot_deg.detach().cpu().numpy().copy()

    # ---- lighting -----------------------------------------------------
    def set_lighting(self, intensity: float) -> None:
        self.ambient_intensity = torch.tensor(float(intensity), device=self.device)

    def get_lighting(self) -> float:
        return float(self.ambient_intensity.detach().cpu().item())

    # ---- material -------------------------------------------------------
    def set_material_color(self, part_name: str, rgb: Tuple[float, float, float]) -> None:
        mesh = self.parts.get(part_name)
        if mesh is None:
            return
        n_verts = mesh.verts_packed().shape[0]
        color = torch.tensor(rgb, device=self.device, dtype=torch.float32)
        colors = color.unsqueeze(0).expand(n_verts, 3).unsqueeze(0).clone()
        mesh.textures = TexturesVertex(verts_features=colors)

    def get_material_color(self, part_name: str) -> np.ndarray:
        mesh = self.parts.get(part_name)
        if mesh is not None and hasattr(mesh.textures, "verts_features_list"):
            return mesh.textures.verts_features_list()[0][0].detach().cpu().numpy().copy()
        return np.array([1.0, 1.0, 1.0])

    def set_vertex_colors(self, part_name: str, colors: torch.Tensor) -> None:
        """Assign differentiable per-vertex RGB values to a mesh part."""
        mesh = self.parts.get(part_name)
        if mesh is None:
            raise KeyError(f"Unknown mesh part: {part_name}")
        expected = mesh.verts_packed().shape[0]
        if colors.shape != (expected, 3):
            raise ValueError(f"expected colors shaped ({expected}, 3), got {tuple(colors.shape)}")
        mesh.textures = TexturesVertex(verts_features=colors.unsqueeze(0))

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

    # ---- standard render (numpy) --------------------------------------
    def render(self) -> Optional[np.ndarray]:
        if not self.parts:
            return None
        try:
            img_tensor = self.render_differentiable()
            img_hwc = img_tensor.permute(1, 2, 0).clamp(0, 1)
            img_np = (img_hwc.detach().cpu().numpy() * 255.0).astype(np.uint8)
            return img_np
        except Exception as exc:
            print(f"[pytorch3d_renderer] render failed: {exc}")
            return None

    def render_differentiable(self) -> torch.Tensor:

        # ============================================================
        # 1. Assemble HUMAN ONLY for debugging
        # ============================================================
        R_obj = self._rotation_matrix()

        meshes = []

        for name, mesh in self.parts.items():

            verts = mesh.verts_padded()[0]

            # Apply object transformation
            verts = torch.matmul(verts, R_obj.T) + self.pos

            m = mesh.clone()
            m = m.update_padded(verts.unsqueeze(0))

            meshes.append(m)

        if len(meshes) == 1:
            scene_mesh = meshes[0]
        else:
            scene_mesh = join_meshes_as_scene(meshes)

        # Use the configured scene camera.  The old code recalculated both
        # target and distance from the human every frame, which guaranteed a
        # close-up and excluded the environment from composition.
        R, T = look_at_view_transform(
            dist=self._cam_distance,
            elev=self._cam_elev,
            azim=self._cam_azim,
            at=(self._cam_target,),
            device=self.device,
        )

        cameras = FoVPerspectiveCameras(
            device=self.device,
            R=R,
            T=T,
            fov=self._cam_fov,
        )

        # ============================================================
        # 6. Lighting
        # ============================================================
        lights = PointLights(
            device=self.device,

            # Position of the point light
            location=[
                [-2.0, 5.0, 5.0]
            ],

            # No ambient light
            ambient_color=(
                (0.0, 0.0, 0.0),
            ),

            # Main illumination from point light
            diffuse_color=(
                (0.7, 0.7, 0.7),
            ),

            # Very small specular highlight
            specular_color=(
                (0.05, 0.05, 0.05),
            ),
        )

        # ============================================================
        # 7. Rasterizer
        # ============================================================
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=self._raster_settings,
        )

        # ============================================================
        # 8. Renderer
        # ============================================================
        renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=SoftPhongShader(
                device=self.device,
                cameras=cameras,
                lights=lights,
            ),
        )

        # ============================================================
        # 9. Render
        # ============================================================
        foreground = renderer(scene_mesh)
        if self.background is not None and self._background_is_textured:
            background = renderer(self.background)
            # SoftPhong's alpha can be fractional at a face even with no
            # intended transparency.  Use a hard rasterized silhouette for
            # compositing so the road texture cannot show through the human.
            alpha = (foreground[..., 3:4] > 0).to(foreground.dtype)
            images = torch.cat(
                (foreground[..., :3] * alpha + background[..., :3] * (1 - alpha), alpha),
                dim=-1,
            )
        else:
            images = foreground

        img = images[0, ..., :3].clamp(0.0, 1.0)

        return img.permute(2, 0, 1)
