from __future__ import annotations

import logging

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import math

import pytorch3d.transforms as T
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.renderer import (
    BlendParams,
    DirectionalLights,
    FoVPerspectiveCameras,
    HardPhongShader,
    Materials,
    MeshRasterizer,
    MeshRenderer,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes, join_meshes_as_scene

from config import CAMERA_AZIM, CAMERA_DIST, CAMERA_ELEV, IMAGE_SIZE, OBJ_PATH
from core.background import load_background
from core.mesh_loader import load_mesh_from_file

logger = logging.getLogger(__name__)

class MeshScene:
    """Wraps a PyTorch3D renderer and exposes differentiable scene parameters."""

    def __init__(self, meshes, device: torch.device):
        self.device = device
        self.mesh   = meshes

        self.renderer, self.cameras = self._build_renderer()

        # Differentiable scene parameters (initialised to defaults)
        self.pos               = torch.zeros(3, device=device, requires_grad=True)
        self.rot               = torch.zeros(3, device=device, requires_grad=True)
        self.light_pos         = torch.tensor([[0.0, 1.0, -2.0]], device=device, requires_grad=True)
        self.ambient_intensity = torch.tensor([0.5], device=device, requires_grad=True)

        if OBJ_PATH:
            self.load_mesh(OBJ_PATH)

    # ── Renderer ──────────────────────────────────────────────────────────────
    def _build_renderer(self):
        R, T_cam = look_at_view_transform(dist=CAMERA_DIST, elev=CAMERA_ELEV, azim=CAMERA_AZIM)
        cameras  = FoVPerspectiveCameras(device=self.device, R=R, T=T_cam)

        raster_settings = RasterizationSettings(
            image_size=IMAGE_SIZE,
            blur_radius=0.0,
            faces_per_pixel=1,
            max_faces_per_bin=50000,
        )
        lights = PointLights(
            device=self.device,
            location=[[0.0, 2.0, -2.0]],
            ambient_color=[[0.5, 0.5, 0.5]],
            diffuse_color=[[0.8, 0.8, 0.8]],
            specular_color=[[0.3, 0.3, 0.3]],
        )
        materials = Materials(device=self.device, shininess=64.0)

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(
                device=self.device,
                cameras=cameras,
                lights=lights,
                materials=materials,
            ),
        )
        return renderer, cameras

    # ── Mesh loading ──────────────────────────────────────────────────────────
    def load_mesh(self, path: str) -> None:
        """Load a new OBJ and reset all scene parameters to defaults.

        Resetting pos/rot is essential: the previous model's adversarially
        optimised position would put the new model completely off-screen.
        """
        self.mesh = load_mesh_from_file(path, self.device)
        logger.info("Mesh loaded: %s", path)

        with torch.no_grad():
            self.pos.zero_()
            self.rot.zero_()
            self.light_pos.copy_(
                torch.tensor([[0.0, 1.0, -2.0]], device=self.device)
            )
            self.ambient_intensity.fill_(0.5)

    # ── Lighting ──────────────────────────────────────────────────────────────
    def _make_lights(self) -> PointLights:
        ambient_color = self.ambient_intensity.expand(3).unsqueeze(0)
        return PointLights(
            device=self.device,
            location=self.light_pos,
            ambient_color=ambient_color,
        )

    # ── Blank fallback ────────────────────────────────────────────────────────
    def _blank(self):
        blank_np = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
        blank_t  = torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE))
        return blank_np, blank_t

    # ── UI render (no gradients) ──────────────────────────────────────────────
    def render(self):
        if self.mesh is None:
            return self._blank()

        bg = load_background()

        with torch.no_grad():
            angles = self.rot.unsqueeze(0)
            R      = T.euler_angles_to_matrix(angles, convention="XYZ")[0]
            verts  = self.mesh.verts_packed()
            verts  = verts @ R.T + self.pos
            mesh   = self.mesh.update_padded(verts.unsqueeze(0))
            images = self.renderer(mesh, lights=self._make_lights())

        rgba  = images[0].cpu().numpy()
        rgb   = rgba[..., :3]
        alpha = rgba[..., 3:4]
        mask  = (alpha > 0.0).astype(np.float32)

        img_np = rgb * mask + bg * (1 - mask)
        img_np = np.clip(img_np, 0, 1)
        img_t  = torch.from_numpy(img_np).permute(2, 0, 1).float()
        return img_np, img_t

    # ── Differentiable render (for adversarial optimisation) ──────────────────
    def render_differentiable(self):
        if self.mesh is None:
            return torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), device=self.device)

        bg_np = load_background()
        bg_t  = torch.from_numpy(bg_np).to(self.device).float()

        angles = self.rot.unsqueeze(0)
        R      = T.euler_angles_to_matrix(angles, convention="XYZ")[0]
        verts  = self.mesh.verts_packed()
        verts  = verts @ R.T + self.pos
        mesh   = self.mesh.update_padded(verts.unsqueeze(0))
        images = self.renderer(mesh, lights=self._make_lights())

        rgba  = images[0]
        rgb   = rgba[..., :3]
        alpha = rgba[..., 3:4]
        mask  = (alpha > 0.0).float()

        img_t = rgb * mask + bg_t * (1 - mask)
        return img_t.permute(2, 0, 1).unsqueeze(0)

    
class MeshScene3D:
    """
    Scene combining a 3D human (body/shirt/pants parts) with a 3D background.
    Supports moving/rotating the human group and adjusting lighting.
    """

    def __init__(self, device, image_size: int = 512):
        self.device = device
        self.image_size = image_size
        self.parts = {}        # {"body": mesh, "shirt": mesh, "pants": mesh}
        self.background = None
        # The environment is deliberately kept as a UV-textured mesh instead
        # of being joined with the vertex-coloured human meshes.  PyTorch3D
        # cannot join those texture types without losing the UV detail.
        self.background_is_textured = False

        # Human-group transform (applied at render time, originals untouched)
        self.pos = torch.zeros(3, device=device)
        self.rot = torch.zeros(3, device=device)  # radians, only Y used for now

        # Lighting
        self.light_pos = torch.tensor([[5.0, 5.0, -5.0]], device=device)
        self.ambient_intensity = torch.tensor(0.6, device=device)
        self.lights = DirectionalLights(
            device=device,
            direction=[[0.0, -1.0, -1.0]],
            ambient_color=self.ambient_intensity.repeat(1, 3),
            diffuse_color=[[0.5, 0.5, 0.5]],
        )

        # Orbit camera state (distance / elevation / azimuth around a target point)
        self.cam_distance = 2.0
        self.cam_elev = 3.0
        self.cam_azim = 90
        self.cam_target = (0.0, 0.0, 0.0)
        self.cam_focal_length = 1.0
        self._camera_fitted = False

        self._renderer = None
        self._renderer_dirty = True

        self.set_camera_orbit(self.cam_distance, self.cam_elev, self.cam_azim, self.cam_target)

    # ------------------------------------------------------------------
    # Texture conversion helper
    # ------------------------------------------------------------------
    def _texturesUV_to_vertex(self, mesh) -> TexturesVertex:
        textures = mesh.textures
        verts = mesh.verts_list()[0]                       # (V, 3)
        faces = mesh.faces_list()[0]                        # (F, 3) vertex indices
        verts_uvs = textures.verts_uvs_list()[0]             # (U, 2)
        faces_uvs = textures.faces_uvs_list()[0]             # (F, 3) uv indices
        maps = textures.maps_padded()[0]                     # (H, W, 3)

        uv = verts_uvs.clone()
        uv[:, 1] = 1.0 - uv[:, 1]
        grid = (uv * 2.0 - 1.0).unsqueeze(0).unsqueeze(0)     # (1, 1, U, 2)
        img = maps.permute(2, 0, 1).unsqueeze(0)              # (1, 3, H, W)
        sampled = F.grid_sample(img, grid, align_corners=True)  # (1, 3, 1, U)
        uv_colors = sampled.squeeze(0).squeeze(1).permute(1, 0)  # (U, 3)

        V = verts.shape[0]
        color_sum = torch.zeros(V, 3, device=verts.device)
        color_count = torch.zeros(V, 1, device=verts.device)
        vert_idx_flat = faces.reshape(-1)          # (F*3,)
        uv_idx_flat = faces_uvs.reshape(-1)        # (F*3,)
        face_colors = uv_colors[uv_idx_flat]       # (F*3, 3)
        color_sum.index_add_(0, vert_idx_flat, face_colors)
        color_count.index_add_(0, vert_idx_flat, torch.ones_like(face_colors[:, :1]))
        color_count = color_count.clamp(min=1.0)
        vertex_colors = color_sum / color_count    # (V, 3) — matches mesh exactly
        return TexturesVertex(verts_features=[vertex_colors])

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_mesh(self, obj_path: str, name: str = None):
        logger.debug("Loading OBJ: %s", obj_path)
        mesh = load_objs_as_meshes(
            [obj_path],
            device=self.device,
            load_textures=True,
        )
        logger.debug("Mesh loaded. Textures: %s", mesh.textures)
        if mesh.textures is not None and hasattr(mesh.textures, "maps_padded"):
            logger.debug("Texture maps shape: %s", mesh.textures.maps_padded().shape)
        if name:
            self.parts[name] = mesh
        return mesh

    def _load_multimaterial_background(self, obj_path: str):
        """Load every material map in an OBJ, not just the first one.

        ``load_objs_as_meshes`` intentionally supports only one texture map
        per OBJ.  The road asset has eight maps, so it previously rendered
        with one incorrect map and then had that map averaged into flat vertex
        colours.  Build a multi-map ``TexturesUV`` object from ``load_obj``
        instead.  Maps are resized to a modest common resolution to keep the
        interactive renderer responsive.
        """
        verts, faces, aux = load_obj(obj_path, load_textures=True, device=self.device)
        texture_images = aux.texture_images or {}
        if not texture_images or faces.textures_idx is None:
            logger.warning("No usable texture maps found for background: %s", obj_path)
            return Meshes(verts=[verts], faces=[faces.verts_idx])

        maps = list(texture_images.values())
        # A UV texture set needs maps with one shared shape.  This retains the
        # actual per-pixel UV mapping (unlike converting textures to vertices)
        # while capping memory use for the eight 4K/8K source images.
        texture_size = 1024
        prepared_maps = []
        for texture_map in maps:
            # Resize before moving the 4K/8K source map to the GPU; otherwise
            # the temporary upload can exhaust VRAM on modest cards.
            image = texture_map[..., :3].float()
            image = image.permute(2, 0, 1).unsqueeze(0)
            image = F.interpolate(
                image,
                size=(texture_size, texture_size),
                mode="bilinear",
                align_corners=False,
            )
            prepared_maps.append(image[0].permute(1, 2, 0).to(self.device))

        map_ids = faces.materials_idx.to(self.device).clamp(min=0)
        if map_ids.max().item() >= len(prepared_maps):
            raise ValueError(
                f"Background material indices exceed loaded texture maps: "
                f"{map_ids.max().item()} >= {len(prepared_maps)}"
            )

        textures = TexturesUV(
            maps=[torch.stack(prepared_maps)],
            verts_uvs=[aux.verts_uvs.to(self.device)],
            faces_uvs=[faces.textures_idx.to(self.device)],
            maps_ids=[map_ids],
        )
        return Meshes(verts=[verts], faces=[faces.verts_idx], textures=textures)

    def load_background(self, obj_path: str, target_extent: float = None, recenter: bool = True):
        mesh = self._load_multimaterial_background(obj_path)

        verts = mesh.verts_list()[0]
        raw_extent = (verts.max(0).values - verts.min(0).values).max().item()
        if target_extent is None:
            target_extent = self._estimate_target_extent()
        scale = target_extent / raw_extent if raw_extent > 1e-8 else 1.0
        verts = verts * scale
        if recenter:
            verts = verts - verts.mean(dim=0)
        self.background = mesh.offset_verts(verts - mesh.verts_list()[0])
        self.background_is_textured = isinstance(self.background.textures, TexturesUV)
        self._renderer_dirty = True
        return scale

    def _estimate_target_extent(self, multiplier: float = 6.0) -> float:
        """Derive a sensible background size from whatever human parts are loaded."""
        if not self.parts:
            return 10.0  # fallback if background is loaded before human parts
        all_verts = torch.cat([m.verts_list()[0] for m in self.parts.values()], dim=0)
        human_extent = (all_verts.max(0).values - all_verts.min(0).values).max().item()
        return max(human_extent * multiplier, 1.0)

    # ------------------------------------------------------------------
    # Color changing
    # ------------------------------------------------------------------
    def set_part_color(self, name: str, rgb):
        """rgb: tuple of 3 floats in [0.0, 1.0]"""
        if name not in self.parts:
            raise KeyError(f"Part '{name}' not found. Loaded: {list(self.parts.keys())}")
        mesh = self.parts[name]
        verts = mesh.verts_list()[0]
        color = torch.tensor(rgb, dtype=torch.float32, device=self.device)
        colors = color.unsqueeze(0).expand(verts.shape[0], 3)
        mesh.textures = TexturesVertex(verts_features=[colors])

    # ------------------------------------------------------------------
    # Transform (position / rotation of the human group)
    # ------------------------------------------------------------------
    def set_position(self, x: float, y: float, z: float):
        self.pos = torch.tensor([x, y, z], device=self.device, dtype=torch.float32)

    def set_rotation_y(self, degrees: float):
        self.rot = torch.tensor([0.0, math.radians(degrees), 0.0], device=self.device)

    def _transformed_part(self, mesh):
        """Return a new Meshes object with the group position/rotation applied."""
        verts = mesh.verts_list()[0]
        angle = self.rot[1]
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        zero = torch.zeros_like(cos_a)
        one = torch.ones_like(cos_a)
        rot_y = torch.stack([
            torch.stack([cos_a, zero, sin_a]),
            torch.stack([zero, one, zero]),
            torch.stack([-sin_a, zero, cos_a]),
        ])
        verts = verts @ rot_y.T + self.pos
        return Meshes(verts=[verts], faces=mesh.faces_list(), textures=mesh.textures)

    # ------------------------------------------------------------------
    # Lighting
    # ------------------------------------------------------------------
    def set_ambient_intensity(self, value: float):
        self.ambient_intensity = torch.tensor(value, device=self.device)

    def set_light_direction(self, x: float, y: float, z: float):
        self.lights.direction = torch.tensor([[x, y, z]], device=self.device)

    def _update_lights(self):
        self.lights.ambient_color = self.ambient_intensity.repeat(1, 3)

    # ------------------------------------------------------------------
    # Debugging / camera fitting
    # ------------------------------------------------------------------
    def debug_scene_info(self):
        """Print bounding boxes of every loaded mesh — run this first when
        something 'isn't showing up'. Compare against the camera's T (~3
        units back along Z by default) to see if geometry is out of frame."""
        for name, mesh in self.parts.items():
            v = mesh.verts_list()[0]
            print(f"[{name}] bounds min={v.min(0).values.tolist()} "
                  f"max={v.max(0).values.tolist()} center={v.mean(0).tolist()}")
        if self.background is not None:
            v = self.background.verts_list()[0]
            print(f"[background] bounds min={v.min(0).values.tolist()} "
                  f"max={v.max(0).values.tolist()} center={v.mean(0).tolist()}")

    def set_camera_orbit(self, distance: float, elev: float, azim: float, at=None, focal_length: float = None):
        at = at if at is not None else self.cam_target
        if focal_length is not None:
            self.cam_focal_length = focal_length
        self.cam_distance, self.cam_elev, self.cam_azim, self.cam_target = distance, elev, azim, at
        R, T = look_at_view_transform(
            dist=distance, elev=elev, azim=azim,
            at=(at,), device=self.device,
        )
        self.camera = PerspectiveCameras(
            focal_length=self.cam_focal_length,
            principal_point=((0.0, 0.0),),
            R=R, T=T,
            device=self.device,
        )
        self._renderer_dirty = True

    def fit_camera_to_scene(self, scene_mesh, margin: float = 0.2):
        verts = torch.cat(
            [m.verts_list()[0] for m in self.parts.values()],
            dim=0,
        )

        bbox_min = verts.min(0).values
        bbox_max = verts.max(0).values

        center = (bbox_min + bbox_max) / 2

        # Put the camera around the chest instead of the feet
        center[1] += 0.3

        extent = (bbox_max - bbox_min).max().item()

        # Leave enough of the environment in frame for the rendered result to
        # read as a scene, rather than a near full-frame portrait.
        distance = extent * 2.6

        self.set_camera_orbit(
            distance=distance,
            elev=10,
            azim=0,
            at=tuple(center.tolist()),
        )
    # ------------------------------------------------------------------
    # Scene assembly
    # ------------------------------------------------------------------
    def get_scene_mesh(self):
        """Merge the human parts for a single render pass.

        UV backgrounds render independently in ``render`` so their texture
        maps survive; vertex-coloured backgrounds remain joinable here.
        """
        human_meshes = [self._transformed_part(m) for m in self.parts.values()]
        all_meshes = list(human_meshes)
        if self.background is not None and not self.background_is_textured:
            all_meshes.append(self.background)
        if not all_meshes:
            return None

        if logger.isEnabledFor(logging.DEBUG):
            for i, mesh in enumerate(all_meshes):
                logger.debug("%d %s", i, type(mesh.textures).__name__)

        scene = join_meshes_as_scene(all_meshes)
        if not self._camera_fitted:
            self.fit_camera_to_scene(scene)
            self._camera_fitted = True
        return scene

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _get_renderer(self, faces_per_pixel: int = 1):
        """Cheap, non-differentiable renderer for interactive/live display.
        Cached and only rebuilt when the camera changes."""
        if self._renderer is None or self._renderer_dirty:
            raster_settings = RasterizationSettings(
                image_size=self.image_size,
                blur_radius=0.0,
                faces_per_pixel=faces_per_pixel,
                cull_backfaces=True,
            )
            self._renderer = MeshRenderer(
                rasterizer=MeshRasterizer(cameras=self.camera, raster_settings=raster_settings),
                shader=HardPhongShader(
                    device=self.device,
                    cameras=self.camera,
                    lights=self.lights,
                    blend_params=BlendParams(background_color=(0.1, 0.1, 0.1)),
                ),
            )
            self._renderer_dirty = False
        return self._renderer

    def _get_renderer(self, faces_per_pixel: int = 1):
        if self._renderer is None or self._renderer_dirty:
            # Count total faces so the coarse rasterizer never overflows
            total_faces = sum(
                m.faces_list()[0].shape[0] for m in self.parts.values()
            )
            if self.background is not None:
                total_faces += self.background.faces_list()[0].shape[0]

            raster_settings = RasterizationSettings(
                image_size=self.image_size,
                blur_radius=0.0,
                faces_per_pixel=faces_per_pixel,
                cull_backfaces=True,
                max_faces_per_bin=max(total_faces, 50_000),
            )
            self._renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=self.camera, raster_settings=raster_settings
                ),
                shader=HardPhongShader(
                    device=self.device,
                    cameras=self.camera,
                    lights=self.lights,
                    blend_params=BlendParams(background_color=(0.1, 0.1, 0.1)),
                ),
            )
            self._renderer_dirty = False
        return self._renderer

    def _get_renderer_differentiable(self, faces_per_pixel: int = 8):
        total_faces = sum(
            m.faces_list()[0].shape[0] for m in self.parts.values()
        )
        if self.background is not None:
            total_faces += self.background.faces_list()[0].shape[0]

        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=1e-5,
            faces_per_pixel=faces_per_pixel,
            cull_backfaces=True,
            max_faces_per_bin=max(total_faces, 50_000),
        )
        return MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.camera, raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=self.camera,
                lights=self.lights,
                blend_params=BlendParams(background_color=(0.1, 0.1, 0.1)),
            ),
        )

    def render(self):
        """Render to a numpy RGB image [H, W, 3] uint8. Safe to call from UI thread."""
        scene_mesh = self.get_scene_mesh()
        if scene_mesh is None and self.background is None:
            return None
        self._update_lights()
        renderer = self._get_renderer()
        with torch.no_grad():
            background_image = (
                renderer(self.background) if self.background_is_textured else None
            )
            foreground_image = renderer(scene_mesh) if scene_mesh is not None else None

        if background_image is None:
            image = foreground_image
        elif foreground_image is None:
            image = background_image
        else:
            # The human should remain visible while retaining the detailed
            # textured environment behind it.  Alpha is the rasterizer's exact
            # silhouette mask, so this avoids a black or grey fallback.
            alpha = foreground_image[..., 3:4]
            image = foreground_image[..., :3] * alpha + background_image[..., :3] * (1 - alpha)
        img = image[0, ..., :3].cpu().numpy()
        return np.clip(img * 255, 0, 255).astype(np.uint8)

    def render_differentiable(self):
        """Keep the computation graph alive for gradient-based attacks."""
        scene_mesh = self.get_scene_mesh()
        if scene_mesh is None and self.background is None:
            return None
        self._update_lights()
        renderer = self._get_renderer_differentiable()
        background_image = (
            renderer(self.background) if self.background_is_textured else None
        )
        foreground_image = renderer(scene_mesh) if scene_mesh is not None else None
        if background_image is None:
            return foreground_image
        if foreground_image is None:
            return background_image
        alpha = foreground_image[..., 3:4]
        rgb = foreground_image[..., :3] * alpha + background_image[..., :3] * (1 - alpha)
        return torch.cat((rgb, torch.ones_like(alpha)), dim=-1)

    def capture_image(self, path: str) -> bool:
        """Render the current scene and save it to disk as a PNG."""
        img_np = self.render()
        if img_np is None:
            return False
        Image.fromarray(img_np).save(path)
        return True
