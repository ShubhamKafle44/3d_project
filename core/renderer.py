from __future__ import annotations

import logging
import numpy as np
import torch

from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    look_at_view_transform,
    Materials
)

from config import CAMERA_AZIM, CAMERA_DIST, CAMERA_ELEV, IMAGE_SIZE, OBJ_PATH
from core.mesh_loader import load_mesh_from_file
from core.background import load_background  
import pytorch3d.transforms as T

logger = logging.getLogger(__name__)


class MeshScene:
    """Wraps a PyTorch3D renderer and exposes differentiable scene parameters."""

    def __init__(self, device: torch.device):
        self.device = device

        # Placeholder for our 3D object once it's loaded
        self.mesh = None

        # Set up the virtual camera and the engine that draws the 3D model
        self.renderer, self.cameras = self._build_renderer()

        # Load the 3D model (like a Jeep or a teapot) if a path is provided in config
        if OBJ_PATH:
            self.load_mesh(OBJ_PATH)

        # These are the "knobs" the model  can turn to trick the classifier:
        # pos: X, Y, Z position | rot: Rotation angles | light_pos: Where the sun is
        self.pos = torch.zeros(3, device=device, requires_grad=True)
        self.rot = torch.zeros(3, device=device, requires_grad=True)
        self.light_pos = torch.tensor([[0.0, 1.0, -2.0]], device=device, requires_grad=True)
        self.ambient_intensity = torch.tensor([0.5], device=device, requires_grad=True)

    # ── Renderer ─────────────────────────────────────────────

    def _build_renderer(self):
        """Sets up the mathematical 'eyes' and 'lighting' of our 3D world."""
        # Calculate the camera position using distance, elevation, and side-to-side angle
        R, T = look_at_view_transform(
            dist=CAMERA_DIST,
            elev=CAMERA_ELEV,
            azim=CAMERA_AZIM
        )

        # Perspective camera makes things further away look smaller (like real life)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)

        # Settings for how to turn 3D math into 2D pixels
        raster_settings = RasterizationSettings(
            image_size=IMAGE_SIZE,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        # Default lighting setup: Point light (like a lightbulb) with colors
        lights = PointLights(
            device=self.device,
            location=[[0.0, 2.0, -2.0]],
            ambient_color=[[0.5, 0.5, 0.5]], # Base light in shadows
            diffuse_color=[[0.8, 0.8, 0.8]], # Main light hitting surfaces
            specular_color=[[0.3, 0.3, 0.3]], # The 'shiny' highlights
        )

        # Define how shiny or dull the 3D surface is
        materials = Materials(
            device=self.device,
            shininess=64.0
        )

        # Combine everything into one rendering engine
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings,
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=cameras,
                lights=lights,
                materials=materials,
                ),
        )

        return renderer, cameras

    # ── Mesh loading ─────────────────────────────────────────

    def load_mesh(self, path: str) -> None:
        """Loads a 3D model file and tells the logger about it."""
        self.mesh = load_mesh_from_file(path, self.device)
        logger.info("Mesh loaded: %s", path)

    # ── Lighting ─────────────────────────────────────────────

    def _make_lights(self) -> PointLights:
        """Dynamic light builder that respects the current ambient intensity knob."""
        # Expand the single intensity value into three (Red, Green, Blue)
        ambient_color = self.ambient_intensity.expand(3).unsqueeze(0)

        return PointLights(
            device=self.device,
            location=self.light_pos,
            ambient_color=ambient_color,
        )

    # ── Blank fallback ───────────────────────────────────────

    def _blank(self):
        """Returns a black void if there's no mesh to render."""
        blank_np = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
        blank_t = torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE))
        return blank_np, blank_t

    # ── UI Render (NO gradients) ────────────────────────────

    def render(self):
        """Generates a clean 2D image for the user interface to display."""
        if self.mesh is None:
            return self._blank()

        bg = load_background()  # Grab the background photo

        with torch.no_grad(): # We don't need to track math history for a simple UI preview
            # Convert rotation angles into a 3D rotation matrix
            angles = self.rot.unsqueeze(0)
            R = T.euler_angles_to_matrix(angles, convention="XYZ")[0]

            # Grab the raw 3D points and spin/move them based on current settings
            verts = self.mesh.verts_packed()
            verts = verts @ R.T + self.pos

            # Update the mesh and draw it
            mesh = self.mesh.update_padded(verts.unsqueeze(0))
            images = self.renderer(mesh, lights=self._make_lights())

        # Pull the pixels out of the GPU and into regular memory
        rgba = images[0].cpu().numpy()   # (H, W, 4) - includes Red, Green, Blue, Alpha
        rgb = rgba[..., :3]
        alpha = rgba[..., 3:4] 

        # --- THE FIX: Hard Mask Composite ---
        # Instead of 'ghostly' semi-transparency, we force pixels to be either fully on or off
        mask = (alpha > 0.0).astype(np.float32)

        # Layer the model over the background image
        img_np = rgb * mask + bg * (1 - mask)
        img_np = np.clip(img_np, 0, 1) # Keep colors in the valid 0.0 - 1.0 range

        # Prep the image for the classifier as a Tensor
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float()

        return img_np, img_t
    
    def render_differentiable(self):
        """
        Differentiable version of the render loop for adversarial optimization.
        This version keeps the 'math trail' alive so the model can learn from it.
        """
        if self.mesh is None:
            return torch.zeros(
                (1, 3, IMAGE_SIZE, IMAGE_SIZE),
                device=self.device
            )

        # 1. Background setup: Move background photo to the GPU
        bg_np = load_background() 
        bg_t = torch.from_numpy(bg_np).to(self.device).float() 

        # 2. Transform the mesh: Apply the current rotation and position
        angles = self.rot.unsqueeze(0)
        R = T.euler_angles_to_matrix(angles, convention="XYZ")[0]
        
        verts = self.mesh.verts_packed()
        verts = verts @ R.T + self.pos
        
        # We 'update' the mesh here which allows gradients to flow back to self.pos and self.rot
        mesh = self.mesh.update_padded(verts.unsqueeze(0))

        # 3. Render: Generate the image using the soft shader (which is gradient-friendly)
        images = self.renderer(mesh, lights=self._make_lights()) 
        
        rgba = images[0]       
        rgb = rgba[..., :3]    
        alpha = rgba[..., 3:4] 

        # 4. Differentiable Blending: A fancy way of layering images that math can understand
        mask = (alpha > 0.0).float()
        img_t = rgb * mask + bg_t * (1 - mask)

        # 5. Reshape to [Batch, Channel, Height, Width] for the neural network
        return img_t.permute(2, 0, 1).unsqueeze(0)