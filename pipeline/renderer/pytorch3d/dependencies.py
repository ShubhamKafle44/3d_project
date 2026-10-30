"""Optional PyTorch3D imports shared by renderer components."""

try:
    from pytorch3d.io import load_obj, load_objs_as_meshes
    from pytorch3d.structures import join_meshes_as_scene, Meshes
    from pytorch3d.renderer import (
        look_at_view_transform, FoVPerspectiveCameras, PointLights,
        RasterizationSettings, MeshRenderer, MeshRasterizer, SoftPhongShader,
        TexturesUV, TexturesVertex,
    )
    from pytorch3d.renderer.blending import softmax_rgb_blend
    from pytorch3d.renderer.mesh.shading import phong_shading
    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False
    load_obj = load_objs_as_meshes = join_meshes_as_scene = None
    Meshes = PointLights = RasterizationSettings = MeshRenderer = None
    MeshRasterizer = SoftPhongShader = TexturesUV = TexturesVertex = None
    look_at_view_transform = FoVPerspectiveCameras = None
    softmax_rgb_blend = phong_shading = None
