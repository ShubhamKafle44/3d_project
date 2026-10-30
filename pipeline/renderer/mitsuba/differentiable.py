from __future__ import annotations

import numpy as np
import mitsuba as mi
import drjit as dr
import torch

from .utils import torch_from_numpy

if mi.variant() is None:
    import config
    mi.set_variant(config.MITSUBA_VARIANT)


class _MitsubaTorchRender(torch.autograd.Function):
    """Bridge Torch cotangents to the Dr.Jit graph retained by mi.render."""

    @staticmethod
    def forward(ctx, scene, position, rotation, lighting, shirt_color):
        """Render the scene and retain Dr.Jit parameters for the backward pass."""

        # We need to convert the pytorchTensor values to numpy Arrays
        pos = position.detach().cpu().numpy().astype(np.float32)
        rot_deg = rotation.detach().cpu().numpy().astype(np.float32)
        intensity = float(lighting.detach().cpu())


        scene_dict = scene._build_scene_dict(differentiable=True)


        #Apply transformation to the Model
        transform = (
            mi.ScalarTransform4f()
            .translate(pos.tolist())
            .rotate([0, 0, 1], float(rot_deg[0]))
            .rotate([1, 0, 0], float(rot_deg[1]))
            .rotate([0, 1, 0], float(rot_deg[2]))
        )
        part_keys = scene._part_shape_keys()
        for key in part_keys:
            scene_dict[key]["to_world"] = transform

        #Change the light
        scene_dict["street_light"]["intensity"]["value"] = [
            20.0 * intensity * float(channel) for channel in scene.light_color
        ]

        #Change the textures
        if shirt_color is not None:
            color = shirt_color.detach().cpu().numpy().astype(np.float32).tolist()
            for key, _, _ in scene._part_shapes.get("shirt", []):
                # The attack uses a uniform differentiable diffuse color for
                # the shirt while preserving the shape and lighting graph.
                scene_dict[key]["bsdf"] = {
                    "type": "diffuse",
                    "reflectance": {"type": "rgb", "value": color},
                }

        #LOAD THE SCENE        
        mitsuba_scene = mi.load_dict(scene_dict)


        #GET PARAMETERS WHOSE GRADIENT NEEDS TO BE CALCULATED
        params = mi.traverse(mitsuba_scene)
        # Mitsuba does not expose vertex_positions for every OBJ shape
        # (for example, some degenerate or specially loaded meshes). Only
        # enable gradients for parameters present in this scene traversal.
        requested_vertex_keys = [f"{key}.vertex_positions" for key in part_keys]
        vertex_keys = [key for key in requested_vertex_keys if key in params]
        if not vertex_keys:
            raise RuntimeError(
                "Mitsuba did not expose vertex_positions for any subject mesh; "
                "cannot differentiate scene position."
            )

        #CALCULATE GRADIENTS FOR VERTEX POSITIONS
        for key in vertex_keys:
            dr.enable_grad(params[key])

        #CALCULATE GRADIENTS FOR LIGHT INTENSITY
        light_key = "street_light.intensity.value"
        dr.enable_grad(params[light_key])

        #CALCULATE GRADIENTS FOR TEXTURES
        shirt_keys = []
        if shirt_color is not None:
            shirt_keys = [
                f"{key}.bsdf.reflectance.value"
                for key, _, _ in scene._part_shapes.get("shirt", [])
                if f"{key}.bsdf.reflectance.value" in params
            ]
            if not shirt_keys:
                raise RuntimeError(
                    "Mitsuba did not expose a shirt reflectance parameter; "
                    "cannot differentiate clothing color."
                )
            for key in shirt_keys:
                dr.enable_grad(params[key])

        #UPDATE THE PARAMETERS        
        params.update()

        #RENDER IMAGE
        image = mi.render(mitsuba_scene, params=params, spp=scene._spp)

        #CONVERT IMAGE TO NUMPY ARRAY
        image_np = np.array(image, copy=True)

        #SAVE CONTEXT 
        ctx.mitsuba_image = image
        ctx.params = params
        ctx.vertex_keys = vertex_keys
        ctx.light_key = light_key
        ctx.light_rgb = np.asarray(scene.light_color, dtype=np.float32)
        ctx.shirt_keys = shirt_keys
        ctx.has_shirt_color = shirt_color is not None
        ctx.rot_deg = rot_deg
        ctx.save_for_backward(torch_from_numpy(image_np))

        #NORMALIZE THE PIXEL VALUES
        return torch_from_numpy(image_np).permute(2, 0, 1).clamp(0.0, 1.0)




    @staticmethod
    def backward(ctx, grad_output):
        """Map image gradients back to position, rotation, light, and shirt color."""
        image_t, = ctx.saved_tensors

        # CHANGE THE GRADIENTS DIMENSIONS
        grad = grad_output.permute(1, 2, 0).contiguous()

        #NORMALIZE THE GRADIENTS
        grad = grad * ((image_t >= 0.0) & (image_t <= 1.0)).to(grad.dtype)
        grad_np = grad.detach().cpu().numpy().astype(np.float32)

        #COMPUTE GRADIENTS
        dr.set_grad(ctx.mitsuba_image, mi.TensorXf(grad_np))

        #PERFORM BACKWARD PROPAGATION
        dr.backward(ctx.mitsuba_image)


        #COMPUTE GRADIENTS
        position_grad = np.zeros(3, dtype=np.float32)
        for key in ctx.vertex_keys:
            vertex_grad = np.array(dr.grad(ctx.params[key]), copy=False).reshape(-1, 3)
            position_grad += vertex_grad.sum(axis=0)
        # Mesh vertex parameters are local-space values, but the optimized
        # position translates the rotated object in world space.
        yaw, pitch, roll = np.radians(ctx.rot_deg)
        cz, sz = np.cos(yaw), np.sin(yaw)
        cx, sx = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(roll), np.sin(roll)
        rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        position_grad = (rz @ rx @ ry) @ position_grad
        radiance_grad = np.array(dr.grad(ctx.params[ctx.light_key]), copy=False).reshape(-1, 3)
        radiance_grad = radiance_grad.sum(axis=0)
        scale = 20.0 * ctx.light_rgb
        lighting_grad = float(np.dot(radiance_grad, scale))
    

        shirt_grad = np.zeros(3, dtype=np.float32)
        for key in ctx.shirt_keys:
            shirt_grad += np.array(dr.grad(ctx.params[key]), copy=False).reshape(-1, 3).sum(axis=0)
        # The vertex-position cotangents also provide derivatives for the
        # Euler angles used by the static Mitsuba object transform.
        rotation_grad = np.zeros(3, dtype=np.float32)
        for key in ctx.vertex_keys:
            local_vertices = np.array(ctx.params[key], copy=False).reshape(-1, 3)
            local_grad = np.array(dr.grad(ctx.params[key]), copy=False).reshape(-1, 3)
            yaw, pitch, roll = np.radians(ctx.rot_deg)
            cz, sz = np.cos(yaw), np.sin(yaw)
            cx, sx = np.cos(pitch), np.sin(pitch)
            cy, sy = np.cos(roll), np.sin(roll)
            rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
            rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
            ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
            drz = np.array([[-sz, -cz, 0], [cz, -sz, 0], [0, 0, 0]], dtype=np.float32)
            drx = np.array([[0, 0, 0], [0, -sx, -cx], [0, cx, -sx]], dtype=np.float32)
            dry = np.array([[-sy, 0, cy], [0, 0, 0], [-cy, 0, -sy]], dtype=np.float32)
            derivatives = (drz @ rx @ ry, rz @ drx @ ry, rz @ rx @ dry)
            rotation_matrix = rz @ rx @ ry
            for index, derivative in enumerate(derivatives):
                local_delta = local_vertices @ (rotation_matrix.T @ derivative).T
                rotation_grad[index] += np.sum(local_grad * local_delta) * (np.pi / 180.0)

        return (
            None,
            torch_from_numpy(position_grad).to(grad_output.device),
            torch_from_numpy(rotation_grad).to(grad_output.device),
            torch.tensor(lighting_grad, dtype=grad_output.dtype, device=grad_output.device),
            (
                torch_from_numpy(shirt_grad).to(grad_output.device)
                if ctx.has_shirt_color
                else None
            ),
        )


