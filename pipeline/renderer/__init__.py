from .base import DifferentiableScene


def build_scene(backend: str, device: str , image_size) -> DifferentiableScene:


    """Factory: build a scene using the requested backend ('pytorch3d' or 'mitsuba')."""
    backend = backend.lower()


    if backend == "pytorch3d":
        from .pytorch3d_renderer import PyTorch3DScene
        return PyTorch3DScene(device=device, image_size=image_size)

    
    elif backend == "mitsuba":
        from .mitsuba_renderer import MitsubaScene
        return MitsubaScene(device=device, image_size=image_size)

    
    else:
        raise ValueError(f"Unknown renderer backend: {backend!r} (use 'pytorch3d' or 'mitsuba')")

