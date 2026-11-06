import torch

import config


def take_gradient_step(scene, classifier, loss_function, optimized_parameters,
                       optimizer, detector_input_size: int, step: int,
                       property_name: str) -> float:
    """Differentiate the detector loss through the renderer and update leaves."""
    optimizer.zero_grad(set_to_none=True)
    differentiable_image = scene.render_differentiable()
    loss = loss_function(differentiable_image, input_size=detector_input_size)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite detector loss at gradient step {step}")
    loss.backward()
    # Mitsuba's point-emitter position is not differentiable in every supported
    # variant/integrator combination. When that happens, estimate its gradient
    # from the same detector loss with a small central difference so LIGHTING
    # search still moves the light in world space.
    if property_name == "LIGHTING" and len(optimized_parameters) > 1:
        light_position = optimized_parameters[1]
        if light_position.grad is None or not torch.isfinite(light_position.grad).all() \
                or torch.linalg.vector_norm(light_position.grad).item() < 1e-8:
            light_position.grad = _finite_difference_light_position_gradient(
                scene, classifier, loss_function, light_position,
                detector_input_size,
            )
    if any(
        parameter.grad is None or not torch.isfinite(parameter.grad).all()
        for parameter in optimized_parameters
    ):
        raise RuntimeError(
            f"{type(scene).__name__} did not produce a finite "
            f"{property_name.lower()} gradient at step {step}"
        )

    loss_value = loss.detach().item()
    # Mitsuba/Dr.Jit can reuse CUDA blocks once Torch releases this graph.
    del differentiable_image, loss
    if classifier.device.type == "cuda":
        torch.cuda.empty_cache()
    optimizer.step()
    return loss_value


def _finite_difference_light_position_gradient(
    scene, classifier, loss_function, light_position, detector_input_size: int,
) -> torch.Tensor:
    """Estimate detector-loss derivatives for a stationary point light."""
    epsilon = 0.05
    original = light_position.detach().clone()
    gradient = torch.zeros_like(light_position)
    try:
        with torch.no_grad():
            for axis in range(light_position.numel()):
                light_position.copy_(original)
                light_position[axis] += epsilon
                plus = loss_function(
                    scene.render_differentiable(), input_size=detector_input_size
                ).detach()
                light_position.copy_(original)
                light_position[axis] -= epsilon
                minus = loss_function(
                    scene.render_differentiable(), input_size=detector_input_size
                ).detach()
                gradient[axis] = (plus - minus) / (2.0 * epsilon)
    finally:
        with torch.no_grad():
            light_position.copy_(original)
    return gradient


def apply_parameter_bounds(property_name: str, parameters, initial_position) -> None:
    """Apply the per-property bounds and fixed-height rule after an update."""
    with torch.no_grad():
        if property_name in ("POSITION", "POSE"):
            position_parameter = parameters[0]
            position_parameter[1] = torch.as_tensor(
                initial_position[1],
                device=position_parameter.device,
                dtype=position_parameter.dtype,
            )
        if property_name in ("ROTATION", "POSE"):
            parameters[1][1:].clamp_(-45.0, 45.0)
        if property_name == "LIGHTING":
            parameters[2].clamp_(*config.PROPERTY_BOUNDS["LIGHTING"])
            bounds = getattr(config, "LIGHT_POSITION_BOUNDS", ((-8.0, 8.0), (0.5, 10.0), (-8.0, 8.0)))
            for axis, (lower, upper) in enumerate(bounds):
                parameters[4][axis].clamp_(lower, upper)
        if property_name == "CLOTHING":
            parameters[3].clamp_(0.0, 1.0)
