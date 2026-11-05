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
            parameters[2].clamp_(min=config.PROPERTY_BOUNDS["LIGHTING"][0])
        if property_name == "CLOTHING":
            parameters[3].clamp_(0.0, 1.0)
