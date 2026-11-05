import numpy as np
import torch


def enforce_scene_constraints(
    scene, property_name: str, parameters, optimized_parameters,
    previous_values, optimizer, exploration_rng,
) -> None:
    """Keep position and pose updates within the valid camera/scene region."""
    if property_name not in ("POSITION", "POSE", "ROTATION"):
        return

    with torch.no_grad():
        if property_name == "POSITION":
            position_parameter = parameters[0]
            old_position = previous_values[0]
            new_position = position_parameter.detach().clone()

            if not scene.is_position_valid(new_position.cpu().numpy()):
                low, high = 0.0, 1.0
                for _ in range(25):
                    mid = (low + high) / 2.0
                    test_position = old_position + mid * (new_position - old_position)
                    position_parameter.copy_(test_position)
                    if scene.is_position_valid(position_parameter.detach().cpu().numpy()):
                        low = mid
                    else:
                        high = mid

                position_parameter.copy_(old_position + low * (new_position - old_position))

                if low < 0.99 and position_parameter in optimizer.state:
                    param_state = optimizer.state[position_parameter]
                    if "exp_avg" in param_state:
                        param_state["exp_avg"].zero_()
                    if "exp_avg_sq" in param_state:
                        param_state["exp_avg_sq"].zero_()

                boundary_position = position_parameter.detach().cpu().numpy().copy()
                outward = new_position.cpu().numpy() - old_position.cpu().numpy()
                horizontal_outward = outward.copy()
                horizontal_outward[1] = 0.0
                norm = np.linalg.norm(horizontal_outward)
                inward = -horizontal_outward / norm if norm > 1e-8 else np.zeros(3, dtype=np.float32)

                probes = [boundary_position + inward * distance for distance in (0.08, 0.16, 0.32, 0.5)]
                for _ in range(32):
                    direction = exploration_rng.normal(size=3).astype(np.float32)
                    direction[1] = 0.0
                    direction_norm = np.linalg.norm(direction)
                    if direction_norm <= 1e-8:
                        continue
                    direction /= direction_norm
                    direction = 0.65 * inward + 0.75 * direction
                    direction[1] = 0.0
                    direction_norm = np.linalg.norm(direction)
                    if direction_norm > 1e-8:
                        direction /= direction_norm
                        probes.append(boundary_position + direction * exploration_rng.uniform(0.08, 0.5))

                valid_probes = [candidate for candidate in probes if scene.is_position_valid(candidate)]
                if valid_probes:
                    destination = max(
                        valid_probes,
                        key=lambda candidate: np.linalg.norm(candidate - boundary_position),
                    )
                    position_parameter.copy_(torch.as_tensor(
                        destination,
                        device=position_parameter.device,
                        dtype=position_parameter.dtype,
                    ))

                for group in optimizer.param_groups:
                    group["lr"] = max(group["lr"] * 0.5, 0.02)
                optimizer.state[position_parameter].clear()

        elif not scene.is_position_valid(scene.get_position()):
            for parameter, previous_value in zip(optimized_parameters, previous_values):
                parameter.copy_(previous_value)
