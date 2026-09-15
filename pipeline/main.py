
import os
import torch

import config
from scene_setup import build_human_scene
from detector import HumanDetectionClassifier
from search import run_3d_appearance_attack, run_adversarial_search, save_image


def main():

    # =========================
    # SETTINGS
    # =========================

    renderer = "pytorch3d"
    # Other value: "pytorch3d", "mitsuba"

    property = "POSITION"
    # Examples: "LIGHTING", "APPEARANCE", "POSITION"

    model = "fasterrcnn_resnet50_fpn_v2"
    # Other value: "retinanet_resnet50_fpn_v2", "fasterrcnn_resnet50_fpn_v2"

    epochs = config.SEARCH["epochs"]

    step_size = config.SEARCH["step_size"]
    # Example: 0.01

    success_threshold = config.SEARCH["success_threshold"]
    # Example: 0.5

    device = "cuda" if torch.cuda.is_available() else "cpu"
  

    out_dir = "outputs"

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("Adversarial Search")
    print(f"Renderer : {renderer}")
    print(f"Detector : {model}")
    attack_label = "3D shirt appearance" if (
        renderer == "pytorch3d" and config.SEARCH["mode"] == "gradient_appearance"
    ) else property
    print(f"Property : {attack_label}")
    print(f"Device   : {device}")
    print("=" * 60)

    gradient_mode = renderer == "pytorch3d" and config.SEARCH["mode"] == "gradient_appearance"
    scene_image_size = (
        config.SEARCH["gradient_image_size"] if gradient_mode
        else config.MITSUBA_IMAGE_SIZE if renderer == "mitsuba"
        else None
    )
    scene = build_human_scene(
        renderer,
        device=device,
        image_size=scene_image_size,
    )

    classifier = HumanDetectionClassifier(
        model_name=model,
        device=device
    )


    # =========================
    # INITIAL RENDER
    # =========================

    initial_img = scene.render()

    if initial_img is None:
        print("ERROR: Initial render failed.")
        return

    save_image(
        initial_img,
        f"{out_dir}/initial_render.png"
    )

    # =========================
    # ADVERSARIAL SEARCH
    # =========================

    if gradient_mode:
        result = run_3d_appearance_attack(
            scene,
            classifier,
            epochs=epochs,
            learning_rate=config.SEARCH["gradient_learning_rate"],
            validate_every=config.SEARCH["gradient_validate_every"],
            detector_input_size=config.SEARCH["gradient_detector_size"],
            success_threshold=success_threshold,
        )
    else:
        result = run_adversarial_search(
            scene,
            classifier,
            property,
            epochs=epochs,
            step_size=step_size,
            success_threshold=success_threshold,
        )

    # =========================
    # SAVE RESULT
    # =========================

    output_path = (
        f"{out_dir}/adversarial_result_"
        f"{renderer}_{property.lower()}.png"
    )

    save_image(result.best_image, output_path)

    print("\nRESULT")
    print(f"Success           : {result.success}")
    print(f"Best step         : {result.best_step}")
    print(f"Human probability : {result.best_prob * 100:.2f}%")
    print(f"Saved              : {output_path}")


if __name__ == "__main__":
    main()
