
import os
import torch

import config
from scene_setup import build_human_scene, auto_zoom_to_subject
from detector import HumanDetectionClassifier
from search import run_adversarial_search, save_image


def main():

    # =========================
    # SETTINGS
    # =========================

    renderer = "mitsuba"
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
    print(f"Property : {property}")
    print(f"Device   : {device}")
    print("=" * 60)

    scene = build_human_scene(renderer, device=device)

    classifier = HumanDetectionClassifier(
        model_name=model,
        device=device
    )

    # Automatically position the camera around the human
    coverage = auto_zoom_to_subject(scene)
    print(f"Subject coverage: {coverage * 100:.1f}%")

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