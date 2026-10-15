
import os
import argparse
import re
from pathlib import Path
import torch

import config
from scene_setup import available_assets, build_human_scene
from detector import HumanDetectionClassifier
from search import run_3d_appearance_attack, run_adversarial_search, save_image


def main():
    parser = argparse.ArgumentParser(description="Run the 3D human detection search")
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.SEARCH["epochs"],
        help=f"Maximum number of search epochs (default: {config.SEARCH['epochs']})",
    )
    parser.add_argument(
        "--model",
        default="human",
        help="OBJ file or model folder under assets/3D_MODELS (default: human)",
    )
    parser.add_argument(
        "--environment",
        default="studio",
        help="Environment OBJ or folder under assets/3D_ENVIRONMENTS; use 'none' for no environment",
    )
    parser.add_argument(
        "--list-assets",
        action="store_true",
        help="List available model and environment names, then exit",
    )
    args = parser.parse_args()
    if args.list_assets:
        models, environments = available_assets()
        print("Models      : " + (", ".join(models) or "(none found)"))
        print("Environments: " + (", ".join(environments) or "(none found)"))
        return
    if args.epochs < 1:
        parser.error("--epochs must be a positive integer")

    # =========================
    # SETTINGS
    # =========================

    renderer = "mitsuba"  # Other value: "mitsuba"
    # Other value: "pytorch3d", "mitsuba"

    # Jointly optimize ground-plane position and yaw, pitch, and roll.
    property = "POSITION"
    # Other targets: "POSITION", "POSE", "LIGHTING", "CLOTHING"

    model = "retinanet_resnet50_fpn_v2"
    # Other value: "retinanet_resnet50_fpn_v2", "fasterrcnn_resnet50_fpn_v2"

    epochs = args.epochs

    step_size = config.SEARCH["step_size"]
    # Example: 0.01

    success_threshold = config.SEARCH["success_threshold"]
    # Example: 0.5

    device = "cuda" if torch.cuda.is_available() else "cpu"
  

    out_dir = "outputs"

    os.makedirs(out_dir, exist_ok=True)

    def safe_name(value):
        label = Path(value).stem if Path(value).suffix.lower() == ".obj" else Path(value).name
        return re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_") or "asset"

    run_name = f"{safe_name(args.model)}_{safe_name(args.environment)}"

    print("=" * 60)
    print("Adversarial Search")
    print(f"Renderer : {renderer}")
    print(f"Detector : {model}")
    print(f"3D model : {args.model}")
    print(f"Environment: {args.environment}")
    gradient_mode = (
        renderer == "pytorch3d"
        and config.SEARCH["mode"] == "gradient_appearance"
        and property == "CLOTHING"
    )
    if gradient_mode and not model.startswith("fasterrcnn_"):
        raise ValueError(
            "gradient_appearance currently requires a Faster R-CNN detector; "
            "use fasterrcnn_resnet50_fpn_v2 or set SEARCH['mode'] to 'random'."
        )
    attack_label = "3D shirt appearance" if gradient_mode else property
    print(f"Property : {attack_label}")
    print(f"Device   : {device}")
    print("=" * 60)

    scene_image_size = (
        config.SEARCH["gradient_image_size"] if gradient_mode
        else config.MITSUBA_IMAGE_SIZE if renderer == "mitsuba"
        else config.PYTORCH3D_IMAGE_SIZE if renderer == "pytorch3d"
        else None
    )
    scene = build_human_scene(
        renderer,
        device=device,
        image_size=scene_image_size,
        model=args.model,
        environment=args.environment,
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
        f"{out_dir}/{run_name}_initial_render.png"
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
        f"{out_dir}/adversarial_result_{run_name}_"
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
