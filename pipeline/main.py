import re
import sys
import math
from datetime import datetime
from pathlib import Path

import torch

import config
from detector import ObjectDetectionClassifier
from scene_setup import  build_scene_layout
from search import (
    PROPERTIES,
    run_gradient_descent,
    save_image,
)


def _safe_name(value):
    path = Path(value)
    label = path.stem if path.suffix.lower() == ".obj" else path.name
    return re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_") or "asset"


def main():
    epochs = config.SEARCH["epochs"]
    learning_rate = config.SEARCH["gradient_learning_rate"]

    # Allow long runs to be started with an epoch override.
    if len(sys.argv) > 1:
        i = 1
        while i < len(sys.argv):
            if sys.argv[i] == "--epochs":
                if i + 1 >= len(sys.argv):
                    raise ValueError("--epochs requires a value")
                try:
                    epochs = int(sys.argv[i + 1])
                except ValueError:
                    raise ValueError("--epochs must be a positive integer")
                i += 2
            elif sys.argv[i] == "--learning-rate":
                if i + 1 >= len(sys.argv):
                    raise ValueError("--learning-rate requires a value")
                try:
                    learning_rate = float(sys.argv[i + 1])
                except ValueError as error:
                    raise ValueError("--learning-rate must be a positive number") from error
                i += 2
            else:
                i += 1

    # =========================
    # SETTINGS
    # =========================

    # "pytorch3d" or "mitsuba"
    renderer = "mitsuba"

    # Other targets: "POSITION", "ROTATION", "POSE", "LIGHTING", "CLOTHING"
    property = "POSITION"

    #DEFINE MODELS
    model = "human"
    environment = "studio"


    detector_name = "fasterrcnn_resnet50_fpn_v2"
    detector_target = "person"


    success_threshold = config.SEARCH["success_threshold"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scene_image_size = config.SEARCH["gradient_image_size"]



    if renderer not in {"pytorch3d", "mitsuba"}:
        raise ValueError("renderer must be 'pytorch3d' or 'mitsuba'")
    if property not in PROPERTIES:
        raise ValueError(f"property must be one of {PROPERTIES}, got {property!r}")

    if epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning rate must be a positive finite number")



    output_root = Path(__file__).resolve().parent / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    # NAME OF THE IMAGE
    run_name = f"{_safe_name(model)}_{_safe_name(environment)}" 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    learning_rate_label = f"{learning_rate:g}".replace(".", "p")
    out_dir = output_root / f"{timestamp}_e{epochs}_lr{learning_rate_label}_{run_name}"
    out_dir.mkdir()



    # OUTPUT FORMAT
    print("=" * 60)
    print("Adversarial Search")
    print(f"Renderer : {renderer}")
    print(f"Detector : {detector_name}")
    print(f"3D model : {model}")
    print(f"Environment: {environment}")
    print(f"Property : {property}")
    print(f"Learning rate: {learning_rate:g}")
    print("Search mode: gradient")
    print(f"Device   : {device}")
    print("=" * 60)



   #BUILD THE SCENE WITH 3d MODEL AND THE ENVIRONMENT
    scene = build_scene_layout(
        renderer,
        device=device,
        image_size=scene_image_size,
        model=model,
        environment=environment,
    )


    #DEFINE THE CLASSIFIER
    classifier = ObjectDetectionClassifier(
        model_name= detector_name,
        target_class=detector_target,
        device=device,
    )


    # =========================
    # INITIAL RENDER
    # =========================

    initial_img = scene.render()

    if initial_img is None:
        print("ERROR: Initial render failed.")
        return

    save_image(initial_img, str(out_dir / f"{run_name}_initial_render.png"))






    # =========================
    # ADVERSARIAL SEARCH
    # =========================

    result = run_gradient_descent(
        scene,
        classifier,
        property,
        epochs=epochs,
        learning_rate=learning_rate,
        validate_every=config.SEARCH["gradient_validate_every"], # THIS BUILDS THE IMAGE AFTER EVERY [gradient_validate_every] value.
        detector_input_size=config.SEARCH["gradient_detector_size"],
        success_threshold=success_threshold,
        snapshot_dir=str(out_dir / "snapshots"),
        snapshot_every=10,
        metrics_csv=str(out_dir / "metrics.csv"),
    )



    # =========================
    # SAVE RESULT
    # =========================

    output_path = (
        out_dir / f"adversarial_result_{run_name}_"
        f"{renderer}_{property.lower()}.png"
    )

    save_image(result.best_image, str(output_path))

    print("\nRESULT")
    print(f"Success           : {result.success}")
    print(f"Best step         : {result.best_step}")
    print(f"{classifier.target_name} probability : {result.best_prob * 100:.2f}%")
    print(f"Saved              : {output_path}")


if __name__ == "__main__":
    main()
