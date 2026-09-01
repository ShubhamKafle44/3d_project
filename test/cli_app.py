import sys

import numpy as np
import torch
import torch.nn as nn

import config as config
from core.classifier import ImageClassifier
from core.renderer import MeshScene

# ── Display helpers ──────────────────────────────────────────────────────────


def print_top5(top5):
    print("\n  Top-5 predictions:")
    for i, (label, prob) in enumerate(top5, start=1):
        bar = "█" * int(prob * 30)
        print(f"    [{i}] {label:<32s}  {prob*100:5.1f}%  {bar}")
    print()


def param_values(scene, prop):
    with torch.no_grad():
        if prop == "position":
            v = scene.pos.cpu().tolist()
            return f"pos=({v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f})"
        elif prop == "rotation":
            v = scene.rot.cpu().tolist()
            return f"rot=({v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f})"
        elif prop == "lighting":
            v = scene.light_pos[0].cpu().tolist()
            return f"light=({v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f})"
        elif prop == "ambient":
            return f"ambient={scene.ambient_intensity.item():+.3f}"
        else:
            p = scene.pos.cpu().tolist()
            r = scene.rot.cpu().tolist()
            l = scene.light_pos[0].cpu().tolist()
            a = scene.ambient_intensity.item()
            return (
                f"pos=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})  "
                f"rot=({r[0]:+.3f},{r[1]:+.3f},{r[2]:+.3f})  "
                f"light=({l[0]:+.3f},{l[1]:+.3f},{l[2]:+.3f})  "
                f"amb={a:+.3f}"
            )


# ── Selection helpers ────────────────────────────────────────────────────────


def pick_from_list(title, options):
    """Show a numbered menu and return the chosen (label, value) tuple."""
    print(f"\n{title}")
    for i, (label, _) in enumerate(options, start=1):
        print(f"  [{i}] {label}")
    while True:
        raw = input(f"\n  Enter choice (1-{len(options)}): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            idx = int(raw) - 1
            print(f"  → Selected: {options[idx][0]}")
            return options[idx]
        print(f"  Invalid — enter a number between 1 and {len(options)}.")


def pick_target_class(top5, classifier):
    """
    Entering 1-5 picks directly from the top-5 list shown on screen.
    Entering 's' opens keyword / raw-index search.
    """
    while True:
        print("  Enter 1-5 to pick from the list above.")
        print("  Enter 's' to search by keyword or raw class index (0-999).")
        raw = input("\n  Your choice: ").strip().lower()

        if raw in ("1", "2", "3", "4", "5"):
            rank = int(raw) - 1
            label, prob = top5[rank]
            target_idx = classifier.categories.index(label)
            print(f"\n  → Selected: [{target_idx}] '{label}'  ({prob*100:.1f}%)")
            return target_idx, label

        elif raw == "s":
            while True:
                query = input("  Search (keyword or 0-999 index): ").strip()
                if query.isdigit():
                    idx = int(query)
                    if 0 <= idx < len(classifier.categories):
                        name = classifier.categories[idx]
                        print(f"  → Selected: [{idx}] '{name}'")
                        return idx, name
                    print(f"  Index must be 0–{len(classifier.categories)-1}.")
                else:
                    matches = [
                        (i, lbl)
                        for i, lbl in enumerate(classifier.categories)
                        if query.lower() in lbl.lower()
                    ]
                    if not matches:
                        print(f"  No class matching '{query}'. Try again.")
                    elif len(matches) == 1:
                        idx, name = matches[0]
                        print(f"  → Found: [{idx}] '{name}'")
                        return idx, name
                    else:
                        print("  Multiple matches:")
                        for idx, lbl in matches[:10]:
                            print(f"    [{idx}] {lbl}")
                        print("  Enter the index number to pick one.")
        else:
            print("  Please enter 1, 2, 3, 4, 5 or 's'.\n")


def render_and_classify(scene, classifier):
    img_np, _ = scene.render()
    top5 = classifier.classify(torch.from_numpy(img_np).to(config.device))
    return img_np, top5


# ── Main ──────────────────────────────────────────────────────────────────────


def run_cli():
    print("=" * 60)
    print("  Adversarial 3-D Scene Optimisation ")
    print("=" * 60)

    # ── 1. Pick 3-D model ─────────────────────────────────────────────────
    _, obj_path = pick_from_list("Select 3D Model:", config.AVAILABLE_3D_MODELS)

    print("\nLoading scene …")
    scene = MeshScene(meshes=None, device=config.device)
    scene.load_mesh(obj_path)  # load the chosen model (resets pose too)

    # ── 2. Pick background image ─────────────────────────────────────────
    _, bg_path = pick_from_list("Select Background Image:", config.AVAILABLE_IMAGES)
    config.BACKGROUND_PATH = bg_path
    print(f"  → Background set to: {bg_path}")

    # ── 3. Pick vision model ──────────────────────────────────────────────
    _, model_key = pick_from_list("Select Vision Model:", config.AVAILABLE_VISION_MODELS)
    print(f"\nLoading vision model '{model_key}' …")
    classifier = ImageClassifier(model_name=model_key)
    print(f"  Device: {config.device}")

    # ── Initial classification ───────────────────────────────────────────
    print("\nInitial classification of the scene:")
    img_np, top5 = render_and_classify(scene, classifier)
    print_top5(top5)
    initial_class, initial_prob = top5[0]
    print(f"  Currently detected as: '{initial_class}'  ({initial_prob*100:.1f}%)\n")

    # ── 4. Pick scene property to change ─────────────────────────────────
    props = {
        "1": "position",
        "2": "rotation",
        "3": "lighting",
        "4": "ambient",
        "5": "all",
    }
    print("Which property do you want to change?")
    for k, v in props.items():
        print(f"  [{k}] {v.capitalize()}")

    while True:
        choice = input("\nEnter choice (1-5): ").strip()
        if choice in props:
            prop = props[choice]
            break
        print("  Invalid — enter 1, 2, 3, 4 or 5.")
    print(f"\n  → Will optimise: {prop.upper()}")

    # ── 5. Pick target class ──────────────────────────────────────────────
    print("\nChoose the class you want the model to use into predicting:")
    print_top5(top5)
    target_idx, target_name = pick_target_class(top5, classifier)

    # ── 6. Learning rate ───────────────────────────────────────────────────
    lr_raw = input("\nLearning rate (press Enter for default 0.05): ").strip()
    lr = float(lr_raw) if lr_raw else 0.05
    epochs = 100

    # ── Build optimiser over chosen property only ─────────────────────────
    param_map = {
        "position": [scene.pos],
        "rotation": [scene.rot],
        "lighting": [scene.light_pos],
        "ambient": [scene.ambient_intensity],
        "all": [scene.pos, scene.rot, scene.light_pos, scene.ambient_intensity],
    }
    optimiser = torch.optim.Adam(param_map[prop], lr=lr)
    target_t = torch.tensor([target_idx], device=config.device)

    # ── Run optimisation ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Starting {epochs}-epoch adversarial optimisation")
    print(f"  3D Model   : {obj_path}")
    print(f"  Background : {bg_path}")
    print(f"  Vision     : {model_key}")
    print(f"  Property   : {prop.upper()}")
    print(f"  Target     : [{target_idx}] {target_name}")
    print(f"  LR         : {lr}")
    print(f"  Success at : target prob ≥ {config.SUCCESS_THRESHOLD*100:.0f}%")
    print("=" * 60)
    print(
        f"\n{'Step':>4}  {'Loss':>7}  {'TargetProb':>10}  "
        f"{'Top Predicted Class':<32}  Parameters"
    )
    print("-" * 110)

    target_prob = 0.0
    success = False

    for step in range(1, epochs + 1):
        optimiser.zero_grad()

        img_t = scene.render_differentiable().float()
        logits = classifier.classify_for_grad(img_t)
        loss = nn.CrossEntropyLoss()(logits, target_t)
        loss.backward()
        optimiser.step()

        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)[0]
            target_prob = probs[target_idx].item()
            top_idx = probs.argmax().item()
            top_label = classifier.categories[top_idx]

        pvals = param_values(scene, prop)
        print(
            f"{step:>4}  {loss.item():>7.4f}  {target_prob*100:>9.2f}%  "
            f"{top_label[:32]:<32}  {pvals}"
        )

        if target_prob >= config.SUCCESS_THRESHOLD:
            success = True
            print("\n" + "=" * 60)
            print(f"  ✓ SUCCESS at step {step}!")
            print(f"  Model now predicts '{target_name}' with {target_prob*100:.1f}% confidence.")
            break

    # ── Final summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"  3D Model            : {obj_path}")
    print(f"  Background          : {bg_path}")
    print(f"  Vision model        : {model_key}")
    print(f"  Property changed    : {prop.upper()}")
    print(f"  Original prediction : '{initial_class}'  ({initial_prob*100:.1f}%)")
    print(f"  Target class        : [{target_idx}] '{target_name}'")

    if success:
        print(f"  Result              : SUCCESS — model targeted wrong class!")
    else:
        print(f"  Result              : UNSUCCESSFUL after {epochs} steps.")
        print(f"  Best target prob    : {target_prob*100:.2f}%")

    print(f"\n  Final parameter values ({prop.upper()}):")
    print(f"    {param_values(scene, prop)}")

    print("\n  Final classification after optimisation:")
    _, final_top5 = render_and_classify(scene, classifier)
    print_top5(final_top5)

    from PIL import Image as PILImage

    img_np, _ = scene.render()
    save_path = f"adversarial_{target_name.replace(' ', '_')}.png"
    PILImage.fromarray((img_np * 255).clip(0, 255).astype(np.uint8)).save(save_path)
    print(f"  Adversarial image saved → {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        run_cli()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)