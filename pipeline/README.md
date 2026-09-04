# Differentiable-Rendering Adversarial Search

Finds camera/lighting/pose parameters where a COCO object detector
(Faster R-CNN or RetinaNet) fails to recognize a rendered 3D human mesh.
Two interchangeable rendering backends — PyTorch3D and Mitsuba/Dr.Jit —
sit behind one shared interface, so you can attack the same scene through
either renderer with the same CLI.

## Layout

```
config.py              # <- point this at your real .obj files
detector.py             # COCO detector wrapper -> human_prob score
scene_setup.py          # builds a scene from config.py, auto-frames subject
search.py               # black-box random search over one scene property
main.py                 # CLI entrypoint
renderer/
  base.py                # DifferentiableScene interface both backends implement
  pytorch3d_renderer.py   # PyTorch3D backend
  mitsuba_renderer.py     # Mitsuba 3 / Dr.Jit backend
assets/human/body.obj   # PLACEHOLDER cube - replace with your real mesh
```

## Setup

```bash
pip install -r requirements.txt
# PyTorch3D needs a separate, environment-matched install - see requirements.txt
```

1. Put your human `.obj` (and any part meshes: shirt/pants/etc., or scene
   background) somewhere under `assets/`.
2. Edit `config.py`:
   - `HUMAN_PARTS` -> your mesh path(s)
   - `BACKGROUND_PATH` -> optional scene geometry
   - `CAMERA` / `LIGHT` -> starting values (auto-zoom will adjust distance)

## Run

```bash
# Mitsuba backend, perturb position (default)
python main.py --renderer mitsuba --property POSITION

# PyTorch3D backend, perturb lighting, RetinaNet detector
python main.py --renderer pytorch3d --property LIGHTING --model retinanet_resnet50_fpn_v2

# More search budget, larger step size
python main.py --renderer mitsuba --property ROTATION --epochs 300 --step-size 0.2
```

You only ever specify **what to target** (`--property`) and **which
renderer/detector to test** — mesh loading, camera setup, and the search
loop are all handled internally from `config.py`.

`--property` options:
- `POSITION` — translate the human in x/y/z
- `ROTATION` — yaw/pitch/roll the human
- `LIGHTING` — scalar light intensity
- `CLOTHING` — shirt RGB color (requires a `shirt` part in `HUMAN_PARTS`)

## Output

- `outputs/initial_render.png` — the starting render
- `outputs/adversarial_result_<renderer>_<property>.png` — the render with
  the lowest human-detection confidence found
- Console log shows human_prob at every step and the final result

## Notes

- The search is **black-box** (random search on the rendered image →
  detector score), so it works identically for both renderer backends
  regardless of whether their AD graph is exposed. Both renderers *are*
  differentiable internally (that's the point of using PyTorch3D /
  Mitsuba), so if you want a **gradient-based** attack instead, see the
  notes at the bottom of `pytorch3d_renderer.py` and
  `mitsuba_renderer.py` for how to expose per-parameter gradients through
  each backend's own AD system (`torch.autograd` / Dr.Jit AD).
- `assets/human/body.obj` in this repo is a placeholder cube used only to
  verify the pipeline runs end-to-end — swap in your real human mesh.
- The Mitsuba backend defaults to CPU (`scalar_rgb` variant); switch to
  `cuda_ad_rgb` (set `MITSUBA_VARIANT` env var) if you have a GPU and want
  it to run faster or want AD.
