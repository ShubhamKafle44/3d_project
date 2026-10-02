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
search.py               # black-box random search over scene properties
main.py                 # CLI entrypoint
renderer/
  base.py                # DifferentiableScene interface both backends implement
  pytorch3d_renderer.py   # PyTorch3D backend
  mitsuba_renderer.py     # Mitsuba 3 / Dr.Jit backend
assets/human/body.obj   # PLACEHOLDER cube - replace with your real mesh
```

## Setup

These commands assume Linux. PyTorch3D must be compiled against the same PyTorch, Python, and CUDA environment used by the pipeline. This repository includes its source in the sibling `pytorch3d/` directory. Install a C++ compiler and, for a CUDA build, the CUDA 11.8 toolkit (`nvcc`) before building. On Debian/Ubuntu, install the compiler with `sudo apt install build-essential`.

From the repository root:

```bash
conda create -n pipeline python=3.10 -y
conda activate pipeline
conda install pytorch=2.1.2 torchvision=0.16.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install cuda-toolkit=11.8 -c nvidia -y
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r pipeline/requirements.txt
python3 -m pip install --no-build-isolation -e ./pytorch3d
```

The PyTorch3D build can take several minutes. If you do not have an NVIDIA GPU, install the CPU builds instead of `pytorch-cuda`:

```bash
conda install pytorch=2.1.2 torchvision=0.16.2 cpuonly -c pytorch -y
```

Then install the Python dependencies and build PyTorch3D with the same pip commands above. The CPU renderer is slower. Do not use a PyTorch3D wheel built for another Python, PyTorch, or CUDA version. If you replace PyTorch or CUDA later, rebuild PyTorch3D.

To verify the environment and start the pipeline from the repository root:

```bash
python -c "import torch, torchvision, pytorch3d, mitsuba; print(torch.__version__, torchvision.__version__, pytorch3d.__version__)"
cd pipeline
python main.py
```

The checked-in `main.py` selects its renderer, detector, and search property in code. Edit those settings in `main.py` before launching to change them.

## Run

From the repository root, activate the environment and launch the pipeline:

```bash
conda activate pipeline
cd pipeline
python main.py
```

`main.py` currently selects the renderer, detector, and property through variables near the top of the file. Change those values there before launching. For the PyTorch3D `gradient_appearance` mode, select the Faster R-CNN model; RetinaNet is supported for random search.

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
