# Differentiable-Rendering Adversarial Search

Finds camera/lighting/pose parameters where a COCO object detector fails to
recognize the selected class in a rendered 3D model.
Two interchangeable rendering backends — PyTorch3D and Mitsuba/Dr.Jit —
sit behind one shared interface, so you can attack the same scene through
either renderer with the same CLI.

## Layout

```
config.py              # <- point this at your real .obj files
detector.py             # COCO detector wrapper -> selected target-class score
scene_setup.py          # builds a scene from config.py, auto-frames subject
search.py               # differentiable search over scene properties
main.py                 # CLI entrypoint
renderer/
  base.py                # DifferentiableScene interface both backends implement
  pytorch3d_renderer.py   # PyTorch3D backend
  mitsuba_renderer.py     # Mitsuba 3 / Dr.Jit backend
assets/3D_MODELS/human/ # default multipart human model
assets/3D_ENVIRONMENTS/studio/ # default studio environment
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

The `renderer`, `property`, and `detector_target` variables near the top of `main.py` select the renderer, differentiable target, and COCO class. Both renderers support `POSITION`, `LIGHTING`, `ROTATION`, `CLOTHING`, and `POSE` (position plus rotation). Set `detector_target` to a class such as `person`, `car`, or `dog`. The search uses a differentiable target-class surrogate and periodically checks that class's actual detector score.

Mitsuba uses `llvm_ad_rgb` by default for CPU differentiation. On a compatible NVIDIA setup, set `MITSUBA_VARIANT=cuda_ad_rgb` before starting the process to use the CUDA AD variant.

### Choose a 3D model and environment

Put each model in `assets/3D_MODELS/` and each environment in
`assets/3D_ENVIRONMENTS/`. OBJ files can be passed directly. A model folder may
contain one OBJ or several OBJ parts that should be loaded together; an
environment folder should contain one OBJ (or a file named `environment.obj`).
Keep each OBJ's MTL and texture files alongside it so its relative texture
references continue to work. The pipeline currently accepts OBJ assets.

List top-level asset names:

```bash
python main.py --list-assets
```

Select assets by folder or filename, and use `none` for a plain background:

```bash
python main.py --model "delivery van" --environment "warehouse" --epochs 100
python main.py --model "delivery van/van.obj" --environment none --epochs 100
```

The built-in `human` model and `studio` environment remain the defaults.
Custom models are automatically framed from their OBJ bounds. Each run saves
its initial render and result with the selected model/environment names in the
filename, so experimenting with another pair does not overwrite those images.

To launch a 10,000-epoch search from inside `pipeline/` and keep it running after the terminal closes:

```bash
bash run_background.sh
```

The script writes a timestamped log and PID file under `pipeline/outputs/`, then prints the commands to follow the log or stop the process. Pass a different positive epoch count as the first argument, followed by any `main.py` options. For example: `bash run_background.sh 100 --model "delivery van" --environment warehouse`. `--epochs` also works when running `python main.py` directly. The search may finish earlier if it reaches the configured success threshold.

## Output

- `outputs/<model>_<environment>_initial_render.png` — the starting render
- `outputs/adversarial_result_<model>_<environment>_<renderer>_<property>.png` — the render with
  the lowest target-class confidence found
- Console log shows the selected target-class probability at every step and the final result

## Notes

- Gradient search minimizes a differentiable surrogate for the selected COCO
  class, then periodically checks the detector's actual score for that class.
  Mitsuba represents clothing as
  a uniform diffuse shirt color during optimization.
- The default multipart model is in `assets/3D_MODELS/human/`; replace or add
  assets in the model and environment libraries to try other scenes.
- The Mitsuba renderer uses `llvm_ad_rgb` by default so gradient mode works
  on CPU. Set `MITSUBA_VARIANT=cuda_ad_rgb` before launch to use Mitsuba AD
  on a compatible NVIDIA GPU.
