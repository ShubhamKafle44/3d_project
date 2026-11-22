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
assets/3D_MODELS/Car/ # default Car model
assets/3D_ENVIRONMENTS/studio/ # default studio environment
```

## Setup

Create a separate Conda environment for the renderer you plan to use. Both
environments include the detector and search dependencies; only the selected
renderer is installed. Commands below assume Linux and run from the repository
root. Use the matching environment every time you launch the pipeline.

### Mitsuba renderer environment

Mitsuba supports CPU differentiation with `llvm_ad_rgb`. CUDA differentiation
requires a compatible NVIDIA driver and a Mitsuba/Dr.Jit CUDA build.

```bash
conda create -n pipeline-mitsuba python=3.10 -y
conda activate pipeline-mitsuba
python -m pip install --upgrade pip
python -m pip install -r pipeline/requirements-mitsuba.txt
python -c "import mitsuba, drjit, torch, torchvision, ultralytics; print('Mitsuba environment ready')"
```

### PyTorch3D renderer environment

PyTorch3D must be built against the same Python, PyTorch, and CUDA versions in
this environment. The project includes PyTorch3D source in `pytorch3d/`.
Install a C++ compiler first; for the CUDA build, also install CUDA Toolkit
11.8 (including `nvcc`). On Debian/Ubuntu:

```bash
sudo apt install build-essential
```

Then, from the repository root:

```bash
conda create -n pipeline-pytorch3d python=3.10 -y
conda activate pipeline-pytorch3d
conda install pytorch=2.1.2 torchvision=0.16.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install cuda-toolkit=11.8 -c nvidia -y
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r pipeline/requirements-pytorch3d.txt
python -m pip install --no-build-isolation -e ./pytorch3d
python -c "import torch, torchvision, pytorch3d, ultralytics; print('PyTorch3D environment ready:', torch.__version__, torchvision.__version__, pytorch3d.__version__)"
```

The PyTorch3D build can take several minutes. For CPU-only use, replace the two
CUDA Conda installs above with:

```bash
conda install pytorch=2.1.2 torchvision=0.16.2 cpuonly -c pytorch -y
```

PyTorch3D's source build still needs a C++ compiler. Keep its Python, PyTorch,
and CUDA versions matched; rebuild PyTorch3D if you change that stack.

### Run

Activate the environment matching the `renderer` setting near the top of
`pipeline/main.py`, then run:

```bash
cd pipeline
python main.py
```

The default renderer is Mitsuba. Set `renderer = "pytorch3d"` to use the
PyTorch3D environment. YOLO model weights download the first time a detector is
loaded; YOLOE also downloads its text encoder.

The square render resolution is selected from `DETECTOR_INPUT_SIZES` in
`config.py` using the configured detector name. YOLO models currently use
640×640 inputs; Faster R-CNN uses 800×800 renders, matching its standard
minimum-side resize setting.

The checked-in `main.py` selects its renderer and search property in code. The detector can be set in `main.py` or with `--detector` as described below.

## Choose the detector

The detector choices are listed in `config.py`: Faster R-CNN, all five YOLO26
sizes (`n`, `s`, `m`, `l`, `x`), and YOLOE-26s for text-prompted open-vocabulary
detection. YOLO26 is the latest released Ultralytics detector family; YOLO27
is still marked as coming soon in the official docs. [YOLO26](https://docs.ultralytics.com/models/yolo26), [YOLO27 status](https://docs.ultralytics.com/models/yolo27)

Select a detector and search property by changing their defaults near the top
of `main.py`, or pass them on the command line:

```bash
python main.py --detector yolo26n
python main.py --detector yolo26l --epochs 100
python main.py --detector yoloe-26s-seg
python main.py --detector yolo26m --property POSE --epochs 100 --learning-rate 0.01
```

For YOLO models, install or upgrade Ultralytics with
`python -m pip install -U ultralytics`; weights download the first time they
are loaded. YOLOE-26 requires Ultralytics 8.4.0 or newer and adds its text
encoder download on first use. The YOLOE option uses the configured target
class as its text prompt. YOLO search uses a differentiable target-class score
from the raw detection head, while validation uses the normal postprocessed
detector outputs. The selected detector name is included in the run folder.

## Run

From the repository root, activate the environment and launch the pipeline:

```bash
conda activate pipeline
cd pipeline
python main.py
```

The `renderer`, `property`, and `detector_target` variables near the top of `main.py` select the renderer, differentiable target, and COCO class. Both renderers support `POSITION`, `LIGHTING`, `ROTATION`, `TEXTURE`, and `POSE` (position plus rotation). Set `detector_target` to a class such as `person`, `car`, or `dog`. The search uses a differentiable target-class surrogate and periodically checks that class's actual detector score.

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

The checked-in `main.py` defaults to the `Car` model and no environment.
Selected models are automatically framed from their OBJ bounds. Each run creates
a timestamped folder labeled with its epoch and learning-rate settings under
`pipeline/outputs/` containing the initial render,
an image snapshot every 10 gradient steps, and the best result. Runs therefore
do not overwrite images from earlier runs.
The run's `loss_curve.png` is refreshed after every optimizer iteration so it
can be opened during the run to follow loss over time.

To launch the detector/property/epoch/learning-rate sweep from inside
`pipeline/` and keep it running after the terminal closes:

```bash
bash run_background.sh
```

Edit the `DETECTORS`, `PROPERTIES`, `EPOCHS`, and `LEARNING_RATES` arrays at
the top of `run_background.sh` to choose the combinations. It runs all epoch
and rate combinations for each property of each detector, in that order. Each
combination gets its own log, and `summary.csv` records its exit status under a
timestamped folder in `pipeline/outputs/sweeps/`. A failed run is recorded and
the sweep continues. The search may finish early if it reaches the configured
success threshold.

To run a 10-by-10 grid of epoch and learning-rate settings sequentially:

```bash
bash run_sweep.sh
```

The sweep script starts in the background, so it keeps running after the
terminal closes. Its `EPOCHS` and `LEARNING_RATES` arrays define the 100
combinations. Each combination gets a separate output folder containing a
`metrics.csv` and a `snapshots/` directory. The CSV records step, parameter,
target-class probability, highest detected class and its score, and snapshot
filename. The optimization workbook records the same detection fields for
each validation step. Snapshots include the initial render and an image every
10 gradient steps. Sweep logs and an overall
`summary.csv` are saved under `outputs/sweeps/`. Each run is a separate Python
process, releasing its CUDA and Dr.Jit allocations before the next run starts.
The sweep continues after a failed run and records its exit status.

## Output

- `outputs/<timestamp>_<property>_e<epochs>_lr<rate>_<model>_<environment>/<model>_<environment>_initial_render.png` — the starting render
- `outputs/<timestamp>_<property>_e<epochs>_lr<rate>_<model>_<environment>/snapshots/step_<step>.png` — renders saved every 10 gradient steps
- `outputs/<timestamp>_<property>_e<epochs>_lr<rate>_<model>_<environment>/loss_curve.png` — loss versus iteration, refreshed after each optimizer step
- `outputs/<timestamp>_<property>_e<epochs>_lr<rate>_<model>_<environment>/adversarial_result_<model>_<environment>_<renderer>_<property>.png` — the render with the lowest target-class confidence found
- Console log shows the selected target-class probability, highest-scoring
  detected class and its score at every validation step, plus the final result

## Download a run from another server

Copy `.env.example` to `.env` in the repository root and set the SSH server
address, username, and password there. `.env` is ignored by Git, so do not
commit it. Install the added downloader dependencies with
`python -m pip install -r pipeline/requirements.txt`.

Download a complete remote output folder, including nested directories and
hidden files:

```bash
python pipeline/download_remote.py \
  --remote-folder '~/3d_project/pipeline/outputs/your_run_folder' \
  --epochs 1000 \
  --learning-rate 0.01
```

The folder is saved under `downloads/` in the repository root by default.
`--epochs` and `--learning-rate` add labels to the local folder name; they do
not filter remote files. Set `--local-dir` to choose another destination.
The SSH server key must already be trusted in `~/.ssh/known_hosts`; connect
once with `ssh username@host` and verify the key before running the downloader.

## Notes

- Gradient search minimizes a differentiable surrogate for the selected COCO
  class, then periodically checks the detector's actual score for that class.
  Mitsuba optimizes one shared 256×256 RGB image across all diffuse materials
  used by the selected OBJ part, preserving their UV mappings. The image is
  initialized from resized source maps and saved in the run output folder.
  Set `texture_attack_target` in `main.py` to a loaded part name such as
  `Jeep`; for folder-based models, part names come from OBJ filenames.
- The `Car` model is in `assets/3D_MODELS/Car/`; replace or add assets in the
  model and environment libraries to try other scenes.
- The Mitsuba renderer uses `llvm_ad_rgb` by default so gradient mode works
  on CPU. Set `MITSUBA_VARIANT=cuda_ad_rgb` before launch to use Mitsuba AD
  on a compatible NVIDIA GPU.
