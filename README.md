# MMA3001: pork packaging defect detection

This project investigates whether a lightweight YOLOv8n object detector can locate
pork-packaging defects in images, comparing 320 and 640 pixel inputs under the
same 15-epoch training configuration. It is a student experiment, not a validated
industrial inspection system. Classes are loose-meat, packaging-error,
twisted-meat, unsealed and wrinkle.

## Data supplied separately

Source: [Roboflow version 4](https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging/dataset/4),
workspace `hello-2aqe0`, project `pork-rasher-error-packaging`. Export YOLO detection
format. Its metadata declares CC BY 4.0; see LICENSING.md for attribution and rights.

Default layout (the checkout folder can have any name):

```text
workspace/
  dataset/
    data.yaml
    train/images/   train/labels/
    valid/images/   valid/labels/
    test/images/    test/labels/
  project/         # this checkout
```

Alternatively set `MMA3001_DATASET` to the dataset directory. In PowerShell:
`$env:MMA3001_DATASET = 'D:\my-data\pork'`; on Linux/macOS:
`export MMA3001_DATASET=/path/to/pork`.
The default is resolved relative to the scripts, not the current working directory.
Run the commands below from the checkout root.

## Python setup

The experiments used Python 3.12.4, torch 2.4.1+cu121, torchvision 0.19.1+cu121,
Ultralytics 8.4.184 and an RTX 3060 Laptop GPU (6 GB). The GPU scripts require
CUDA device 0; CPU-only machines can run the audit, tests and documentation.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

On Linux use `source .venv/bin/activate`. Choose an appropriate build from
[PyTorch](https://pytorch.org/get-started/locally/) if the recorded CUDA build does
not support your platform. Do not replace a working CUDA installation unnecessarily.
The requirements pin direct tools; they are not a complete cross-platform lockfile.

## Audit and training reproduction

```powershell
python -B dataset_audit.py
python -B prepare_training_data.py
python -B train_model.py --data training_data.yaml --imgsz 320 --epochs 1 --batch 4 --seed 42 --name smoke_test
python -B train_model.py --data training_data.yaml --imgsz 320 --epochs 15 --batch 4 --seed 42 --name experiment_a_320
python -B train_model.py --data training_data.yaml --imgsz 640 --epochs 15 --batch 4 --seed 42 --name experiment_b_640
```

These are reproduction instructions, not a requirement to retrain completed work.
Preparation copies the dataset to `training_dataset/`, records source hashes and
generates a machine-local `training_data.yaml`. It must be run only when that copy
does not exist. Training loads pretrained `yolov8n.pt` independently for each run
(downloaded if absent), uses GPU 0, AdamW and the same seed and augmentations.
Run experiments sequentially. `--output-dir` controls the output root. The script
refuses an existing run directory. The repository includes compact evidence in
`runs/`; for new training preserve those folders elsewhere or use another output
directory. Full settings are recorded in each run's `args.yaml`.

## Saved-model test evaluation and benchmark

Supply authorised checkpoints separately at:

```text
runs/experiment_a_320/weights/best.pt
runs/experiment_b_640/weights/best.pt
```

```powershell
python -B evaluate_models.py
```

This single command evaluates both checkpoints on the same 80-image test split,
benchmarks inference and generates examples. No training is performed. It refuses
to overwrite `results/test_evaluation`; preserve the supplied evidence folder under
another name before a deliberate new evaluation. It creates a private test-data
copy so loader caches cannot modify the source. Test results remain distinct from
validation results under `runs/`.

Benchmark: fused FP32, GPU 0, batch 1, 20 warm-up forwards and three passes over
the same sorted 80 test images. CUDA synchronization surrounds each timed forward.
Timing includes Python dispatch/synchronization and excludes disk access,
preprocessing, transfers and NMS. FPS is forward-only, not end-to-end throughput.
Single-session laptop timings are affected by clocks, thermals and host overhead.
Peak memory is PyTorch allocated memory, not total driver/application memory.

Wrinkle has no validation or test ground-truth boxes; its performance is
unevaluable, with null/blank per-class scores. The tiny minority-class samples,
class imbalance and single-seed comparison limit conclusions. Do not optimise
models using the test results. Inspect `results/test_evaluation/README.md` for
metric conventions, example selection and recorded warnings.

## Tests and API documentation

```powershell
python -B -m pytest tests -v --basetemp=test_temp_submission --junitxml=pytest_submission.xml
python -B -m pdoc --docformat google -o docs project_paths.py dataset_audit.py train_model.py prepare_training_data.py verify_experiments.py evaluation_utils.py evaluate_models.py
```

Tests use synthetic fixtures and do not need a dataset, weights, CUDA or training.
Use a dedicated temporary folder: pytest may clear its chosen basetemp on reruns.
Open `docs/index.html` in a browser for the generated Google-docstring API reference.

## Evidence and licensing

Keep `results/dataset_analysis/*.csv`, `runs/experiment_comparison.csv`, each run's
`results.csv`, `results.png`, `args.yaml` and `run_summary.json`, test metrics and
benchmark CSV/JSON, pytest text report, PR/confusion figures and `docs/`.
Historical evidence can contain original absolute paths; it is preserved verbatim
and is not used as portable runtime configuration. Some generated `val_batch*`
names refer to test images because of the library's naming convention.

`.gitignore` excludes datasets/copies, environments, caches, weights, logs and
image-bearing overlays. They remain locally available. Dataset-image redistribution
requires a separate rights decision; selected numerical charts can be included.
Source is AGPL-3.0; see LICENSE and LICENSING.md. This does not grant dataset or
third-party weight redistribution rights.

## GitHub and a genuine Moodle clone

No remote upload is implied by these instructions. Create an empty GitHub repository
without an automatically generated README/licence. From this project root:

```powershell
git init
git branch -M main
git config user.name "YOUR NAME"
git config user.email "YOUR EMAIL OR GITHUB NOREPLY ADDRESS"
git add .
git status --short
git diff --cached --stat
git commit -m "Prepare reproducible MMA3001 experiment submission"
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

Review the staged filenames before committing: no dataset or `.pt` files should
appear. HTTPS push normally opens Git Credential Manager authentication; sign in
with your own account. Never put a password/token in source files or a remote URL.
If Git requests a token, use an appropriately scoped GitHub token as the credential,
not your account password. Verify the files and commit on GitHub after push.

For Moodle, clone the actual repository into a NEW sibling directory and zip that
clone including `.git` (a GitHub Download ZIP is not a Git clone):

```powershell
cd ..
git clone https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git MMA3001-moodle
git -C MMA3001-moodle status --short
git -C MMA3001-moodle log -1 --oneline
python -c "import shutil; shutil.make_archive('MMA3001-moodle', 'zip', '.', 'MMA3001-moodle')"
python -c "import zipfile; z=zipfile.ZipFile('MMA3001-moodle.zip'); assert any(n.endswith('/.git/HEAD') for n in z.namelist()); print('Git metadata included')"
```

If offline after making a local commit, use `git clone --no-local ./project
MMA3001-moodle` from its parent instead. The clone includes committed files only;
uncommitted changes and ignored datasets/weights are not included. Check Moodle's
size and checkpoint requirements and supply authorised extra files separately if
required. Do not upload account credentials or confidential Git remote URLs.
