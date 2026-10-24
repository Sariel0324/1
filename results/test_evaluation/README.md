# Final evaluation: how to use these files

This folder contains TEST results only. Earlier validation metrics remain under
`runs/experiment_a_320` and `runs/experiment_b_640`. No models were retrained.

## Scripts and commands

- `evaluate_models.py`: evaluates both existing best checkpoints, benchmarks forwards and saves examples. Run from the project directory with `python -B evaluate_models.py`. It refuses to overwrite this results folder. Preserve this folder before a deliberate new evaluation.
- `evaluation_utils.py`: label parsing, configuration, CSV and timing helpers.
- Tests: `python -B -m pytest tests -v --basetemp=C:\MMA3001\project\test_temp --junitxml=results/test_evaluation/pytest_report.xml`. Use a fresh temporary-directory name if retaining an earlier test workspace.
- HTML docs: `python -B -m pdoc --docformat google -o docs dataset_audit.py train_model.py prepare_training_data.py verify_experiments.py evaluation_utils.py evaluate_models.py`.

## Metrics and figures

- `test_metrics.csv/json`: aggregate test metrics for 80 images and 79 boxes.
- `per_class_test_metrics.csv/json`: class metrics and ground-truth counts. Wrinkle is not evaluable on validation or test: no ground-truth boxes exist. Null JSON values and blank CSV cells mean unavailable, not zero. Ultralytics averages over represented classes. Precision/recall use its F1 operating-point convention; AP integrates the confidence curve. A reported precision of 1 with recall 0 is a library interpolation convention and does not mean successful detection of that class.
- Each model folder contains `test_metrics.json` with exact settings and checkpoint hash, `predictions.json` with detections, PR/F1/confusion figures, and `examples/` with three genuine side-by-side examples: detection, missed ground truth and false positive. Filenames `val_batch*` are Ultralytics' naming convention; they contain TEST images here because the selected split is test.
- `prediction_examples.csv/json` records image-level diagnostics at confidence 0.25 with greedy same-class matching at IoU 0.5. These diagnostics illustrate behavior and do not replace the aggregate AP calculation. Example selection takes the first sorted image satisfying each category, not a random or representative sample.
- `test_image_manifest.json` identifies the exact shared test image set.
- `dataset_integrity.json` confirms original dataset hashes did not change. `evaluation_dataset` is an isolated copy; loader caches are written there.

## Benchmark definition and limitations

`inference_benchmark.csv/json` and each model's `benchmark.json` summarize
`inference_latencies.csv/json`. Each model uses FP32, fused layers, batch 1,
square letterboxed inputs, device cuda:0, 20 untimed warm-up forwards and three
passes over the same 80 sorted images (240 measured forwards). A CPU high-resolution
timer and CUDA synchronization before/after each forward include Python dispatch
and synchronization overhead. Disk access, preprocessing, host-to-GPU transfer,
NMS, visualization and training are excluded. FPS is 1000 divided by mean latency,
not end-to-end video throughput.

Peak allocated memory is the process's PyTorch CUDA tensor allocation peak after
warm-up, including model, input and forward activations. Reserved memory is also
recorded. Neither number includes all driver/context memory or full application
memory; do not compare it directly to total usage shown by nvidia-smi.

These are one-session measurements in sequential model order, not a controlled
clock/power study. CPU dispatch overhead, laptop power/temperature and clock
variation can affect rankings; B's lower measured latency does not establish that
640-pixel inference is generally faster. GPU model, package versions and seed are
in `environment.json`; driver information is in `nvidia_info.csv`.

## Tests, documentation and warnings

`pytest_report.txt` and `pytest_report.xml` contain the successful actual test run.
The initial report is retained in `pytest_initial_temp_error.*`: pytest could not
create its default sandbox temporary directory. Using a project-local temporary
directory resolved this environmental problem.

HTML API documentation starts at `../../docs/index.html`. The documentation build
log is `documentation_build.log`. The full evaluation console log is `console.log`.

Ultralytics warned that `half=False` is deprecated (it still selects FP32).
Optional faster-coco-eval was unavailable and its automatic install was skipped
offline. The saved metrics are Ultralytics' built-in detection metrics; no external
COCO score is claimed. No CUDA memory errors occurred.
