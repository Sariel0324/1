"""Train one controlled YOLOv8n experiment, without using the test split."""

import argparse
import csv
import json
import os
import time
import traceback
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT / ".ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".matplotlib"))

import torch
import ultralytics
from ultralytics import YOLO


def parse_arguments():
    """Read experiment settings from the command line.

    Returns:
        argparse.Namespace: Settings for this training run.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "training_data.yaml")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="experiment_a_320")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "runs")
    return parser.parse_args()


def main():
    """Train from pretrained weights and preserve timings, metrics, and errors."""
    args = parse_arguments()
    destination = args.output_dir.resolve() / args.name
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {destination}")
    destination.mkdir(parents=True)
    summary = {"status": "started", "requested": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "torch": torch.__version__, "ultralytics": ultralytics.__version__}
    started = time.perf_counter()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; this experiment requires GPU device 0.")
        summary["gpu"] = torch.cuda.get_device_name(0)
        # A new YOLO object always starts from the same original pretrained file.
        model = YOLO(str(PROJECT / "yolov8n.pt"))
        metrics = model.train(
            data=str(args.data.resolve()), imgsz=args.imgsz, epochs=args.epochs,
            batch=args.batch, seed=args.seed, device=0,
            project=str(args.output_dir.resolve()), name=args.name, exist_ok=True,
            pretrained=True, resume=False, val=True, split="val", save=True,
            plots=True, cache=False, workers=0, deterministic=True, patience=0,
            optimizer="AdamW", lr0=0.001, weight_decay=0.0005,
            amp=True, close_mosaic=10,
        )
        # Ultralytics returns validation metrics for best.pt after training.
        summary["best_validation_metrics"] = {k: float(v) for k, v in metrics.results_dict.items()}
        summary["actual_batch"] = model.trainer.args.batch
        summary["oom_retries_last_epoch"] = model.trainer._oom_retries
        with (destination / "results.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        summary["epochs_completed"] = len(rows)
        summary["stopped_early"] = len(rows) < args.epochs
        summary["status"] = "success"
    except Exception:
        summary["status"] = "failed"
        (destination / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        summary["elapsed_seconds"] = time.perf_counter() - started
        (destination / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
