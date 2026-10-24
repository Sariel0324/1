"""Evaluate saved checkpoints on test data, benchmark forwards, and show examples.

Run with `python -B evaluate_models.py`. No training or parameter tuning occurs.
Test metrics are stored separately from training/validation outputs.
"""

import gc
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from project_paths import DATASET as DATASET_ROOT

PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT / "results" / "test_evaluation"
SOURCE = DATASET_ROOT
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT / ".ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".matplotlib"))

import cv2
import numpy as np
import torch
import ultralytics
import yaml
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

from evaluation_utils import evaluation_settings, latency_statistics, load_names, parse_labels, save_csv


def write_json(path, value):
    """Save JSON with strict finite numbers and null for unavailable metrics.

    Args:
        path: Output filename.
        value: JSON-compatible data.
    """
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def source_hashes():
    """Calculate a read-only integrity snapshot.

    Returns:
        Dictionary mapping original dataset paths to SHA-256 hashes.
    """
    return {str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(SOURCE.rglob("*")) if p.is_file()}


def prepare_test_data(names):
    """Copy the existing test split so loader caches cannot alter the original.

    Args:
        names: Ordered class names.

    Returns:
        Paths to an explicit evaluation YAML and the sorted copied test images.
    """
    root = OUTPUT / "evaluation_dataset"
    shutil.copytree(SOURCE / "test", root / "test")
    config = dict(path=root.as_posix(), train="test/images", val="test/images", test="test/images", names=names, nc=len(names))
    # The loader requires train/val keys, but this script calls split='test' only.
    config_path = OUTPUT / "test_data.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    images = sorted(p for p in (root / "test" / "images").rglob("*") if p.suffix.lower() in extensions)
    if len(images) != 80:
        raise ValueError(f"Expected 80 existing test images; found {len(images)}")
    write_json(OUTPUT / "test_image_manifest.json", [dict(file=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in images])
    return config_path, images


def per_class_rows(metrics, names, counts, model_name):
    """Extract supported class metrics without inventing scores for absent classes.

    Args:
        metrics: Ultralytics detection metrics.
        names: Ordered class names.
        counts: Ground-truth box counts by class ID.
        model_name: Experiment label.

    Returns:
        One record per class; unsupported metrics are None, saved as null/blank.
    """
    indices = {int(class_id): i for i, class_id in enumerate(metrics.box.ap_class_index)}
    rows = []
    for class_id, name in enumerate(names):
        row = dict(model=model_name, split="test", class_id=class_id, class_name=name,
                   ground_truth_boxes=counts[class_id], precision=None, recall=None,
                   mAP50=None, mAP50_95=None, status="not_evaluable_no_ground_truth")
        if counts[class_id] > 0 and class_id in indices:
            values = metrics.box.class_result(indices[class_id])
            row.update(zip(("precision", "recall", "mAP50", "mAP50_95"), map(float, values)))
            row["status"] = "evaluated"
        elif counts[class_id] > 0:
            row["status"] = "metrics_unavailable"
        rows.append(row)
    return rows


def benchmark(checkpoint, images, imgsz, model_name):
    """Measure fused FP32 model forwards on preprocessed GPU-resident images.

    Preprocessing, CPU-to-GPU copies, NMS, plotting, and disk reads are excluded.
    CUDA synchronization surrounds each batch-one forward. Twenty warm-up
    passes precede three passes over the identical sorted set of 80 test images.

    Args:
        checkpoint: Saved best.pt path.
        images: Sorted image paths shared by both models.
        imgsz: Square input resolution.
        model_name: Experiment label.

    Returns:
        Summary record and individual latency measurements.
    """
    gc.collect()
    torch.cuda.empty_cache()
    model = YOLO(str(checkpoint)).model.to("cuda:0").float().eval()
    model.fuse()
    transform = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)
    tensors = []
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot decode {path}")
        array = transform(image=image)[:, :, ::-1].transpose(2, 0, 1)
        tensors.append(torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0).float() / 255)
    timings = []
    with torch.inference_mode():
        for i in range(20):
            tensor = tensors[i % len(tensors)].to("cuda:0")
            result = model(tensor)
            del result, tensor
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for repeat in range(3):
            for path, cpu_tensor in zip(images, tensors):
                tensor = cpu_tensor.to("cuda:0")
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = model(tensor)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                timings.append(dict(model=model_name, repeat=repeat + 1, image=path.name, milliseconds=elapsed))
                del result, tensor
    summary = dict(model=model_name, imgsz=imgsz, batch=1, precision="FP32", fused=True,
                   device=torch.cuda.get_device_name(0), warmup_passes=20, repeats=3,
                   images=len(images), measurements=len(timings),
                   peak_allocated_MiB=torch.cuda.max_memory_allocated() / 1024**2,
                   peak_reserved_MiB=torch.cuda.max_memory_reserved() / 1024**2,
                   scope="GPU-resident fused forward only; excludes preprocessing, transfers and NMS",
                   **latency_statistics([row["milliseconds"] for row in timings]))
    del model, tensors
    gc.collect()
    torch.cuda.empty_cache()
    return summary, timings


def match_boxes(ground_truth, predictions, threshold=0.5):
    """Greedily match same-class predictions by IoU for illustrative examples.

    Args:
        ground_truth: Rows containing class ID and pixel xyxy coordinates.
        predictions: Rows containing class ID and pixel xyxy coordinates.
        threshold: Minimum IoU to count an illustrative true positive.

    Returns:
        True-positive, false-negative and false-positive counts. This simple
        diagnostic does not replace Ultralytics' aggregate AP calculation.
    """
    candidates = []
    for i, gt in enumerate(ground_truth):
        for j, pred in enumerate(predictions):
            if gt[0] != pred[0]:
                continue
            left, top = max(gt[1], pred[1]), max(gt[2], pred[2])
            right, bottom = min(gt[3], pred[3]), min(gt[4], pred[4])
            intersection = max(0, right-left) * max(0, bottom-top)
            union = (gt[3]-gt[1]) * (gt[4]-gt[2]) + (pred[3]-pred[1]) * (pred[4]-pred[2]) - intersection
            iou = intersection / union if union > 0 else 0
            if iou >= threshold:
                candidates.append((iou, i, j))
    used_gt, used_pred = set(), set()
    for _, i, j in sorted(candidates, reverse=True):
        if i not in used_gt and j not in used_pred:
            used_gt.add(i)
            used_pred.add(j)
    return len(used_gt), len(ground_truth)-len(used_gt), len(predictions)-len(used_pred)


def visual_examples(checkpoint, images, imgsz, names, model_name):
    """Save genuine detection and failure examples with ground truth alongside.

    Args:
        checkpoint: Saved checkpoint path.
        images: Sorted test image paths.
        imgsz: Input size.
        names: Class names.
        model_name: Experiment label.

    Returns:
        Image-level diagnostic records with fixed confidence 0.25 and IoU 0.5.
    """
    model = YOLO(str(checkpoint))
    folder = OUTPUT / model_name / "examples"
    folder.mkdir(parents=True)
    selected = set()
    outcomes = []
    for path in images:
        result = model.predict(str(path), imgsz=imgsz, device=0, conf=0.25, iou=0.7,
                               max_det=300, half=False, rect=False, verbose=False)[0]
        height, width = result.orig_shape
        labels = parse_labels(path.parent.parent / "labels" / (path.stem + ".txt"), len(names))
        gt = [(c, (x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height) for c,x,y,w,h in labels]
        pred = [(int(c), *box) for c, box in zip(result.boxes.cls.cpu().tolist(), result.boxes.xyxy.cpu().tolist())]
        tp, fn, fp = match_boxes(gt, pred)
        categories = []
        if tp: categories.append("detection")
        if fn: categories.append("missed_ground_truth")
        if fp: categories.append("false_positive")
        category = next((c for c in categories if c not in selected), None)
        outcomes.append(dict(model=model_name, image=path.name, ground_truth=len(gt), predictions=len(pred),
                             matched=tp, missed=fn, false_positives=fp, example_category=category))
        if category:
            selected.add(category)
            truth = result.orig_img.copy()
            for c, x1, y1, x2, y2 in gt:
                cv2.rectangle(truth, (int(x1), int(y1)), (int(x2), int(y2)), (0,200,0), 2)
                cv2.putText(truth, names[c], (max(0,int(x1)), max(15,int(y1))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,180,0), 1)
            panels = []
            for image, title in ((truth, "Ground truth"), (result.plot(), f"Prediction conf>=0.25 | TP={tp} FN={fn} FP={fp}")):
                panel = cv2.copyMakeBorder(image, 35, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))
                cv2.putText(panel, title, (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                panels.append(panel)
            cv2.imwrite(str(folder / f"{category}_{path.stem}.jpg"), np.hstack(panels))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return outcomes


def main():
    """Run fixed test evaluation, benchmark and examples; verify source integrity."""
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation results: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    before = source_hashes()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device 0 is required")
        torch.manual_seed(42)
        np.random.seed(42)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        names = load_names(SOURCE / "data.yaml")
        config, images = prepare_test_data(names)
        counts = [0] * len(names)
        for path in images:
            for box in parse_labels(path.parent.parent / "labels" / (path.stem + ".txt"), len(names)):
                counts[box[0]] += 1
        write_json(OUTPUT / "environment.json", dict(torch=torch.__version__, ultralytics=ultralytics.__version__,
                   cuda=torch.version.cuda, device=torch.cuda.get_device_name(0), seed=42,
                   cudnn_benchmark=False, cudnn_deterministic=True,
                   limitation="Wrinkle has no ground truth in validation or test and cannot be evaluated."))
        summaries, classes, speeds, latencies, examples = [], [], [], [], []
        for name, size in (("model_a_320", 320), ("model_b_640", 640)):
            experiment = "experiment_a_320" if size == 320 else "experiment_b_640"
            checkpoint = PROJECT / "runs" / experiment / "weights" / "best.pt"
            settings = evaluation_settings(size)
            model = YOLO(str(checkpoint))
            metrics = model.val(data=str(config), project=str(OUTPUT), name=name, exist_ok=False, **settings)
            precision, recall, map50, map95 = map(float, metrics.mean_results())
            row = dict(model=name, split="test", images=len(images), ground_truth_boxes=sum(counts),
                       imgsz=size, precision=precision, recall=recall, mAP50=map50, mAP50_95=map95)
            summaries.append(row)
            class_rows = per_class_rows(metrics, names, counts, name)
            classes.extend(class_rows)
            write_json(OUTPUT / name / "test_metrics.json", dict(overall=row, per_class=class_rows,
                       settings=settings, checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
            del model, metrics
            gc.collect()
            speed, measurements = benchmark(checkpoint, images, size, name)
            speeds.append(speed)
            latencies.extend(measurements)
            write_json(OUTPUT / name / "benchmark.json", speed)
            examples.extend(visual_examples(checkpoint, images, size, names, name))
            print(json.dumps(dict(test=row, benchmark=speed), indent=2), flush=True)
        for filename, rows in (("test_metrics", summaries), ("per_class_test_metrics", classes),
                               ("inference_benchmark", speeds), ("inference_latencies", latencies), ("prediction_examples", examples)):
            save_csv(OUTPUT / f"{filename}.csv", rows, list(rows[0]))
            write_json(OUTPUT / f"{filename}.json", rows)
    except Exception:
        import traceback
        (OUTPUT / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        after = source_hashes()
        write_json(OUTPUT / "dataset_integrity.json", dict(unchanged=before == after, files=len(after)))


if __name__ == "__main__":
    main()
