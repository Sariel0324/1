"""Small, testable helpers for the final evaluation. No model training occurs."""

import csv
import math
from pathlib import Path

import numpy as np
import yaml


def parse_labels(path, class_count=5):
    """Read normalized YOLO detection labels, rejecting invalid annotations.

    Args:
        path: Label text file. Empty files represent background images.
        class_count: Number of supported classes.

    Returns:
        List of (class_id, x_center, y_center, width, height) tuples.

    Raises:
        ValueError: A line has malformed fields, invalid class, or coordinates.
        FileNotFoundError: The label file is absent.
    """
    boxes = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = list(map(float, line.split()))
            if len(values) != 5 or not all(math.isfinite(v) for v in values):
                raise ValueError("Expected five finite numbers")
            class_id, x, y, width, height = values
            if not class_id.is_integer() or not 0 <= class_id < class_count:
                raise ValueError("Invalid class ID")
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                raise ValueError("Invalid normalized coordinates")
        except ValueError as error:
            raise ValueError(f"{path}, line {number}: {error}") from error
        boxes.append((int(class_id), x, y, width, height))
    return boxes


def load_names(path):
    """Read ordered class names and check the declared class count.

    Args:
        path: Dataset YAML path.

    Returns:
        List of class names ordered by integer class ID.

    Raises:
        ValueError: Names or the declared class count are inconsistent.
    """
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    names = config["names"]
    if isinstance(names, dict):
        if set(names) != set(range(len(names))):
            raise ValueError("Class IDs must be consecutive, starting at zero")
        names = [names[i] for i in range(len(names))]
    if not isinstance(names, list) or not names or config.get("nc", len(names)) != len(names):
        raise ValueError("Class names do not match nc")
    return names


def evaluation_settings(imgsz):
    """Provide identical evaluation settings except for image resolution.

    Args:
        imgsz: Square model input side length.

    Returns:
        Dictionary of Ultralytics validation arguments for the test split.
    """
    if imgsz <= 0 or imgsz % 32:
        raise ValueError("Image size must be a positive multiple of 32")
    return dict(imgsz=imgsz, batch=4, device=0, split="test", conf=0.001,
                iou=0.7, max_det=300, half=False, rect=False, augment=False,
                workers=0, plots=True, save_json=True, seed=42)


def latency_statistics(milliseconds):
    """Summarize measured batch-one forward latencies.

    Args:
        milliseconds: Nonempty sequence of positive latency measurements.

    Returns:
        Dictionary with mean/median milliseconds and forward-only FPS.
    """
    values = np.asarray(milliseconds, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Latencies must be finite and positive")
    return dict(mean_ms=float(values.mean()), median_ms=float(np.median(values)),
                fps=float(1000 / values.mean()))


def save_csv(path, rows, fields):
    """Save dictionaries with explicit columns, including empty reports.

    Args:
        path: Destination filename.
        rows: Iterable of row dictionaries. None values become empty cells.
        fields: Ordered CSV column names.
    """
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
