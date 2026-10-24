"""Read-only YOLO label audit; write summaries and a chart in results/."""

import csv
import math
from collections import Counter
from pathlib import Path
from project_paths import DATASET as DATASET_ROOT

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

DATASET = DATASET_ROOT
OUTPUT = Path(__file__).parent / "results" / "dataset_analysis"
IMAGE_EXTENSIONS = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm", ".heic"}


def save_csv(filename, rows, fields):
    """Save records as a CSV file, including headers for empty reports.

    Args:
        filename: Output filename.
        rows: List of dictionaries to save.
        fields: Ordered column names.
    """
    with (OUTPUT / filename).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit_split(split, names):
    """Count files and validate every nonempty YOLO annotation line.

    Args:
        split: Dataset directory name.
        names: Class names ordered by class ID.

    Returns:
        Summary dictionary, per-class counts, and issue records.
    """
    root = DATASET / split
    for directory in (root / "images", root / "labels"):
        if not directory.is_dir():
            raise FileNotFoundError(f"Required dataset directory is missing: {directory}")
    images = sorted(p for p in (root / "images").rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    labels = sorted((root / "labels").rglob("*.txt"))
    image_keys = Counter(str(p.relative_to(root / "images").with_suffix("")) for p in images)
    label_keys = Counter(str(p.relative_to(root / "labels").with_suffix("")) for p in labels)
    issues = []
    counts = Counter()
    summary = dict(split=split, images=len(images), label_files=len(labels), total_boxes=0,
                   valid_boxes=0, empty_labels=0, missing_labels=0, missing_images=0,
                   invalid_class_ids=0, invalid_coordinates=0, malformed_rows=0, unreadable_labels=0)

    def issue(path, line, kind, detail):
        """Append one issue with its file location.

        Args:
            path: Affected file path or relative pairing key.
            line: One-based label line, or zero for a file issue.
            kind: Issue category.
            detail: Human-readable explanation.
        """
        issues.append(dict(split=split, file=str(path), line=line, issue=kind, detail=detail))

    for key in sorted(image_keys.keys() - label_keys.keys()):
        summary["missing_labels"] += image_keys[key]
        issue(key, 0, "missing_label", "Image has no matching .txt label")
    for key in sorted(label_keys.keys() - image_keys.keys()):
        summary["missing_images"] += label_keys[key]
        issue(key, 0, "missing_image", "Label has no matching image")
    for key, count in image_keys.items():
        if count > 1:
            issue(key, 0, "ambiguous_image_stem", f"{count} images share a label key")

    for label in labels:
        try:
            content = label.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as error:
            summary["unreadable_labels"] += 1
            issue(label, 0, "unreadable_label", str(error))
            continue
        if not content.strip():
            summary["empty_labels"] += 1
        for number, line in enumerate(content.splitlines(), 1):
            if not line.strip():
                continue
            # Each nonempty line is one declared box, even if invalid.
            summary["total_boxes"] += 1
            parts = line.split()
            if len(parts) != 5:
                summary["malformed_rows"] += 1
                issue(label, number, "malformed_row", f"Expected 5 fields, found {len(parts)}: {line}")
                continue
            try:
                class_value = float(parts[0])
                class_ok = math.isfinite(class_value) and class_value.is_integer() and 0 <= class_value < len(names)
            except ValueError:
                class_ok = False
            if class_ok:
                counts[int(class_value)] += 1
            else:
                summary["invalid_class_ids"] += 1
                issue(label, number, "invalid_class_id", parts[0])
            try:
                x, y, width, height = map(float, parts[1:])
                coords_ok = (all(math.isfinite(v) for v in (x, y, width, height))
                             and 0 <= x <= 1 and 0 <= y <= 1
                             and 0 < width <= 1 and 0 < height <= 1)
            except ValueError:
                coords_ok = False
            if not coords_ok:
                summary["invalid_coordinates"] += 1
                issue(label, number, "invalid_coordinates", "Require finite normalized x,y in [0,1] and width,height in (0,1]: " + line)
            elif x - width / 2 < -1e-6 or y - height / 2 < -1e-6 or x + width / 2 > 1 + 1e-6 or y + height / 2 > 1 + 1e-6:
                issue(label, number, "box_outside_image", "Normalized values pass, but box edges extend outside image")
            if class_ok and coords_ok:
                summary["valid_boxes"] += 1
    return summary, counts, issues


def main():
    """Audit all splits and save numerical summaries and a bar chart."""
    with (DATASET / "data.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    raw_names = config["names"]
    names = raw_names if isinstance(raw_names, list) else [raw_names[i] for i in range(len(raw_names))]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries, distribution, issues, zero_classes = [], [], [], []
    for split in ("train", "valid", "test"):
        summary, counts, split_issues = audit_split(split, names)
        summaries.append(summary)
        issues.extend(split_issues)
        for class_id, name in enumerate(names):
            distribution.append(dict(split=split, class_id=class_id, class_name=name, boxes=counts[class_id]))
            if split in ("valid", "test") and counts[class_id] == 0:
                zero_classes.append(dict(split=split, class_id=class_id, class_name=name))
        print(summary)
        print("  Class boxes:", {name: counts[i] for i, name in enumerate(names)})
    save_csv("split_summary.csv", summaries, list(summaries[0]))
    save_csv("class_distribution.csv", distribution, ["split", "class_id", "class_name", "boxes"])
    save_csv("issues.csv", issues, ["split", "file", "line", "issue", "detail"])
    save_csv("zero_example_classes.csv", zero_classes, ["split", "class_id", "class_name"])
    fig, ax = plt.subplots(figsize=(11, 6))
    for offset, split in enumerate(("train", "valid", "test")):
        values = [row["boxes"] for row in distribution if row["split"] == split]
        bars = ax.bar([i + (offset - 1) * 0.25 for i in range(len(names))], values, width=0.25, label=split)
        ax.bar_label(bars, fontsize=8, padding=2)
    ax.set_xticks(range(len(names)), names, rotation=15)
    ax.set_ylabel("Bounding boxes")
    ax.set_title("YOLO dataset class distribution")
    ax.legend()
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(OUTPUT / "class_distribution.png", dpi=180)
    plt.close(fig)
    print("Zero-example validation/test classes:", zero_classes or "None")
    print("Issue counts:", dict(Counter(row["issue"] for row in issues)))
    print("Outputs:", OUTPUT)
    print("Counts include valid class IDs even when coordinates are invalid; total_boxes counts nonempty annotation lines.")


if __name__ == "__main__":
    main()
