"""Verify experiment artifacts and save a machine-readable comparison."""

import csv
import hashlib
import json
from pathlib import Path
from project_paths import DATASET as DATASET_ROOT


def main():
    """Check original dataset hashes and completed training artifacts."""
    project = Path(__file__).resolve().parent
    source = DATASET_ROOT
    original = json.loads((project / "original_dataset_hashes.json").read_text(encoding="utf-8"))
    current = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in source.rglob("*") if p.is_file()}
    integrity = {"unchanged": original == current, "original_files": len(original), "current_files": len(current)}
    (project / "runs" / "dataset_integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    print("Source dataset:", integrity)
    rows = []
    for name in ("experiment_a_320", "experiment_b_640"):
        folder = project / "runs" / name
        summary = json.loads((folder / "run_summary.json").read_text(encoding="utf-8"))
        artifacts = ["weights/best.pt", "weights/last.pt", "results.csv", "results.png", "args.yaml"]
        missing = [file for file in artifacts if not (folder / file).is_file()]
        print(name, "missing artifacts:", missing)
        metrics = summary.get("best_validation_metrics", {})
        rows.append(dict(experiment=name, status=summary["status"], epochs=summary.get("epochs_completed"),
                         batch=summary.get("actual_batch"), stopped_early=summary.get("stopped_early"),
                         precision=metrics.get("metrics/precision(B)"), recall=metrics.get("metrics/recall(B)"),
                         mAP50=metrics.get("metrics/mAP50(B)"), mAP50_95=metrics.get("metrics/mAP50-95(B)"),
                         elapsed_seconds=summary["elapsed_seconds"], missing_artifacts=";".join(missing)))
    with (project / "runs" / "experiment_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
