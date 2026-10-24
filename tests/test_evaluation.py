"""Small CPU-only tests; no checkpoints, GPU, or retraining required."""

import pytest

import dataset_audit
from evaluation_utils import evaluation_settings, latency_statistics, load_names, parse_labels


def test_valid_and_empty_labels(tmp_path):
    """Accept valid class IDs and empty background labels.

    Args:
        tmp_path: Temporary directory provided by pytest.
    """
    path = tmp_path / "labels.txt"
    path.write_text("0 0.5 0.5 0.2 0.3\n4 0 1 1 0.1\n", encoding="utf-8")
    assert [box[0] for box in parse_labels(path)] == [0, 4]
    path.write_text("\n  \n", encoding="utf-8")
    assert parse_labels(path) == []


@pytest.mark.parametrize("line", ["5 0.5 0.5 0.2 0.2", "-1 0.5 0.5 0.2 0.2", "0.5 0.5 0.5 0.2 0.2",
    "0 nan 0.5 0.2 0.2", "0 0.5 0.5 0 0.2", "0 1.1 0.5 0.2 0.2", "0 0.5 0.5 -0.2 0.2",
    "0 0.5 0.5 0.2", "bad 0.5 0.5 0.2 0.2", "0 inf 0.5 0.2 0.2"])
def test_reject_invalid_labels(tmp_path, line):
    """Reject malformed, nonfinite, out-of-range and fractional-class labels.

    Args:
        tmp_path: Temporary test directory.
        line: Invalid annotation under test.
    """
    path = tmp_path / "bad.txt"
    path.write_text(line, encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        parse_labels(path)


def test_configuration_consistency():
    """Use identical test settings except for image resolution."""
    a, b = evaluation_settings(320), evaluation_settings(640)
    assert {k for k in a if a[k] != b[k]} == {"imgsz"}
    assert a["split"] == "test" and a["device"] == 0 and a["half"] is False
    with pytest.raises(ValueError):
        evaluation_settings(321)


def test_class_names(tmp_path):
    """Validate ordered class names and declared class count.

    Args:
        tmp_path: Temporary test directory.
    """
    path = tmp_path / "data.yaml"
    path.write_text("nc: 2\nnames: {0: pork, 1: package}\n", encoding="utf-8")
    assert load_names(path) == ["pork", "package"]
    path.write_text("nc: 3\nnames: [pork, package]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_names(path)


def test_latency_statistics():
    """Check the millisecond-to-FPS calculation with known values."""
    result = latency_statistics([2, 4, 6])
    assert result == {"mean_ms": 4, "median_ms": 4, "fps": 250}
    with pytest.raises(ValueError):
        latency_statistics([])


def test_absent_class_is_not_zero():
    """Preserve unavailable class metrics as None rather than false zero scores."""
    from types import SimpleNamespace
    from evaluate_models import per_class_rows

    box = SimpleNamespace(ap_class_index=[0], class_result=lambda i: (0.8, 0.7, 0.6, 0.5))
    rows = per_class_rows(SimpleNamespace(box=box), ["pork", "wrinkle"], [3, 0], "example")
    assert rows[0]["mAP50"] == 0.6
    assert rows[1]["mAP50"] is None
    assert rows[1]["precision"] is None
    assert rows[1]["status"] == "not_evaluable_no_ground_truth"


def test_example_box_matching():
    """Ensure one ground-truth box cannot match two duplicate predictions."""
    from evaluate_models import match_boxes

    gt = [(0, 10, 10, 20, 20)]
    assert match_boxes(gt, gt) == (1, 0, 0)
    assert match_boxes(gt, gt + gt) == (1, 0, 1)
    assert match_boxes(gt, [(1, 10, 10, 20, 20)]) == (0, 1, 1)


def test_audit_pairing_and_invalid_rows(tmp_path, monkeypatch):
    """Check real audit behavior against a synthetic split with known issues.

    Args:
        tmp_path: Temporary dataset directory.
        monkeypatch: Pytest fixture to isolate the audit's dataset path.
    """
    images = tmp_path / "test" / "images"
    labels = tmp_path / "test" / "labels"
    images.mkdir(parents=True)
    labels.mkdir()
    for name in ("valid", "empty", "missing_label"):
        (images / f"{name}.jpg").touch()
    (labels / "valid.txt").write_text("0 0.5 0.5 0.2 0.2\n5 0.5 0.5 0.2 0.2\n0 2 0.5 0.2 0.2\n", encoding="utf-8")
    (labels / "empty.txt").write_text("", encoding="utf-8")
    (labels / "orphan.txt").write_text("", encoding="utf-8")
    monkeypatch.setattr(dataset_audit, "DATASET", tmp_path)
    summary, counts, issues = dataset_audit.audit_split("test", ["a", "b", "c", "d", "e"])
    assert summary["images"] == 3 and summary["label_files"] == 3
    assert summary["total_boxes"] == 3 and summary["valid_boxes"] == 1
    assert summary["empty_labels"] == 2
    assert summary["missing_labels"] == summary["missing_images"] == 1
    assert summary["invalid_class_ids"] == summary["invalid_coordinates"] == 1
    assert counts[0] == 2
    assert len(issues) == 4


def test_missing_split_fails(tmp_path, monkeypatch):
    """Missing directories must fail instead of reporting misleading zero counts.

    Args:
        tmp_path: Empty temporary dataset directory.
        monkeypatch: Fixture to replace the configured dataset root.
    """
    monkeypatch.setattr(dataset_audit, "DATASET", tmp_path)
    with pytest.raises(FileNotFoundError):
        dataset_audit.audit_split("test", ["pork"])
