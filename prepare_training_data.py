"""Make an isolated dataset copy and record hashes of the untouched source."""

import hashlib
import json
import shutil
from pathlib import Path
from project_paths import DATASET as DATASET_ROOT

import yaml


def main():
    """Copy dataset files once and write an explicit training configuration."""
    project = Path(__file__).resolve().parent
    source = DATASET_ROOT
    target = project / "training_dataset"
    manifest = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(source.rglob("*")) if p.is_file()}
    (project / "original_dataset_hashes.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shutil.copytree(source, target)
    config = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8"))
    config.update(path=target.as_posix(), train="train/images", val="valid/images", test="test/images")
    (project / "training_data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Copied {len(manifest)} files. Original dataset remains unchanged.")


if __name__ == "__main__":
    main()
