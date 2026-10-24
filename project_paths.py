"""Portable project paths; set MMA3001_DATASET to use a different dataset folder."""

import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
DATASET = Path(os.environ.get("MMA3001_DATASET", str(PROJECT.parent / "dataset"))).expanduser().resolve()
