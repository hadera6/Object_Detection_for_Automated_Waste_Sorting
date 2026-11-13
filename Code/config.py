"""
All settings for the garbage detection project, in one place.

data.py and train.py import from here; change settings only in this file.
Paths are built from this file's location, so the scripts work from any folder.
"""
from pathlib import Path

# =================================================================== paths
ROOT = Path(__file__).resolve().parent.parent        # project root (the folder that contains Code/)
DATASET_DIR = ROOT / "GarbageDataset"                # unzipped dataset: images/ + labels/
DATA_YAML = DATASET_DIR / "data.yaml"                # written by data.py


# =================================================================== dataset
# Class IDs follow alphabetical order (verified against the filename prefixes).
CLASS_NAMES = ["biodegradable", "cardboard", "glass", "metal", "paper", "plastic"]
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42