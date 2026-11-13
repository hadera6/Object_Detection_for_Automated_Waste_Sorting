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