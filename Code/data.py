"""
Step 1: prepare the data for training.

    python Code/data.py

1. Loads all images and YOLO labels.
2. Removes zero-size boxes (width or height = 0) from the label files.
3. Splits the images 70/15/15 into train/val/test, stratified by image class, with a fixed seed.
   The split is virtual: train.txt, val.txt and test.txt list the image paths (no copying).
4. Writes data.yaml for YOLO.

All settings (paths, classes, split ratios, seed) are in config.py.

Project layout:
    Computer_Vision/
    ├── Code/config.py, Code/data.py, Code/train.py
    └── GarbageDataset/images, labels   (+ train.txt, val.txt, test.txt, data.yaml written here)
"""
import re

from config import CLASS_NAMES, DATASET_DIR


# =================================================================== loading
def image_class(filename):
    """Class the image was collected for, from the filename prefix ('glass123_jpg...' -> 'glass')."""
    return re.match(r"[a-z]+", filename.lower()).group(0)