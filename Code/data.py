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
import random
import re
from collections import defaultdict

from config import CLASS_NAMES, DATASET_DIR


# =================================================================== loading
def image_class(filename):
    """Class the image was collected for, from the filename prefix ('glass123_jpg...' -> 'glass')."""
    return re.match(r"[a-z]+", filename.lower()).group(0)

def parse_label_line(line):
    """Return (class, x, y, w, h) for a valid YOLO line, or None if the line is malformed or zero-size."""
    parts = line.split()
    if len(parts) != 5:
        return None
    c, x, y, w, h = int(parts[0]), *map(float, parts[1:])
    if not 0 <= c < len(CLASS_NAMES) or w <= 0 or h <= 0:
        return None
    if not all(0 <= v <= 1 for v in (x, y, w, h)):
        return None
    return c, x, y, w, h


def load_dataset(dataset_dir):
    """Read all images and labels. Returns a list of records and the number of invalid boxes per image."""
    records, invalid = [], {}
    for img in sorted((dataset_dir / "images").glob("*.jpg")):
        label = dataset_dir / "labels" / (img.stem + ".txt")
        lines = [ln for ln in label.read_text().splitlines() if ln.strip()] if label.exists() else []
        boxes = [b for b in map(parse_label_line, lines) if b is not None]
        if len(boxes) < len(lines):
            invalid[img.name] = len(lines) - len(boxes)
        records.append({"image": img, "label": label, "source": image_class(img.name), "boxes": boxes})
    return records, invalid


# =================================================================== cleaning
def remove_invalid_boxes(records, invalid):
    """Rewrite only the label files that contain invalid lines, keeping their valid boxes."""
    for r in records:
        if r["image"].name in invalid:
            lines = [f"{c} {x} {y} {w} {h}" for c, x, y, w, h in r["boxes"]]
            r["label"].write_text("\n".join(lines) + ("\n" if lines else ""))

# =================================================================== split
def stratified_split(records, ratios, seed):
    """Assign each record to train/val/test, keeping the same ratio inside every image class."""
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for r in records:
        by_class[r["source"]].append(r)
    splits = {name: [] for name in ratios}
    for cls in sorted(by_class):
        group = by_class[cls]
        rng.shuffle(group)
        n_train = round(len(group) * ratios["train"])
        n_val = round(len(group) * ratios["val"])
        splits["train"] += group[:n_train]
        splits["val"] += group[n_train:n_train + n_val]
        splits["test"] += group[n_train + n_val:]
    return splits