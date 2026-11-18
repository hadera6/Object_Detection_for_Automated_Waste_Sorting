"""Step 1: clean the labels, split the dataset 70/15/15 and write data.yaml for YOLO."""
import random
import re
from collections import defaultdict

from config import CLASS_NAMES, DATA_YAML, DATASET_DIR, SEED, SPLIT_RATIOS


# ----------------------------------------------------------------- reading
def image_class(filename):
    """Class of an image, taken from its filename prefix: 'glass123_jpg...' -> 'glass'."""
    return re.match(r"[a-z]+", filename.lower()).group(0)


def parse_label_line(line):
    """Convert one label line 'class x y w h' to numbers; return None if the box is invalid."""
    parts = line.split()
    if len(parts) != 5:
        return None
    c, x, y, w, h = int(parts[0]), *map(float, parts[1:])
    valid = (0 <= c < len(CLASS_NAMES) and w > 0 and h > 0
             and all(0 <= v <= 1 for v in (x, y, w, h)))
    return (c, x, y, w, h) if valid else None


def load_dataset(dataset_dir):
    """Read all images and labels.

    Returns:
        records: one dict per image with its paths, class and valid boxes
        invalid: number of invalid boxes per image name
    """
    records, invalid = [], {}
    for img in sorted((dataset_dir / "images").glob("*.jpg")):
        label = dataset_dir / "labels" / f"{img.stem}.txt"
        lines = [ln for ln in label.read_text().splitlines() if ln.strip()] if label.exists() else []
        boxes = [b for b in map(parse_label_line, lines) if b is not None]
        if len(boxes) < len(lines):
            invalid[img.name] = len(lines) - len(boxes)
        records.append({"image": img, "label": label, "source": image_class(img.name), "boxes": boxes})
    return records, invalid


# ----------------------------------------------------------------- cleaning
def remove_invalid_boxes(records, invalid):
    """Rewrite the label files that had invalid boxes, keeping only their valid boxes."""
    for r in records:
        if r["image"].name in invalid:
            lines = [f"{c} {x} {y} {w} {h}" for c, x, y, w, h in r["boxes"]]
            r["label"].write_text("\n".join(lines) + ("\n" if lines else ""))


# ----------------------------------------------------------------- splitting
def stratified_split(records, ratios, seed):
    """Split each class separately by the given ratios, so every split keeps the same class mix."""
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


# ----------------------------------------------------------------- output files
def write_split_lists(splits, dataset_dir):
    """Save the split as train.txt, val.txt and test.txt, one image path per line."""
    for name, recs in splits.items():
        paths = sorted(f"./images/{r['image'].name}" for r in recs)
        (dataset_dir / f"{name}.txt").write_text("\n".join(paths) + "\n")


def write_data_yaml(dataset_dir):
    """Write data.yaml: where the split lists are, and the class names."""
    names = "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES))
    (dataset_dir / "data.yaml").write_text(
        f"path: {dataset_dir.resolve()}\n"
        f"train: train.txt\nval: val.txt\ntest: test.txt\n\n"
        f"nc: {len(CLASS_NAMES)}\nnames:\n{names}")


# ----------------------------------------------------------------- main
def main():
    if not (DATASET_DIR / "images").is_dir():
        raise SystemExit(f"Dataset not found: {DATASET_DIR}/images\n"
                         "Unzip GarbageDataset.zip in the project folder first.")

    records, invalid = load_dataset(DATASET_DIR)
    remove_invalid_boxes(records, invalid)
    n_boxes = sum(len(r["boxes"]) for r in records)
    print(f"{len(records):,} images, {n_boxes:,} boxes, {sum(invalid.values())} invalid boxes removed")

    splits = stratified_split(records, SPLIT_RATIOS, SEED)
    write_split_lists(splits, DATASET_DIR)
    write_data_yaml(DATASET_DIR)
    print("Split: " + ", ".join(f"{name} {len(recs):,}" for name, recs in splits.items()))
    print(f"data.yaml written to {DATA_YAML}")


if __name__ == "__main__":
    main()