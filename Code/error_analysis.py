"""Step 4: sort every test box into correct / wrong class / false positive / missed and save the counts as CSV."""
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from config import BATCH, CLASS_NAMES, DATASET_DIR, EVAL_DIR, IMG_SIZE, MAX_DET, WEIGHTS_DIR

CONF_THRESHOLD = 0.25   # count only predictions at least this confident (ultralytics default)
IOU_MATCH = 0.5         # overlap needed to match a real box (same as mAP@0.5)


def load_gt(label_path, W, H):
    """Real boxes of one image as pixel corners [x1, y1, x2, y2] and class IDs."""
    rows = [list(map(float, ln.split())) for ln in label_path.read_text().splitlines() if ln.strip()]
    boxes = [[(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H] for _, x, y, w, h in rows]
    return np.array(boxes).reshape(-1, 4), np.array([int(r[0]) for r in rows])


def iou(a, b):
    """IoU between every box in a (N, 4) and every box in b (M, 4)."""
    lt, rb = np.maximum(a[:, None, :2], b[None, :, :2]), np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match_image(boxes, cls, conf, gt_boxes, gt_cls):
    """Match predictions to real boxes, most confident first; return one outcome per box."""
    ious, used, out = iou(boxes, gt_boxes), np.zeros(len(gt_boxes), bool), []
    for i in np.argsort(-conf):
        cand = np.where(~used & (ious[i] >= IOU_MATCH))[0]
        same = cand[gt_cls[cand] == cls[i]]
        cand = same if len(same) else cand                      # prefer a real box of the same class
        if len(cand):
            j = cand[np.argmax(ious[i, cand])]
            used[j] = True
            kind = "correct" if gt_cls[j] == cls[i] else "wrong_class"
            out.append(dict(kind=kind, box=boxes[i], cls=int(cls[i]), gt_cls=int(gt_cls[j]), conf=conf[i]))
        else:
            out.append(dict(kind="false_pos", box=boxes[i], cls=int(cls[i]), gt_cls=None, conf=conf[i]))
    out += [dict(kind="missed", box=gt_boxes[j], cls=None, gt_cls=int(gt_cls[j]), conf=None)
            for j in np.where(~used)[0]]
    return out


def match_test_set():
    """Predict all test images in batches and match them; one result per image."""
    from ultralytics import YOLO

    model = YOLO(str(WEIGHTS_DIR / "best.pt"))
    images = [str(DATASET_DIR / p) for p in (DATASET_DIR / "test.txt").read_text().split()]
    results = []
    for start in range(0, len(images), BATCH):
        for r in model.predict(images[start:start + BATCH], imgsz=IMG_SIZE, conf=CONF_THRESHOLD,
                               max_det=MAX_DET, verbose=False):
            H, W = r.orig_shape
            gt_boxes, gt_cls = load_gt(DATASET_DIR / "labels" / f"{Path(r.path).stem}.txt", W, H)
            b = r.boxes
            out = match_image(b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy(),
                              gt_boxes, gt_cls)
            results.append(dict(image=r.path, n_gt=len(gt_boxes), outcomes=out, n=Counter(o["kind"] for o in out)))
    return results


def save_csv(results):
    """Write error_analysis.csv: per class and in total, how many objects are correct, wrong, missed."""
    count = Counter()
    for r in results:
        for o in r["outcomes"]:
            count[o["kind"], o["cls"] if o["kind"] == "false_pos" else o["gt_cls"]] += 1

    rows = []
    for name, ids in [*((n, [i]) for i, n in enumerate(CLASS_NAMES)), ("total", range(len(CLASS_NAMES)))]:
        n = {k: sum(count[k, i] for i in ids) for k in ("correct", "wrong_class", "missed", "false_pos")}
        objects = n["correct"] + n["wrong_class"] + n["missed"]
        rows.append({"class": name, "objects": objects, **n, "recall": round(n["correct"] / objects, 3)})

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVAL_DIR / "error_analysis.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows[-1]


def main():
    results = match_test_set()
    t = save_csv(results)
    print(f"{t['objects']:,} test objects: {t['correct']:,} correct, {t['wrong_class']:,} wrong class, "
          f"{t['missed']:,} missed, {t['false_pos']:,} false positives -> {EVAL_DIR / 'error_analysis.csv'}")
    return results


if __name__ == "__main__":
    main()