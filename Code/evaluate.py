"""
Step 3: evaluate the trained model on the test set.

    python Code/evaluate.py

Requires weights/best.pt (from train.py) and the split lists from data.py.
Computes precision, recall, F1, AP@0.5 and AP@[.50:.95] per class, plus the overall values
(mAP = mean AP over the classes), on the held-out test set that was never used in training.
Saves test_metrics.json and test_per_class.csv to results/evaluation/; ultralytics also saves
the confusion matrix and the precision-recall curve to results/evaluation/test/.
"""
import csv
import json

from config import BATCH, DATA_YAML, EVAL_DIR, IMG_SIZE, MAX_DET, WEIGHTS_DIR


# =================================================================== metrics
def f1_score(precision, recall):
    """Harmonic mean of precision and recall (0 if both are 0)."""
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate():
    """Run the ultralytics evaluation on the test split; return overall metrics and one row per class."""
    from ultralytics import YOLO

    best = WEIGHTS_DIR / "best.pt"
    if not best.exists():
        raise SystemExit(f"{best} not found. Run 'python Code/train.py' first.")
    if not DATA_YAML.exists():
        raise SystemExit(f"{DATA_YAML} not found. Run 'python Code/data.py' first.")

    model = YOLO(str(best))
    m = model.val(data=str(DATA_YAML), split="test", imgsz=IMG_SIZE, batch=BATCH, max_det=MAX_DET,
                  project=str(EVAL_DIR), name="test", exist_ok=True, plots=True)

    overall = {"mAP50": float(m.box.map50), "mAP50-95": float(m.box.map),
               "precision": float(m.box.mp), "recall": float(m.box.mr)}
    overall["f1"] = f1_score(overall["precision"], overall["recall"])

    rows = []
    for k, class_id in enumerate(m.box.ap_class_index):
        p, r = float(m.box.p[k]), float(m.box.r[k])
        rows.append({"class": model.names[int(class_id)], "precision": p, "recall": r,
                     "f1": f1_score(p, r), "AP50": float(m.box.ap50[k]), "AP50-95": float(m.box.ap[k])})
    rows.sort(key=lambda row: row["AP50-95"])
    return overall, rows


# =================================================================== output
def save_results(overall, rows):
    """Write test_metrics.json and test_per_class.csv to EVAL_DIR."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "test_metrics.json").write_text(json.dumps({k: round(v, 4) for k, v in overall.items()}, indent=2))
    with open(EVAL_DIR / "test_per_class.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()} for row in rows)


def print_results(overall, rows):
    print(f"\nTest set:  mAP@0.5 = {overall['mAP50']:.3f}   mAP@[.50:.95] = {overall['mAP50-95']:.3f}   "
          f"precision = {overall['precision']:.3f}   recall = {overall['recall']:.3f}   F1 = {overall['f1']:.3f}\n")
    print(f"{'class':<15}{'precision':>10}{'recall':>8}{'F1':>7}{'AP@0.5':>8}{'AP@[.50:.95]':>14}")
    for row in rows:
        print(f"{row['class']:<15}{row['precision']:>10.3f}{row['recall']:>8.3f}{row['f1']:>7.3f}"
              f"{row['AP50']:>8.3f}{row['AP50-95']:>14.3f}")
    print(f"\nSaved to {EVAL_DIR}")


# =================================================================== main
def main():
    overall, rows = evaluate()
    save_results(overall, rows)
    print_results(overall, rows)


if __name__ == "__main__":
    main()