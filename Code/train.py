"""
Step 2: train the YOLOv5 detector and save the weights.

    python Code/train.py              train (skipped if weights/best.pt already exists)
    python Code/train.py --retrain    train again even if the weights exist

Requires data.yaml and the split lists from 'python Code/data.py'.
Fine-tunes a COCO-pretrained YOLOv5s with the default ultralytics augmentation
(mosaic, horizontal flip, HSV colour, scale, translate) and early stopping on the validation set.
The best checkpoint (by validation mAP) is copied to weights/best.pt.
All settings (model, epochs, image size, batch, paths) are in config.py.
"""
import argparse
import shutil

from config import (BATCH, DATA_YAML, EPOCHS, IMG_SIZE, MODEL, PATIENCE, RESULTS_DIR, RUN_NAME, SEED,
                    WEIGHTS_DIR)


# =================================================================== training
def train():
    """Train with ultralytics, copy the best checkpoint to weights/best.pt and return its path."""
    from ultralytics import YOLO

    if not DATA_YAML.exists():
        raise SystemExit(f"{DATA_YAML} not found. Run 'python Code/data.py' first.")

    model = YOLO(MODEL)
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMG_SIZE,
        batch=BATCH,
        seed=SEED,
        project=str(RESULTS_DIR / "runs"),
        name=RUN_NAME,
        exist_ok=True,
    )
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    best = WEIGHTS_DIR / "best.pt"
    shutil.copy2(RESULTS_DIR / "runs" / RUN_NAME / "weights" / "best.pt", best)
    return best


# =================================================================== main
def main():
    ap = argparse.ArgumentParser(description="Train the YOLOv5 garbage detector.")
    ap.add_argument("--retrain", action="store_true", help="train again even if the weights already exist")
    args = ap.parse_args()

    best = WEIGHTS_DIR / "best.pt"
    if best.exists() and not args.retrain:
        print(f"{best} already exists: skipping training (use --retrain to train again).")
        return
    print(f"Best model saved to {train()}")


if __name__ == "__main__":
    main()