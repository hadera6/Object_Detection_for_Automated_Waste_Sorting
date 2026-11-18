"""Settings for the garbage detection project (used by all scripts in Code/)."""
from pathlib import Path

# Paths (relative to the project root, so the code runs from any folder)
ROOT = Path(__file__).resolve().parent.parent   # project folder that contains Code/
DATASET_DIR = ROOT / "GarbageDataset"           # images/ and labels/
DATA_YAML = DATASET_DIR / "data.yaml"           # YOLO dataset file, written by data.py
RESULTS_DIR = ROOT / "results"                  # training runs
WEIGHTS_DIR = ROOT / "weights"                  # final model (best.pt)
EVAL_DIR = RESULTS_DIR / "evaluation"           # test metrics and plots

# Dataset
CLASS_NAMES = ["biodegradable", "cardboard", "glass", "metal", "paper", "plastic"]  # class IDs 0-5
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42                                       # same split on every run

# Training
MODEL = "yolov5su.pt"     # YOLOv5s, COCO-pretrained
RUN_NAME = "yolov5s_640"
EPOCHS = 50
PATIENCE = 15             # early stopping after 15 epochs without improvement
IMG_SIZE = 640            # images upscaled from 416 px to help small objects
BATCH = 16

# Evaluation
MAX_DET = 500             # max boxes per image (some images have > 300 objects)