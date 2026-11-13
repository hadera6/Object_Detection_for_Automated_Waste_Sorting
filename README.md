# Garbage Detection with YOLOv5

Object detection for automated waste sorting: find and classify
biodegradable, cardboard, glass, metal, paper and plastic items in images.
Computer Vision Portfolio WS2026, Assignment 1.

## Structure

```
Code/
  config.py    all settings: paths, classes, split, model, epochs
  data.py      load labels, remove zero-size boxes, 70/15/15 split, data.yaml
  train.py     train YOLOv5s and save weights/best.pt
weights/best.pt   trained model
packages.txt      dependencies
GarbageDataset/   dataset (not in the repository): images/ + labels/
```

## Usage

```bash
pip install -r packages.txt
# unzip the dataset so that GarbageDataset/images and GarbageDataset/labels exist
python Code/data.py               # clean labels, split, write data.yaml
python Code/train.py              # train (skipped if weights/best.pt exists)
python Code/train.py --retrain    # train again
```

A GPU is recommended (about 2 hours on a T4 for 50 epochs).

## Result

YOLOv5s, 640 px, 50 epochs, on the held-out test set (1,571 images):
mAP@0.5 = 0.645, mAP@[.50:.95] = 0.444, precision 0.70, recall 0.57.