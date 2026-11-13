# Garbage Detection with YOLOv5

Object detection for automated waste sorting: find and classify
biodegradable, cardboard, glass, metal, paper and plastic items in images.
Computer Vision Portfolio WS2026, Assignment 1.

## Structure

```
Code/
  config.py      all settings: paths, classes, split, model, epochs, evaluation
  data.py        load labels, remove zero-size boxes, 70/15/15 split, data.yaml
  train.py       train YOLOv5s and save weights/best.pt
  evaluate.py    test-set metrics: precision, recall, F1, mAP (overall and per class)
  main.py        runs data.py -> train.py -> evaluate.py in one command
garbage_detection.ipynb   full story: dataset analysis, decisions, training,
                          evaluation, error analysis, example detections
weights/best.pt   trained model
packages.txt      dependencies
GarbageDataset/   dataset (not in the repository): images/ + labels/
```

## Usage

```bash
pip install -r packages.txt
# unzip the dataset so that GarbageDataset/images and GarbageDataset/labels exist
python Code/main.py               # whole pipeline: prepare data -> train -> evaluate
python Code/main.py --retrain     # same, but train again even if weights/best.pt exists
```

Or run the steps one by one:

```bash
python Code/data.py               # clean labels, split, write data.yaml
python Code/train.py              # train (skipped if weights/best.pt exists)
python Code/evaluate.py           # test-set metrics -> results/evaluation/
```

A GPU is recommended (about 2 hours on a T4 for 50 epochs).
For the full analysis with explanations, open `garbage_detection.ipynb` (also runs on Google Colab).

## Result

YOLOv5s, 640 px, 50 epochs, on the held-out test set (1,571 images, 11,031 objects):

| Metric | Value |
|---|---|
| mAP@0.5 | 0.645 |
| mAP@[.50:.95] | 0.444 |
| Precision | 0.70 |
| Recall | 0.57 |
| F1 | 0.63 |

Best class: glass (AP@0.5 0.765). Hardest: small, crowded biodegradable items
(AP@[.50:.95] 0.341). Most errors are missed objects; wrong classes are rare (about 2%).