"""Step 5: save 3 example test images per outcome (correct, missed, wrong class, false positive) with coloured boxes."""
import random

from PIL import Image, ImageDraw

from config import CLASS_NAMES, EVAL_DIR
from error_analysis import match_test_set

COLORS = {"correct": "#1baf7a", "missed": "#2a78d6", "wrong_class": "#eda100", "false_pos": "#e34948"}
N_EXAMPLES, MAX_OBJECTS, EXAMPLE_SEED = 3, 15, 0   # 3 images per outcome, at most 15 objects each, same picks every run


def draw(image_path, outcomes):
    """Draw every box of one image in its outcome colour with a short label."""
    im = Image.open(image_path).convert("RGB")
    d = ImageDraw.Draw(im)
    for o in outcomes:
        pred = CLASS_NAMES[o["cls"]] if o["cls"] is not None else ""
        real = CLASS_NAMES[o["gt_cls"]] if o["gt_cls"] is not None else ""
        text = {"correct": f"{pred} {o['conf'] or 0:.2f}", "missed": f"MISSED {real}",
                "wrong_class": f"{pred} (is {real})", "false_pos": f"FP {pred} {o['conf'] or 0:.2f}"}[o["kind"]]
        x1, y1 = float(o["box"][0]), float(o["box"][1])
        d.rectangle([float(v) for v in o["box"]], outline=COLORS[o["kind"]], width=3)
        d.rectangle([x1, y1, x1 + d.textlength(text) + 4, y1 + 11], fill=COLORS[o["kind"]])
        d.text((x1 + 2, y1), text, fill="white")
    return im


def save_examples(results):
    """Save examples_<outcome>.png with N_EXAMPLES random test images side by side."""
    rng = random.Random(EXAMPLE_SEED)
    for kind in COLORS:
        pool = [r for r in results if r["n"][kind] and r["n_gt"] <= MAX_OBJECTS]
        if not pool:
            continue
        images = [draw(r["image"], r["outcomes"]) for r in rng.sample(pool, min(N_EXAMPLES, len(pool)))]
        grid = Image.new("RGB", (sum(im.width + 10 for im in images), max(im.height for im in images)), "white")
        for k, im in enumerate(images):
            grid.paste(im, (sum(i.width + 10 for i in images[:k]), 0))
        grid.save(EVAL_DIR / f"examples_{kind}.png")


def main(results=None):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    save_examples(results or match_test_set())   # predicts itself if run alone
    print(f"Saved examples_correct/missed/wrong_class/false_pos.png to {EVAL_DIR}")


if __name__ == "__main__":
    main()