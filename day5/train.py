"""
Step 3 -- fine-tune a YOLO detector on our labeled images.

    python train.py                       # yolov8n, 40 epochs, imgsz 640
    python train.py --epochs 60 --model yolov8s.pt --imgsz 512

What it does:
  1. splits data/raw + data/labels into train/val (copied into data/dataset/)
  2. writes data/dataset.yaml for ultralytics
  3. trains (transfer learning from the COCO-pretrained weights, which are
     downloaded automatically the first time) on Apple GPU (mps) if available
  4. copies the best checkpoint to models/minifig.pt
  5. runs validation and prints precision / recall / mAP -- the numbers for
     "how good is your model?" -- and writes them to models/metrics.txt
"""

import argparse
import os
import random
import shutil

from common import RAW_DIR, LABEL_DIR, DATASET_DIR, DATASET_YAML, MODEL_PATH, CLASS_NAMES, HERE


def split_dataset(val_frac, seed):
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(RAW_DIR) if f.lower().endswith(".jpg"))
    labeled = [n for n in names if os.path.exists(os.path.join(LABEL_DIR, n + ".txt"))]
    if len(labeled) < 10:
        raise SystemExit(f"Only {len(labeled)} labeled images -- run capture.py / label.py first "
                         f"(aim for 150+).")
    random.Random(seed).shuffle(labeled)
    n_val = max(1, int(len(labeled) * val_frac))
    splits = {"val": labeled[:n_val], "train": labeled[n_val:]}

    if os.path.isdir(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    for split, items in splits.items():
        os.makedirs(os.path.join(DATASET_DIR, "images", split))
        os.makedirs(os.path.join(DATASET_DIR, "labels", split))
        for n in items:
            shutil.copy(os.path.join(RAW_DIR, n + ".jpg"), os.path.join(DATASET_DIR, "images", split))
            shutil.copy(os.path.join(LABEL_DIR, n + ".txt"), os.path.join(DATASET_DIR, "labels", split))

    negatives = sum(1 for n in labeled if os.path.getsize(os.path.join(LABEL_DIR, n + ".txt")) == 0)
    with open(DATASET_YAML, "w") as f:
        f.write(f"path: {DATASET_DIR}\ntrain: images/train\nval: images/val\n")
        f.write("names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASS_NAMES)))
    print(f"dataset: {len(splits['train'])} train / {len(splits['val'])} val "
          f"({negatives} negatives)  ->  {DATASET_YAML}")
    return len(splits["train"]), len(splits["val"]), negatives


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt", help="pretrained weights to start from")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="mps / cpu (default: auto)")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    n_train, n_val, n_neg = split_dataset(args.val_frac, args.seed)

    model = YOLO(args.model)
    model.train(data=DATASET_YAML, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=device, project=os.path.join(HERE, "runs"), name="minifig",
                exist_ok=True, patience=max(10, args.epochs // 3), workers=4, plots=True)

    best = str(model.trainer.best)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    shutil.copy(best, MODEL_PATH)
    print(f"\nbest weights -> {MODEL_PATH}")

    m = YOLO(MODEL_PATH).val(data=DATASET_YAML, device=device, plots=True,
                             project=os.path.join(HERE, "runs"), name="minifig_val", exist_ok=True)
    report = (
        f"model: {args.model} fine-tuned {args.epochs} epochs @ {args.imgsz}px on {device}\n"
        f"dataset: {n_train} train / {n_val} val images ({n_neg} negatives)\n"
        f"precision: {m.box.mp:.3f}\nrecall:    {m.box.mr:.3f}\n"
        f"mAP50:     {m.box.map50:.3f}\nmAP50-95:  {m.box.map:.3f}\n"
    )
    with open(os.path.join(os.path.dirname(MODEL_PATH), "metrics.txt"), "w") as f:
        f.write(report)
    print("\n=== validation ===\n" + report)
    print(f"curves + confusion matrix: {os.path.join(HERE, 'runs', 'minifig_val')}")


if __name__ == "__main__":
    main()
