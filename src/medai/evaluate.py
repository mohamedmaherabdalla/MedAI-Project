"""Our one evaluation: Dice and IoU per frame (masks binarised at 0.5), mean over frames.

Score one run, and with --save append its row to docs/results.xlsx:
    PYTHONPATH=src python -m medai.evaluate configs/<run>.yaml --save
"""
import argparse
from datetime import date
from pathlib import Path

import numpy as np
import yaml
from openpyxl import Workbook, load_workbook
from PIL import Image

RESULTS = Path(__file__).resolve().parents[2] / "docs" / "results.xlsx"
COLUMNS = ["date", "model", "repo@commit", "train centres", "val centre", "seed", "epochs",
           "Dice", "IoU", "notes", "output path"]


def binary_mask(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255 > 0.5


def dice_iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0, 1.0  # both masks empty: the prediction is exactly right
    return 2 * inter / (pred.sum() + gt.sum()), inter / union


def evaluate(pred_dir, gt_dir, final=False):
    """Mean (Dice, IoU) over every mask in gt_dir; the prediction for <frame> is pred_dir/<frame>.png."""
    gt_paths = sorted(Path(gt_dir).iterdir())
    if not final and any("data_C6" in str(p.resolve()) for p in gt_paths):
        raise SystemExit("C6 is the held-out test centre: evaluate on it only with --final")
    scores = []
    for gt_path in gt_paths:
        pred, gt = binary_mask(Path(pred_dir) / f"{gt_path.stem}.png"), binary_mask(gt_path)
        if pred.shape != gt.shape:
            raise ValueError(f"{gt_path.stem}: prediction {pred.shape} but mask {gt.shape}")
        scores.append(dice_iou(pred, gt))
    dice, iou = np.mean(scores, axis=0)
    return float(dice), float(iou)


def add_result_row(row, path=RESULTS):
    """Append one run to the results sheet, creating the sheet with its header if needed."""
    if path.exists():
        book = load_workbook(path)
    else:
        book = Workbook()
        book.active.append(COLUMNS)
    book.active.append(row)
    book.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--final", action="store_true", help="allow the held-out C6 centre")
    parser.add_argument("--save", action="store_true", help="append the row to docs/results.xlsx")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    dice, iou = evaluate(cfg["pred_dir"], Path(cfg["val_dir"]) / "masks", args.final)
    print(f"{cfg['name']}: Dice {dice:.4f}  IoU {iou:.4f}  ({cfg['pred_dir']})")
    if args.save:
        add_result_row([date.today().isoformat(), cfg["model"], f"{cfg['repo_url']}@{cfg['commit']}",
                        cfg["train_centres"], cfg["val_centre"], cfg["seed"], cfg["epochs"],
                        round(dice, 4), round(iou, 4), cfg["notes"], cfg["run_dir"]])


if __name__ == "__main__":
    main()
