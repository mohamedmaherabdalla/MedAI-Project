"""Our one evaluation: Dice and IoU per frame (masks binarised at 0.5), mean over frames.

Score one run, and with --save add its row to the "runs" sheet of docs/results.xlsx:
    PYTHONPATH=src python -m medai.evaluate configs/<run>.yaml --save --logs <slurm log> ...
GPU type, date and train time come from the Slurm logs; --date and --hours give them for runs whose
logs predate the timestamps (the sacct database was unreachable on 2026-10-14).
"""
import argparse
import re
from pathlib import Path

import numpy as np
import yaml
from openpyxl import Workbook, load_workbook
from PIL import Image

RESULTS = Path(__file__).resolve().parents[2] / "docs" / "results.xlsx"
RUN_COLUMNS = ["run_id", "date", "member", "model", "paper (venue, year)", "repo@commit", "backbone",
               "pretrained weights", "input size", "augmentation", "loss", "optimizer", "learning rate",
               "batch size", "epochs", "checkpoint rule", "train centres", "n_train frames", "val centre",
               "n_val frames", "seed", "GPU type", "train time (h)", "Dice", "IoU",
               "empty-prediction frames on val", "notes", "output path"]
NOTE_COLUMNS = ["date", "author", "note"]
# Run columns copied straight from the config, and the config key each one comes from.
CONFIG_KEYS = {"run_id": "name", "member": "member", "model": "model", "paper (venue, year)": "paper",
               "backbone": "backbone", "pretrained weights": "pretrained_weights", "input size": "input_size",
               "augmentation": "augmentation", "loss": "loss", "optimizer": "optimizer",
               "learning rate": "learning_rate", "batch size": "batch_size", "epochs": "epochs",
               "checkpoint rule": "checkpoint_rule", "train centres": "train_centres",
               "val centre": "val_centre", "seed": "seed", "notes": "notes", "output path": "run_dir"}


def binary_mask(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255 > 0.5


def dice_iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0, 1.0  # both masks empty: the prediction is exactly right
    return 2 * inter / (pred.sum() + gt.sum()), inter / union


def evaluate(pred_dir, gt_dir, final=False):
    """Mean Dice, mean IoU and the number of empty predictions over every mask in gt_dir.

    The prediction for <frame> is pred_dir/<frame>.png.
    """
    gt_paths = sorted(Path(gt_dir).iterdir())
    if not final and any("data_C6" in str(p.resolve()) for p in gt_paths):
        raise SystemExit("C6 is the held-out test centre: evaluate on it only with --final")
    scores, empty = [], 0
    for gt_path in gt_paths:
        pred, gt = binary_mask(Path(pred_dir) / f"{gt_path.stem}.png"), binary_mask(gt_path)
        if pred.shape != gt.shape:
            raise ValueError(f"{gt_path.stem}: prediction {pred.shape} but mask {gt.shape}")
        scores.append(dice_iou(pred, gt))
        empty += int(not pred.any())
    dice, iou = np.mean(scores, axis=0)
    return float(dice), float(iou), empty


def open_results(path):
    """Load the results workbook, or create it with its 'runs' and 'notes' sheets."""
    if Path(path).exists():
        return load_workbook(path)
    book = Workbook()
    book.active.title = "runs"
    book.active.append(RUN_COLUMNS)
    book.create_sheet("notes").append(NOTE_COLUMNS)
    return book


def add_run(row, path=RESULTS):
    """Append one run to the 'runs' sheet; row is a dict with a value for every column."""
    book = open_results(path)
    book["runs"].append([row[column] for column in RUN_COLUMNS])
    book.save(path)


def add_note(date, author, note, path=RESULTS):
    book = open_results(path)
    book["notes"].append([date, author, note])
    book.save(path)


def log_facts(logs):
    """GPU type, start date and summed Slurm run time (h) from the job logs; None where a log lacks it."""
    gpu, date, hours = None, None, 0.0
    for text in (Path(log).read_text() for log in logs):
        found = re.search(r"^(NVIDIA [^,]+),", text, re.M)  # the nvidia-smi line
        gpu = gpu or (found and found[1])
        start, runtime = re.search(r"StartTime=(\S+)", text), re.search(r"RunTime=(\S+)", text)
        if not (start and runtime):
            return gpu, None, None
        date = date or start[1][:10]
        days, _, clock = runtime[1].rpartition("-")
        h, m, s = map(int, clock.split(":"))
        hours += int(days or 0) * 24 + h + m / 60 + s / 3600
    return gpu, date, round(hours, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--final", action="store_true", help="allow the held-out C6 centre")
    parser.add_argument("--save", action="store_true", help="add the run to docs/results.xlsx")
    parser.add_argument("--logs", nargs="+", default=[], help="Slurm logs of the run's jobs")
    parser.add_argument("--date", help="run date, when the logs do not have it")
    parser.add_argument("--hours", type=float, help="Slurm run time in hours, when the logs do not have it")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    dice, iou, empty = evaluate(cfg["pred_dir"], Path(cfg["val_dir"]) / "masks", args.final)
    print(f"{cfg['name']}: Dice {dice:.4f}  IoU {iou:.4f}  empty {empty}  ({cfg['pred_dir']})")
    if args.save:
        gpu, date, hours = log_facts(args.logs)
        row = {column: cfg[key] for column, key in CONFIG_KEYS.items()}
        row.update({"date": args.date or date or "unknown", "repo@commit": f"{cfg['repo_url']}@{cfg['commit']}",
                    "n_train frames": len(list(Path(cfg["train_dir"], "images").iterdir())),
                    "n_val frames": len(list(Path(cfg["val_dir"], "images").iterdir())),
                    "GPU type": gpu or "unknown", "train time (h)": args.hours or hours or "unknown",
                    "Dice": round(dice, 4), "IoU": round(iou, 4), "empty-prediction frames on val": empty})
        add_run(row)


if __name__ == "__main__":
    main()
