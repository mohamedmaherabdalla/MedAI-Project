"""Build the baseline-stage split as symlink folders: train = PolypGen C1-C4, val = C5.

Each split folder has images/<frame>.jpg and masks/<frame>.png, the layout PraNet and
Polyp-PVT read. nnU-Net gets its own copy in its raw format (it reads no JPEG).
C6 and the sequence data are never used.
Run:
    PYTHONPATH=src python -m medai.split
"""
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

DATA_ROOT = Path("/l/users/mohamed.abdalla/medai/data/polypgen/PolypGen2021_MultiCenterData_v3")
SPLIT_ROOT = Path("/l/users/mohamed.abdalla/medai/data/splits")
SPLITS = {"train_c1-c4": ["C1", "C2", "C3", "C4"], "val_c5": ["C5"]}
NNUNET_RAW = Path("/l/users/mohamed.abdalla/medai/data/nnunet_raw/Dataset501_PolypGenC1C4")


def centre_frames(centre, data_root=DATA_ROOT):
    """Return (frame name, image path, mask path) for every single frame of one centre."""
    folder = data_root / f"data_{centre}"
    # One C3 image is released as "C3_EndoCV2021_00489].jpg"; its mask has no bracket.
    images = {p.stem.rstrip("]"): p for p in (folder / f"images_{centre}").glob("*.jpg")}
    frames = []
    for mask in sorted((folder / f"masks_{centre}").glob("*_mask.jpg")):
        stem = mask.stem.removesuffix("_mask")
        frames.append((f"{centre}_{stem}", images.pop(stem), mask))
    if images:
        raise ValueError(f"{centre} images without a mask: {sorted(images)}")
    return frames


def build_split(name, centres, data_root=DATA_ROOT, split_root=SPLIT_ROOT):
    """Create split_root/name/{images,masks} with one symlink per frame. Returns the folder."""
    out = Path(split_root) / name
    if out.exists():
        shutil.rmtree(out)  # rebuild from scratch so no stale link survives
    (out / "images").mkdir(parents=True)
    (out / "masks").mkdir()
    for centre in centres:
        for frame, image, mask in centre_frames(centre, data_root):
            (out / "images" / f"{frame}.jpg").symlink_to(image)
            # The repos list only *.png masks; PIL and OpenCV read the JPEG content anyway.
            (out / "masks" / f"{frame}.png").symlink_to(mask)
    return out


def build_nnunet_raw(out=NNUNET_RAW, split_root=SPLIT_ROOT):
    """Write train (imagesTr, labelsTr) and C5 (imagesTs) in nnU-Net's raw format, as PNG."""
    for folder in ["imagesTr", "labelsTr", "imagesTs"]:
        (Path(out) / folder).mkdir(parents=True, exist_ok=True)
    train, val = Path(split_root) / "train_c1-c4", Path(split_root) / "val_c5"
    for image in sorted((train / "images").iterdir()):
        Image.open(image).convert("RGB").save(Path(out) / "imagesTr" / f"{image.stem}_0000.png")
        mask = np.asarray(Image.open(train / "masks" / f"{image.stem}.png").convert("L")) > 127  # 0.5, as in evaluate
        Image.fromarray(mask.astype(np.uint8)).save(Path(out) / "labelsTr" / f"{image.stem}.png")
    for image in sorted((val / "images").iterdir()):
        Image.open(image).convert("RGB").save(Path(out) / "imagesTs" / f"{image.stem}_0000.png")
    dataset = {"channel_names": {"0": "R", "1": "G", "2": "B"}, "labels": {"background": 0, "polyp": 1},
               "numTraining": len(list((train / "images").iterdir())), "file_ending": ".png"}
    (Path(out) / "dataset.json").write_text(json.dumps(dataset, indent=2))


def main():
    for name, centres in SPLITS.items():
        out = build_split(name, centres)
        print(f"{out}: {len(list((out / 'images').iterdir()))} frames from {centres}")
    build_nnunet_raw()
    print(f"{NNUNET_RAW}: nnU-Net raw dataset")


if __name__ == "__main__":
    main()
