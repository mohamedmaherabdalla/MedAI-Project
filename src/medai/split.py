"""Build the baseline-stage split as symlink folders: train = PolypGen C1-C4, val = C5.

Each split folder has images/<frame>.jpg and masks/<frame>.png, the layout PraNet and
Polyp-PVT read. C6 and the sequence data are never linked.
Run:
    PYTHONPATH=src python -m medai.split
"""
import shutil
from pathlib import Path

DATA_ROOT = Path("/l/users/mohamed.abdalla/medai/data/polypgen/PolypGen2021_MultiCenterData_v3")
SPLIT_ROOT = Path("/l/users/mohamed.abdalla/medai/data/splits")
SPLITS = {"train_c1-c4": ["C1", "C2", "C3", "C4"], "val_c5": ["C5"]}


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


def main():
    for name, centres in SPLITS.items():
        out = build_split(name, centres)
        print(f"{out}: {len(list((out / 'images').iterdir()))} frames from {centres}")


if __name__ == "__main__":
    main()
