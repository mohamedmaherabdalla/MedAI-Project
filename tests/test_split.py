import hashlib

import pytest

from medai.split import DATA_ROOT, SPLITS, build_split

pytestmark = pytest.mark.skipif(not DATA_ROOT.exists(), reason="PolypGen is not on this machine")


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_train_split_has_no_c5_or_c6_frame(tmp_path):
    train = build_split("train_c1-c4", SPLITS["train_c1-c4"], split_root=tmp_path)
    images = sorted((train / "images").iterdir())
    masks = sorted((train / "masks").iterdir())
    assert len(images) == len(masks) == 256 + 301 + 457 + 227

    # Every link points into C1-C4, never into C5 or C6.
    centres = {p.resolve().relative_to(DATA_ROOT).parts[0] for p in images + masks}
    assert centres == {"data_C1", "data_C2", "data_C3", "data_C4"}

    # No training image has the same bytes as a C5 or C6 image (catches copies across centres).
    held_out = {md5(p) for c in ["C5", "C6"] for p in (DATA_ROOT / f"data_{c}" / f"images_{c}").glob("*")}
    assert not {md5(p) for p in images} & held_out


def test_image_and_mask_names_pair_up(tmp_path):
    val = build_split("val_c5", SPLITS["val_c5"], split_root=tmp_path)
    images = [p.stem for p in sorted((val / "images").iterdir())]
    masks = [p.stem for p in sorted((val / "masks").iterdir())]
    assert images == masks and len(images) == 208
