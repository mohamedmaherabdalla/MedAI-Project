import numpy as np
import pytest
from PIL import Image

from medai.evaluate import evaluate


def save(folder, name, pixels):
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.array(pixels, dtype=np.uint8)).save(folder / f"{name}.png")


def make_example(gt_dir, pred_dir):
    # Frame a: 4 true pixels, 4 predicted, 2 shared -> Dice 4/8 = 0.5, IoU 2/6.
    # 127 is below 0.5 after /255 and must count as background; 200 counts as polyp.
    save(gt_dir, "a", [[255, 255, 0], [255, 255, 0], [0, 0, 0]])
    save(pred_dir, "a", [[200, 255, 255], [127, 0, 255], [0, 0, 0]])
    # Frame b: prediction equals the mask -> Dice 1, IoU 1.
    save(gt_dir, "b", [[0, 255], [0, 0]])
    save(pred_dir, "b", [[0, 255], [0, 0]])


def test_mean_dice_and_iou_on_a_known_example(tmp_path):
    make_example(tmp_path / "gt", tmp_path / "pred")
    dice, iou = evaluate(tmp_path / "pred", tmp_path / "gt")
    assert dice == pytest.approx((0.5 + 1) / 2)
    assert iou == pytest.approx((2 / 6 + 1) / 2)


def test_refuses_c6_without_final(tmp_path):
    gt_dir, pred_dir = tmp_path / "data_C6" / "masks_C6", tmp_path / "pred"
    make_example(gt_dir, pred_dir)
    with pytest.raises(SystemExit):
        evaluate(pred_dir, gt_dir)
    assert evaluate(pred_dir, gt_dir, final=True)[0] == pytest.approx(0.75)
