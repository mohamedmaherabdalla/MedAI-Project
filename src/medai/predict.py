"""Predict the validation masks with one trained baseline, following the repo test scripts.

Each image is resized to 352 x 352; the sigmoid of the output is resized back to the mask size and
saved as an 8-bit PNG in pred_dir. PraNet (MyTest.py) and Polyp-PVT (Test.py) also stretch every map
to [0, 1] before saving; we keep that step for them because it is their default.
    PYTHONPATH=src python -m medai.predict configs/<run>.yaml
"""
import sys
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import yaml
from PIL import Image

from medai.train import MEAN, STD, import_repo

SIZE = 352  # train and test size of all three baselines


def load_model(cfg):
    if cfg["model"] == "U-Net":
        model = smp.Unet("resnet34", encoder_weights=None, classes=1)
    elif cfg["model"] == "PraNet":
        import_repo(cfg)
        from lib.PraNet_Res2Net import PraNet
        model = PraNet()
    else:
        import_repo(cfg)
        from lib.pvt import PolypPVT
        model = PolypPVT()
    model.load_state_dict(torch.load(Path(cfg["run_dir"]) / cfg["checkpoint"], map_location="cpu"))
    return model.cuda().eval()


def logits(name, model, x):
    out = model(x)
    if name == "PraNet":
        return out[3]  # lateral map 2, the map MyTest.py saves
    if name == "Polyp-PVT":
        return out[0] + out[1]  # P1 + P2, as in Test.py
    return out


@torch.no_grad()
def predict(cfg):
    model = load_model(cfg)
    val, out_dir = Path(cfg["val_dir"]), Path(cfg["pred_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for image_path in sorted((val / "images").iterdir()):
        image = Image.open(image_path).convert("RGB")
        width, height = Image.open(val / "masks" / f"{image_path.stem}.png").size
        x = TF.normalize(TF.to_tensor(image.resize((SIZE, SIZE), Image.BILINEAR)), MEAN, STD)[None].cuda()
        out = F.interpolate(logits(cfg["model"], model, x), size=(height, width), mode="bilinear", align_corners=False)
        prob = out.sigmoid()[0, 0].cpu().numpy()
        if cfg["model"] != "U-Net":
            prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)
        Image.fromarray((prob * 255).round().astype(np.uint8)).save(out_dir / f"{image_path.stem}.png")
    print(f"wrote {len(list(out_dir.iterdir()))} masks to {out_dir}")


if __name__ == "__main__":
    predict(yaml.safe_load(open(sys.argv[1])))
