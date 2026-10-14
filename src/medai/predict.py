"""Predict the validation masks with one trained baseline, following the repo test scripts.

Each image is resized to 352 x 352; the sigmoid of the output is resized back to the mask size and
saved as an 8-bit PNG in pred_dir. The repo test scripts of PraNet, Polyp-PVT, EMCAD and SAM2-UNet
also stretch every map to [0, 1] before saving; we keep that step for them because it is their default.
nnU-Net predicts with its own command and default settings; its 0/1 label maps are saved as 0/255.
    PYTHONPATH=src python -m medai.predict configs/<run>.yaml
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import yaml
from PIL import Image

from medai.train import MEAN, STD, emcad_net_args, import_repo, set_nnunet_env

SIZE = 352  # train and test size of every baseline except nnU-Net


def load_model(cfg):
    if cfg["model"] == "U-Net":
        model = smp.Unet("resnet34", encoder_weights=None, classes=1)
    else:
        import_repo(cfg)
    if cfg["model"] == "PraNet":
        from lib.PraNet_Res2Net import PraNet
        model = PraNet()
    if cfg["model"] == "Polyp-PVT":
        from lib.pvt import PolypPVT
        model = PolypPVT()
    if cfg["model"] == "EMCAD":
        from lib.networks import EMCADNet
        model = EMCADNet(**emcad_net_args(cfg["args"]), pretrain=False)  # as in test_polyp.py
    if cfg["model"] == "SAM2-UNet":
        from SAM2UNet import SAM2UNet
        model = SAM2UNet()  # as in test.py: the trained checkpoint holds the encoder too
    model.load_state_dict(torch.load(Path(cfg["run_dir"]) / cfg["checkpoint"], map_location="cpu"))
    return model.cuda().eval()


def logits(name, model, x):
    out = model(x)
    if name == "PraNet":
        return out[3]  # lateral map 2, the map MyTest.py saves
    if name == "Polyp-PVT":
        return out[0] + out[1]  # P1 + P2, as in Test.py
    if name == "EMCAD":
        return out[-1]  # as in test_polyp.py
    if name == "SAM2-UNet":
        return out[0]  # as in test.py
    return out


def val_inputs(cfg):
    """Yield (frame, 1 x 3 x 352 x 352 input) for every validation image, preprocessed as for training."""
    val = Path(cfg["val_dir"])
    if cfg["model"] == "EMCAD":
        # EMCAD trains on OpenCV/albumentations resizing, so its test inputs come from its own loader.
        from utils.dataloader_polyp import PolypDataset
        data = PolypDataset(f"{val}/images/", f"{val}/masks/", SIZE, augmentation=False, split="test")
        for i in range(len(data)):
            yield Path(data.images[i]).stem, data[i][0][None]
        return
    for path in sorted((val / "images").iterdir()):
        image = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        yield path.stem, TF.normalize(TF.to_tensor(image), MEAN, STD)[None]


@torch.no_grad()
def predict(cfg):
    model = load_model(cfg)
    masks, out_dir = Path(cfg["val_dir"]) / "masks", Path(cfg["pred_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for frame, x in val_inputs(cfg):
        width, height = Image.open(masks / f"{frame}.png").size
        out = F.interpolate(logits(cfg["model"], model, x.cuda()), size=(height, width), mode="bilinear",
                            align_corners=False)
        prob = out.sigmoid()[0, 0].cpu().numpy()
        if cfg["model"] != "U-Net":
            prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)
        Image.fromarray((prob * 255).round().astype(np.uint8)).save(out_dir / f"{frame}.png")
    print(f"wrote {len(list(out_dir.iterdir()))} masks to {out_dir}")


def predict_nnunet(cfg):
    bin_dir = set_nnunet_env(cfg)
    labels, out_dir = Path(cfg["run_dir"]) / "nnunet_labels_c5", Path(cfg["pred_dir"])
    subprocess.run([bin_dir / "nnUNetv2_predict", "-i", f"{cfg['nnunet_raw']}/{cfg['dataset_name']}/imagesTs",
                    "-o", labels, "-d", str(cfg["dataset_id"]), "-c", "2d", "-f", "all"], check=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for label in sorted(labels.glob("*.png")):
        Image.fromarray(np.asarray(Image.open(label)).astype(np.uint8) * 255).save(out_dir / label.name)
    print(f"wrote {len(list(out_dir.iterdir()))} masks to {out_dir}")


if __name__ == "__main__":
    config = yaml.safe_load(open(sys.argv[1]))
    if config["model"] == "nnU-Net":
        predict_nnunet(config)
    else:
        predict(config)
