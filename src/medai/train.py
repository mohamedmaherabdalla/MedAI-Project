"""Train one baseline on the train split, as set in its config (configs/<run>.yaml).

PraNet and Polyp-PVT run their own training code from the clone in /l/users/.../baselines/<name>;
we change only what stops it running here. U-Net is our short segmentation_models_pytorch script.
Run on a GPU node (see scripts/train_baseline.sbatch):
    PYTHONPATH=src python -m medai.train configs/<run>.yaml
"""
import argparse
import os
import random
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torchvision.transforms.functional as TF
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PVT_WEIGHTS = "/l/users/mohamed.abdalla/medai/baselines/weights/pvt_v2_b2.pth"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def import_repo(cfg):
    """Check the clone is at the recorded commit, put it on the path and work in the run folder."""
    head = subprocess.run(["git", "-C", cfg["repo"], "rev-parse", "HEAD"], capture_output=True, text=True)
    if head.stdout.strip() != cfg["commit"]:
        raise SystemExit(f"{cfg['repo']} is at {head.stdout.strip()}, config says {cfg['commit']}")
    subprocess.run(["git", "-C", cfg["repo"], "diff", "--quiet", "HEAD"], check=True)  # no local edits
    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)  # both repos read and write relative paths
    sys.path.insert(0, cfg["repo"])
    if cfg["model"] == "PraNet":
        import lib.PraNet_Res2Net as pranet
        import lib.Res2Net_v1b as res2net
        # res2net50_v1b_26w_4s loads from the authors' own disk (/media/nercms/...). res2net50_v1b in the
        # same file builds the same network and loads the official Res2Net-50 v1b ImageNet weights.
        pranet.res2net50_v1b_26w_4s = res2net.res2net50_v1b
    if cfg["model"] == "Polyp-PVT":
        link = run_dir / "pretrained_pth" / "pvt_v2_b2.pth"  # where PolypPVT() looks for its backbone
        link.parent.mkdir(exist_ok=True)
        if not link.exists():
            link.symlink_to(PVT_WEIGHTS)


def train_pranet(cfg):
    """Run the repo's MyTrain.py unchanged on our train folder; snapshots go to run_dir/snapshots."""
    import_repo(cfg)
    sys.argv = ["MyTrain.py", f"--train_path={cfg['train_dir']}"] + [f"--{k}={v}" for k, v in cfg["args"].items()]
    runpy.run_path(f"{cfg['repo']}/MyTrain.py", run_name="__main__")


def train_polyp_pvt(cfg):
    """Run the training loop of the repo's Train.py with its default settings on our train folder.

    Train.py scores five public test sets after every epoch and keeps the epoch that is best on its
    test folder. Those sets are not part of this stage, and picking the epoch on C5 would bias the C5
    score, so that per-epoch test is switched off and the last epoch is kept, as for PraNet and U-Net.
    """
    import_repo(cfg)
    import Train  # the repo's Train.py; importing it does not start training
    Train.test = lambda model, path, dataset: 0.0
    opt = argparse.Namespace(**cfg["args"], train_path=cfg["train_dir"], test_path=None,
                             train_save=f"{cfg['run_dir']}/model_pth/")
    names = ["CVC-300", "CVC-ClinicDB", "Kvasir", "CVC-ColonDB", "ETIS-LaribPolypDB", "test"]
    Train.opt, Train.best, Train.dict_plot = opt, 0, {name: [] for name in names}
    # From here on, the same steps as the __main__ block of Train.py.
    model = Train.PolypPVT().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay=1e-4)
    loader = Train.get_loader(f"{opt.train_path}/images/", f"{opt.train_path}/masks/", batchsize=opt.batchsize,
                              trainsize=opt.trainsize, augmentation=opt.augmentation)
    Train.total_step = len(loader)
    for epoch in range(1, opt.epoch):
        Train.adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        Train.train(loader, model, optimizer, epoch, opt.test_path)
    # Train.py saves every epoch (about 100 MB each); keep only the last one.
    for path in Path(opt.train_save).glob("*PolypPVT.pth"):
        if path.name != f"{opt.epoch - 1}PolypPVT.pth":
            path.unlink()


class Frames(Dataset):
    """Image and binary mask pairs of a split folder, resized to size x size."""

    def __init__(self, folder, size):
        self.images = sorted(Path(folder, "images").iterdir())
        self.size = size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        path, size = self.images[i], (self.size, self.size)
        image = Image.open(path).convert("RGB").resize(size, Image.BILINEAR)
        mask = Image.open(path.parent.parent / "masks" / f"{path.stem}.png").convert("L").resize(size, Image.NEAREST)
        return TF.normalize(TF.to_tensor(image), MEAN, STD), (TF.to_tensor(mask) > 0.5).float()


def train_unet(cfg):
    """U-Net with an ImageNet ResNet-34 encoder, Dice + BCE loss, Adam; the last epoch is saved."""
    a = cfg["args"]
    model = smp.Unet("resnet34", encoder_weights="imagenet", classes=1).cuda()
    loader = DataLoader(Frames(cfg["train_dir"], a["size"]), batch_size=a["batchsize"], shuffle=True, num_workers=4)
    dice_loss, bce_loss = smp.losses.DiceLoss("binary"), torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=a["lr"])
    for epoch in range(1, a["epochs"] + 1):
        model.train()
        total = 0.0
        for images, masks in loader:
            images, masks = images.cuda(), masks.cuda()
            logits = model(images)
            loss = dice_loss(logits, masks) + bce_loss(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"epoch {epoch}/{a['epochs']} loss {total / len(loader):.4f}", flush=True)
    checkpoint = Path(cfg["run_dir"]) / cfg["checkpoint"]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)


TRAIN = {"U-Net": train_unet, "PraNet": train_pranet, "Polyp-PVT": train_polyp_pvt}


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    print(f"config {sys.argv[1]}: {cfg}", flush=True)
    print(f"seed {cfg['seed']}", flush=True)
    set_seed(cfg["seed"])
    TRAIN[cfg["model"]](cfg)


if __name__ == "__main__":
    main()
