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
from scipy.ndimage import gaussian_filter
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
            link.symlink_to(Path(PVT_WEIGHTS).resolve())  # Windows needs the drive in the target


def train_pranet(cfg):
    """Run the repo's MyTrain.py unchanged on our train folder; snapshots go to run_dir/snapshots."""
    import_repo(cfg)
    sys.argv = ["MyTrain.py", f"--train_path={cfg['train_dir']}"] + [f"--{k}={v}" for k, v in cfg["args"].items()]
    runpy.run_path(f"{cfg['repo']}/MyTrain.py", run_name="__main__")


def bigaug(image, p):
    """BigAug image-quality then appearance transforms (Zhang et al., IEEE TMI 2020) on one RGB image.

    Each transform runs with probability p["probability"], in the paper's order, on intensities in [0, 1]
    (the paper's range); p holds the ranges from the config. The paper's spatial transforms are not used.
    """
    blur = lambda x, std: gaussian_filter(x, sigma=(std, std, 0))  # within each colour channel
    u = lambda key: np.random.uniform(*p[key])

    def sharpen(x):  # unsharp masking
        blurred = blur(x, u("sharpen_blur_std"))
        return blurred + u("sharpen_strength") * (blurred - blur(blurred, p["sharpen_filter_std"]))

    steps = [sharpen,
             lambda x: blur(x, u("blur_std")),
             lambda x: x + np.random.normal(0, u("noise_std"), x.shape),
             lambda x: x + u("brightness_shift"),
             lambda x: x ** u("gamma"),  # contrast
             lambda x: x * (1 + u("intensity_scale")) + u("intensity_shift")]  # intensity perturbation
    x = np.asarray(image, dtype=np.float32) / 255
    for step in steps:
        if np.random.rand() < p["probability"]:
            # A transform must not leave the [0, 1] range: gamma is undefined below 0 and a frame
            # cannot be brighter than white.
            x = np.clip(step(x), 0, 1)
    return Image.fromarray((x * 255).round().astype(np.uint8))


def add_bigaug_to_polyp_pvt_loader(p):
    """Make the repo's training dataset apply bigaug to every image it loads."""
    import utils.dataloader as loader  # the module whose PolypDataset Train.get_loader builds
    load_rgb = loader.PolypDataset.rgb_loader

    def rgb_loader(dataset, path):
        # The repo's own resize, done first so that blur and noise act at the 352 px training size;
        # its Resize transform then leaves the image unchanged.
        size = (dataset.trainsize, dataset.trainsize)
        return bigaug(load_rgb(dataset, path).resize(size, Image.BILINEAR), p)

    loader.PolypDataset.rgb_loader = rgb_loader


def train_polyp_pvt(cfg):
    """Run the training loop of the repo's Train.py with its default settings on our train folder.

    Train.py scores five public test sets after every epoch and keeps the epoch that is best on its
    test folder. Those sets are not part of this stage, and picking the epoch on C5 would bias the C5
    score, so that per-epoch test is switched off and the last epoch is kept, as for PraNet and U-Net.
    A config with a "bigaug" entry also trains with the BigAug transforms.
    """
    import_repo(cfg)
    import Train  # the repo's Train.py; importing it does not start training
    Train.test = lambda model, path, dataset: 0.0
    if "bigaug" in cfg:
        add_bigaug_to_polyp_pvt_loader(cfg["bigaug"])
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


def emcad_net_args(a):
    """EMCADNet arguments, derived from the train_polyp.py options as its __main__ block does."""
    return dict(num_classes=1, kernel_sizes=a["kernel_sizes"], expansion_factor=a["expansion_factor"],
                dw_parallel=not a["no_dw_parallel"], add=not a["concatenation"], lgag_ks=a["lgag_ks"],
                activation=a["activation_mscb"], encoder=a["encoder"])


def train_emcad(cfg):
    """Run the training loop of the repo's train_polyp.py once, with its default settings.

    train_polyp.py trains five runs in a row and keeps the epoch with the best Dice on its own val folder.
    We train one run (seed 0); C5 is our only held-out centre, so the per-epoch test is switched off and
    the last epoch is kept, as for Polyp-PVT.
    """
    import_repo(cfg)
    import train_polyp as T  # the repo's train_polyp.py; importing it does not start training
    T.test = lambda model, path, dataset, opt: (0.0, 0.0, 0)
    opt = argparse.Namespace(**cfg["args"], train_path=cfg["train_dir"], test_path=None,
                             train_save=f"{cfg['run_dir']}/model_pth/")
    T.dict_plot, T.best, T.test_dice_at_best_val, T.total_train_time = {"val": [], "test": []}, 0.0, 0.0, 0
    # From here on, the same steps as one run of the __main__ block of train_polyp.py.
    model = T.EMCADNet(**emcad_net_args(cfg["args"]), pretrain=True, pretrained_dir=opt.pretrained_dir).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay=1e-4)
    scheduler = T.CosineAnnealingLR(optimizer, T_max=opt.epoch, eta_min=1e-6)
    loader = T.get_loader(image_root=f"{opt.train_path}/images/", gt_root=f"{opt.train_path}/masks/",
                          batchsize=opt.batchsize, trainsize=opt.img_size, shuffle=True,
                          augmentation=opt.augmentation, split="train", color_image=opt.color_image)
    for epoch in range(1, opt.epoch + 1):
        T.adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        T.train(loader, model, optimizer, epoch, opt, "EMCAD")
        scheduler.step()


def train_sam2_unet(cfg):
    """Run the repo's train.py unchanged (its own seed call is commented out; main() sets ours)."""
    import_repo(cfg)
    train = cfg["train_dir"]
    sys.argv = ["train.py", f"--hiera_path={cfg['weights']}", f"--train_image_path={train}/images/",
                f"--train_mask_path={train}/masks/", f"--save_path={cfg['run_dir']}/snapshots"]
    sys.argv += [f"--{k}={v}" for k, v in cfg["args"].items()]
    runpy.run_path(f"{cfg['repo']}/train.py", run_name="__main__")


def set_nnunet_env(cfg):
    """Point nnU-Net at its folders. Must run before nnunetv2 is imported: it reads them at import."""
    os.environ.update(nnUNet_raw=cfg["nnunet_raw"], nnUNet_preprocessed=cfg["nnunet_preprocessed"],
                      nnUNet_results=f"{cfg['run_dir']}/nnUNet_results",
                      nnUNet_n_proc_DA=str(cfg["loader_processes"]))  # one loader process per allocated CPU
    return Path(sys.executable).parent  # the nnUNetv2_* commands live next to the venv python


def train_nnunet(cfg):
    """Plan, preprocess and train nnU-Net v2 (default plans and trainer, 2d, fold 'all').

    Running this again continues from the latest checkpoint, so a job that hits its time limit is
    simply resubmitted.
    """
    import_repo(cfg)  # the venv has nnunetv2 installed from this clone (pip install -e)
    bin_dir = set_nnunet_env(cfg)
    from nnunetv2.run.run_training import run_training
    dataset = str(cfg["dataset_id"])
    if not Path(cfg["nnunet_preprocessed"], cfg["dataset_name"], "nnUNetPlans_2d").exists():
        subprocess.run([bin_dir / "nnUNetv2_plan_and_preprocess", "-d", dataset, "-c", "2d",
                        "-np", str(cfg["loader_processes"]), "--verify_dataset_integrity"], check=True)
    fold_dir = Path(cfg["run_dir"], cfg["checkpoint"]).parent
    if (fold_dir / "checkpoint_final.pth").exists():
        return
    run_training(dataset, "2d", "all", continue_training=(fold_dir / "checkpoint_latest.pth").exists())


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


TRAIN = {"U-Net": train_unet, "PraNet": train_pranet, "Polyp-PVT": train_polyp_pvt,
         "EMCAD": train_emcad, "SAM2-UNet": train_sam2_unet, "nnU-Net": train_nnunet}


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    print(f"config {sys.argv[1]}: {cfg}", flush=True)
    print(f"seed {cfg['seed']}", flush=True)
    set_seed(cfg["seed"])
    TRAIN[cfg["model"]](cfg)


if __name__ == "__main__":
    main()
