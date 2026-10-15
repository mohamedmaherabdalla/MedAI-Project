"""Clone the baseline repos at the commits in configs/ and download their pretrained weights.

Rebuilds /l/users/mohamed.abdalla/medai/baselines on a new machine. The clones are used as
released: we changed no file in them. PraNet downloads its Res2Net weights through torch.hub,
and U-Net its ResNet-34 weights from the Hugging Face hub, at the first run (online).
Run:
    python scripts/setup_baselines.py
"""
import hashlib
import subprocess
import urllib.request
from pathlib import Path

import yaml

BASELINES = Path("/l/users/mohamed.abdalla/medai/baselines")
CONFIGS = Path(__file__).resolve().parent.parent / "configs"
# file name: (official URL, MD5 of the file we trained with)
WEIGHTS = {
    "pvt_v2_b2.pth": ("https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth",
                      "efd6f8e5375cf869ab44484a9e89ac39"),
    "sam2_hiera_large.pt": ("https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
                            "08083462423be3260cd6a5eef94dc01c"),
}


def clone(repo_url, commit, target):
    if not target.exists():
        subprocess.run(["git", "clone", repo_url, str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "-q", commit], check=True)


def md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    for config in sorted(CONFIGS.glob("*_seed0.yaml")):
        cfg = yaml.safe_load(config.read_text())
        if "repo" in cfg:
            clone(cfg["repo_url"], cfg["commit"], BASELINES / Path(cfg["repo"]).name)
    (BASELINES / "weights").mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in WEIGHTS.items():
        path = BASELINES / "weights" / name
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        if md5(path) != expected:
            raise ValueError(f"{path}: MD5 differs from the file we trained with")
        print(f"{path}: OK")


if __name__ == "__main__":
    main()
