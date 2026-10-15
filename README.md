# What photometric clipping costs cross-centre polyp segmentation

ENG2440 Capstone (Fall 2026). Mohamed Abdalla, Brandon Adebayo, Mohamed Farrag.

Polyp segmentation models lose accuracy at new hospitals. We measure how much of that loss comes from clipped (over-exposed) pixels, and test whether enhanced views (CLAHE, gamma) that teach each other only where it helps reduce it. Proposal: `docs/proposal/proposal.tex`.

## Layout
| Path | What |
|---|---|
| `src/medai/` | Code: data, model, training, evaluation |
| `configs/` | One YAML per experiment |
| `scripts/` | Slurm and shell wrappers |
| `tests/` | pytest |
| `docs/` | Proposal, course reports, experiment log, AI-use log |

Data and outputs are not in the repository. They live in `/l/users/mohamed.abdalla/medai/`.

## Reproduce from scratch
All paths are under `/l/users/mohamed.abdalla/medai/`; on a new machine, create the same path (on Windows: `C:\l\users\mohamed.abdalla\medai`, run from drive C:).
1. Python 3.12 venv in `medai/venv`, then `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128`.
2. Data: make a Synapse token (scopes "view" and "download"), store it with `synapse config`, then `python scripts/download_polypgen.py`.
3. Split: `PYTHONPATH=src python -m medai.split` (train = C1-C4 single frames, val = C5; C6 is never touched). On Windows, turn on Developer Mode first: the split folders are symlinks.
4. Baselines and weights: `python scripts/setup_baselines.py`, then `pip install -e medai/baselines/nnUNet`.
5. One run: `scripts/train_baseline.sbatch configs/<run>.yaml` with Slurm. Without Slurm, run its three steps: `python -m medai.train`, `python -m medai.predict`, `python -m medai.evaluate --save` on the same config.
6. Check: `pytest`, then `python -m medai.evaluate configs/<run>.yaml` on a finished run must give the Dice in `docs/results.xlsx`.
