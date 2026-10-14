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
