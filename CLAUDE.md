# Rules for this repository (humans and AI agents)

ENG2440 Capstone, Fall 2026: "What photometric clipping costs cross-centre polyp segmentation".
The proposal in `docs/proposal/proposal.tex` is final. Do not change the research question.

## Keep it small
- Few files. Before you add a file, check whether an existing file is the right place.
- No new top-level folders. The layout is fixed:
  `src/medai/` (code), `configs/` (one YAML per experiment), `scripts/` (Slurm and shell wrappers), `tests/` (pytest), `docs/` (proposal, reports, logs).
- No notebooks, no scratch files, no copies of files ("_v2", "_old", "_final") in the repo. Delete dead code; git keeps the history.
- Data, checkpoints, predictions and logs live OUTSIDE the repo, in `/l/users/mohamed.abdalla/medai/` (see `.gitignore`).

## Write very simple code
- Plain Python and PyTorch. Functions, not class hierarchies. No frameworks, registries or plugin systems.
- A function does one thing and is short (aim for under 40 lines). Names say what things are.
- One config file per experiment run. Every run sets and logs its seed.
- Copy nothing large from other repositories. If we must use outside code (for example the Polyp-PVT decoder), keep it in one file with its source, commit and licence at the top.
- Comments explain WHY, not what.

## Engineering practice
- Commit every working day: small commits, one change each, message in the imperative ("Add centre split loader"). Commits are authored by team members only: no AI co-author lines. AI use goes in docs/AI_USE_LOG.md instead.
- Run `pytest -q` before every commit. Tests cover the data split, the metrics and the clipping detector.
- Never commit data, weights, API keys or patient images. Images in the reports are ours or are from the public datasets, with credit.
- Add a line to `docs/EXPERIMENT_LOG.md` for every run: date, config, git commit, main number, output path.
- Every meaningful AI use goes into `docs/AI_USE_LOG.md`: who, what, what changed, how it was checked. This is required by the course.

## Scientific rules (from the proposal)
- Split by centre (PolypGen C1-C6). Train on four of C1-C5, select on the fifth, test ONCE on C6, after every choice is frozen. The code must refuse to evaluate on C6 unless it is run with `--final`.
- The baseline to beat is Polyp-PVT trained WITH BigAug image-quality and appearance transforms.
- Controls add one ingredient at a time, at matched size and compute.
- Confidence intervals: bootstrap over video sequences; three seeds for final comparisons.
- Never invent a number. If something fails, report it.

## Current stage: BASELINES ONLY (decided 2026-10-14)
- Only train existing baselines from their official GitHub code. Nothing else from the papers (no gate, no clipping study, no new method).
- Baselines: U-Net (ResNet-34 encoder, segmentation_models_pytorch), PraNet (github.com/DengPingFan/PraNet), Polyp-PVT (github.com/DengPingFan/Polyp-PVT). Use each repo's default settings.
- External repos are cloned OUTSIDE this repo, in `/l/users/mohamed.abdalla/medai/baselines/<name>`, at a recorded commit. We only add small adapter scripts here.
- Split for this stage: train = PolypGen single frames C1-C4; validation = single frames C5; C6 is not touched; sequences 1-15 are not used (their centre is unknown).
- Every model is scored by OUR one evaluation script (Dice and IoU per frame, mean over C5), so the numbers are comparable.
- ALL results go into one sheet: `docs/results.xlsx` (one row per run: date, model, repo@commit, train centres, val centre, seed, epochs, Dice, IoU, notes, output path).
