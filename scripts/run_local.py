"""Run one baseline without Slurm (the PC): train, predict C5, score it and record the run.

Does what scripts/train_baseline.sbatch does. The log has the same lines as a Slurm log (job header,
nvidia-smi, StartTime/RunTime), so evaluate.py reads the GPU type, date and train time from it:
    python scripts/run_local.py configs/<run>.yaml --note "seed 0 ran on RTX 5000 Ada (HPC)"
    python scripts/run_local.py <scratch config> --no-save     (a check run: scored, not recorded)
"""
import argparse
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

MEDAI = Path("/l/users/mohamed.abdalla/medai")
REPO = Path(__file__).resolve().parents[1]


def environment(online):
    """The sbatch environment: caches under medai/cache, offline once the weights are downloaded."""
    cache = MEDAI / "cache"
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"), TORCH_HOME=str(cache / "torch"), HF_HOME=str(cache / "hf"),
               MPLCONFIGDIR=str(cache / "mpl"), PYTHONUNBUFFERED="1", PYTHONUTF8="1")
    if not online:
        env["HF_HUB_OFFLINE"] = "1"
    return env


def step(args, log, env):
    """Run one python step with its output appended to the log; stop the run if it fails."""
    with open(log, "a", encoding="utf-8") as out:
        subprocess.run([sys.executable, *args], stdout=out, stderr=subprocess.STDOUT, env=env, cwd=REPO, check=True)


def write(log, text):
    with open(log, "a", encoding="utf-8") as out:
        out.write(text + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--no-save", action="store_true", help="score only; do not add the run to the sheet")
    parser.add_argument("--note", help="passed to evaluate.py, added to the row's notes")
    parser.add_argument("--online", action="store_true", help="allow Hugging Face downloads")
    args = parser.parse_args()
    config = Path(args.config).resolve()
    start = datetime.now().replace(microsecond=0)
    job = start.strftime("%Y%m%d%H%M%S")  # a number, like a Slurm job id
    log = MEDAI / "outputs" / f"local-{yaml.safe_load(open(config))['name']}-{job}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = environment(args.online)
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    write(log, f"job {job} on {platform.node()}, config {config}, code {commit}")
    gpu = ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv"]
    write(log, subprocess.run(gpu, capture_output=True, text=True).stdout.strip())
    print(f"log: {log}")
    write(log, f"train start {start.isoformat()}")
    step(["-m", "medai.train", str(config)], log, env)
    write(log, f"train end {datetime.now().replace(microsecond=0).isoformat()}")
    step(["-m", "medai.predict", str(config)], log, env)
    # The train time recorded is the run time up to here: training plus C5 prediction, as on Slurm.
    seconds = int((datetime.now() - start).total_seconds())
    write(log, f"StartTime={start.isoformat()} RunTime={seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}")
    score = ["-m", "medai.evaluate", str(config)]
    if not args.no_save:
        score += ["--save", "--logs", str(log)] + (["--note", args.note] if args.note else [])
    step(score, log, env)
    print(log.read_text(encoding="utf-8").strip().splitlines()[-1])


if __name__ == "__main__":
    main()
