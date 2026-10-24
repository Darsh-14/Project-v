"""Measure end-to-end training throughput (env step + 1 gradient step per env step) for
candidate algo settings, and append to a CSV. Used for the Day-1 pilot / compute projection.

    python scripts/bench_fps.py --exp configs/exp/push_dr.yaml --device cpu --threads 1 \
        --variants 2048x512x3,256x256x3 --out results/pilot/fps.csv
"""

from __future__ import annotations

import argparse
import csv
import platform
import time
from pathlib import Path

import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from projectv.envs.make_env import make_env
from projectv.train.train import build_model
from projectv.utils.config import load_exp


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--variants", default="2048x512x3", help="BATCHxWIDTHxDEPTH,...")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--timed", type=int, default=300)
    p.add_argument("--host", default=platform.node())
    p.add_argument("--out", default="results/pilot/fps.csv")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    exp = load_exp(args.exp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for v in args.variants.split(","):
        batch, width, depth = (int(x) for x in v.split("x"))
        algo = dict(
            exp["algo_cfg"], batch_size=batch, net_arch=[width] * depth, learning_starts=args.warmup
        )
        model = build_model(algo, DummyVecEnv([lambda: make_env(exp, seed=0)]), 0, None, args.device)
        model.learn(args.warmup)
        t = time.time()
        model.learn(args.timed, reset_num_timesteps=False)
        fps = args.timed / (time.time() - t)
        row = {
            "host": args.host,
            "exp": exp["name"],
            "algo": algo["algo"],
            "batch": batch,
            "net": f"{width}x{depth}",
            "device": args.device,
            "threads": args.threads,
            "env_steps_per_s": round(fps, 2),
            "hours_per_1M": round(1e6 / fps / 3600, 1),
        }
        new = not out.exists()
        with open(out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print(row, flush=True)


if __name__ == "__main__":
    main()
