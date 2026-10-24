"""Train SAC+HER (or TQC+HER fallback) on a Project V experiment.

    python -m projectv.train.train --exp configs/exp/push_dr.yaml --seed 1 --resume auto

Outputs (run = {exp}_s{seed}[_{tag}]):
    {out}/ckpt/{run}/step_{n}.zip, replay.pkl     checkpoints (resumable)
    {out}/train_logs/{run}/eval.csv               nominal success per checkpoint
    {out}/train_logs/{run}/run_meta.json          git SHA, config hash, versions, wall-clock
    {out}/ckpt/{run}_final.zip                    final model
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv

from projectv.envs.dr_wrapper import fixed_spec
from projectv.envs.make_env import make_env
from projectv.train.callbacks import CheckpointEvalCallback, latest_checkpoint, save_replay
from projectv.utils.config import load_exp, load_yaml
from projectv.utils.io import write_run_meta
from projectv.utils.seeding import set_global_seed


def algo_class(name: str):
    if name == "sac":
        return SAC
    if name == "tqc":
        from sb3_contrib import TQC

        return TQC
    raise ValueError(name)


def build_model(algo: dict, venv, seed: int, tb_dir: str | None):
    return algo_class(algo["algo"])(
        algo["policy"],
        venv,
        learning_rate=algo["learning_rate"],
        buffer_size=algo["buffer_size"],
        learning_starts=algo["learning_starts"],
        batch_size=algo["batch_size"],
        tau=algo["tau"],
        gamma=algo["gamma"],
        ent_coef=algo["ent_coef"],
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(algo["her"]),
        policy_kwargs=dict(net_arch=list(algo["net_arch"]), n_critics=algo["n_critics"]),
        tensorboard_log=tb_dir,
        seed=seed,
        device="cpu",
        verbose=0,
    )


def run_name(exp_name: str, seed: int, tag: str | None) -> str:
    return f"{exp_name}_s{seed}" + (f"_{tag}" if tag else "")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--total-steps", type=int, default=None, help="override algo total_steps")
    p.add_argument("--algo", default=None, help="override algo config path (fallback ladder)")
    p.add_argument("--tag", default=None)
    p.add_argument("--resume", choices=["auto", "none"], default="auto")
    p.add_argument("--out", default="results")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--max-wall-hours", type=float, default=None, help="clean stop + save")
    p.add_argument("--ckpt-every", type=int, default=50_000)
    p.add_argument("--replay-every", type=int, default=200_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    t_start = time.time()
    torch.set_num_threads(args.threads)
    set_global_seed(args.seed)

    exp = load_exp(args.exp)
    if args.algo:
        exp["algo"], exp["algo_cfg"] = args.algo, load_yaml(args.algo)
    algo = exp["algo_cfg"]
    total = args.total_steps or algo["total_steps"]
    name = run_name(exp["name"], args.seed, args.tag)
    out = Path(args.out)
    ckpt_dir, log_dir = out / "ckpt" / name, out / "train_logs" / name
    log_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = not args.no_wandb and bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        import wandb

        wandb.init(
            project="project-v",
            name=name,
            config=exp,
            sync_tensorboard=True,
            resume="allow",
            id=name,
        )
    tb_dir = str(out / "tb" / name)

    venv = DummyVecEnv([lambda: make_env(exp, seed=args.seed)])
    nominal = fixed_spec({k: v["value"] for k, v in load_yaml("configs/dr/none.yaml").items()})
    eval_env = make_env(exp, seed=args.seed + 500, dr_spec=nominal)

    found = latest_checkpoint(ckpt_dir) if args.resume == "auto" else None
    if found:
        step, path = found
        model = algo_class(algo["algo"]).load(path, env=venv, device="cpu")
        replay = ckpt_dir / "replay.pkl"
        if replay.exists():
            model.load_replay_buffer(replay)
        rb_step = (ckpt_dir / "replay_step.txt").read_text() if replay.exists() else "none"
        print(
            f"[resume] {path} (step {model.num_timesteps}), replay from step {rb_step}", flush=True
        )
    else:
        model = build_model(algo, venv, args.seed, tb_dir)
        print(f"[start] {name}: {algo['algo']} total={total}", flush=True)

    cb = CheckpointEvalCallback(
        ckpt_dir,
        log_dir / "eval.csv",
        eval_env,
        ckpt_every=args.ckpt_every,
        replay_every=args.replay_every,
        eval_episodes=args.eval_episodes,
        max_wall_s=args.max_wall_hours * 3600 if args.max_wall_hours else None,
    )
    remaining = total - model.num_timesteps
    step0 = model.num_timesteps
    if remaining > 0:
        model.learn(remaining, callback=cb, reset_num_timesteps=False, tb_log_name="run")
    finished = model.num_timesteps >= total
    if finished:
        model.save(out / "ckpt" / f"{name}_final.zip")
    elif not cb.stopped_on_wall_clock:
        save_replay(model, ckpt_dir)

    wall = time.time() - t_start
    write_run_meta(
        log_dir,
        exp,
        sys.argv,
        t_start,
        run=name,
        seed=args.seed,
        total_steps=total,
        num_timesteps=model.num_timesteps,
        finished=finished,
        stopped_on_wall_clock=cb.stopped_on_wall_clock,
        session_fps=round((model.num_timesteps - step0) / max(wall, 1e-9), 1),
    )
    print(
        f"[done] {name} steps={model.num_timesteps} finished={finished} wall={wall:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
