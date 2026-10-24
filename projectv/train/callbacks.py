"""Training callbacks: periodic checkpoints (+ replay buffer), nominal-success evaluation,
and a wall-clock guard that stops cleanly before a Kaggle session limit.

Checkpoint layout (CLAUDE.md §7):
    {ckpt_dir}/step_{n}.zip      every `ckpt_every` steps (last `keep_last` kept, plus milestones)
    {ckpt_dir}/replay.pkl        every `replay_every` steps and at a clean stop
    {ckpt_dir}/replay_step.txt   env step at which replay.pkl was written
"""

from __future__ import annotations

import csv
import re
import time
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

STEP_RE = re.compile(r"step_(\d+)\.zip$")


def list_checkpoints(ckpt_dir: str | Path) -> list[tuple[int, Path]]:
    d = Path(ckpt_dir)
    if not d.exists():
        return []
    found = [(int(m.group(1)), p) for p in d.iterdir() if (m := STEP_RE.search(p.name))]
    return sorted(found)


def latest_checkpoint(ckpt_dir: str | Path) -> tuple[int, Path] | None:
    ckpts = list_checkpoints(ckpt_dir)
    return ckpts[-1] if ckpts else None


def save_replay(model, ckpt_dir: Path) -> None:
    tmp = ckpt_dir / "replay_tmp.pkl"
    model.save_replay_buffer(tmp)
    tmp.replace(ckpt_dir / "replay.pkl")
    (ckpt_dir / "replay_step.txt").write_text(str(model.num_timesteps))


def evaluate_success(model, env, n_episodes: int, seed0: int) -> float:
    succ = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        info = {}
        for _ in range(50):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                break
        succ.append(float(info.get("is_success", 0.0)))
    return float(np.mean(succ))


class CheckpointEvalCallback(BaseCallback):
    def __init__(
        self,
        ckpt_dir: Path,
        log_csv: Path,
        eval_env,
        ckpt_every: int = 50_000,
        replay_every: int = 200_000,
        keep_last: int = 2,
        milestone_every: int = 250_000,
        eval_episodes: int = 20,
        eval_seed0: int = 900_000,
        max_wall_s: float | None = None,
    ):
        super().__init__()
        self.ckpt_dir, self.log_csv, self.eval_env = Path(ckpt_dir), Path(log_csv), eval_env
        self.ckpt_every, self.replay_every = ckpt_every, replay_every
        self.keep_last, self.milestone_every = keep_last, milestone_every
        self.eval_episodes, self.eval_seed0 = eval_episodes, eval_seed0
        self.max_wall_s = max_wall_s
        self.t0 = time.time()
        self.step0 = 0
        self.stopped_on_wall_clock = False

    def _on_training_start(self) -> None:
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_csv.parent.mkdir(parents=True, exist_ok=True)
        self.t0, self.step0 = time.time(), self.model.num_timesteps

    def _on_step(self) -> bool:
        n = self.model.num_timesteps
        if n % self.ckpt_every == 0:
            self._checkpoint(n)
            if n % self.replay_every == 0:
                save_replay(self.model, self.ckpt_dir)
        if self.max_wall_s is not None and time.time() - self.t0 > self.max_wall_s:
            self._checkpoint(n)
            save_replay(self.model, self.ckpt_dir)
            self.stopped_on_wall_clock = True
            return False
        return True

    def _checkpoint(self, n: int) -> None:
        self.model.save(self.ckpt_dir / f"step_{n}.zip")
        self._prune()
        succ = evaluate_success(self.model, self.eval_env, self.eval_episodes, self.eval_seed0)
        train_succ = (
            float(np.mean(self.model.ep_success_buffer)) if self.model.ep_success_buffer else np.nan
        )
        elapsed = time.time() - self.t0
        row = {
            "step": n,
            "eval_success_nominal": succ,
            "train_success_last100": train_succ,
            "fps_session": round((n - self.step0) / max(elapsed, 1e-9), 1),
            "session_wall_s": round(elapsed, 1),
            "unix_time": round(time.time(), 1),
        }
        new = not self.log_csv.exists()
        with open(self.log_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"[ckpt] {row}", flush=True)

    def _prune(self) -> None:
        ckpts = list_checkpoints(self.ckpt_dir)
        for step, path in ckpts[: -self.keep_last]:
            if step % self.milestone_every != 0:
                path.unlink(missing_ok=True)
