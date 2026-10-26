"""Render the 3 rollout GIFs of mvp.md A9 to `results/gifs/`.

    python scripts/make_gifs.py                            # results/ -> results/gifs/
    python scripts/make_gifs.py --check-render out.gif     # render the analytic controller only

1. push_dr_nominal_success.gif: a nominal-dynamics push_dr episode that the step log records
   as a success (needs checkpoint + steplogs/push_dr_s{seed}/cell00.npz).
2. push_dr_heldout_failure_asm.gif: a held-out push_dr failure (test split). From the first step
   where the episode's EWMA ASM score exceeds the calib threshold, frames get a red border and
   an "ASM ALARM" label (needs checkpoint, step log of that cell, the frozen heldout.yaml, and
   results/c3/c3_push_dr_episode_scores.csv).
3. reach_shortcut_failure.gif: reach_shortcut_v2 (the variant that learned the shortcut) with the hint replaced by a random point
   (needs the reach_shortcut_v2 checkpoint); the first failing episode of 20 seeded ones.

Every GIF is skipped with a message if an input is missing. Episodes are re-simulated from their
logged reset seed with the deterministic policy; the outcome printed on the last frame is the
outcome of the rendered rollout itself, and a mismatch with the log is reported.

Rendering backend: MuJoCo reads MUJOCO_GL. On Linux, if unset, this script sets `egl`
(GPU sessions); use `MUJOCO_GL=osmesa` on CPU-only sessions. On Windows/macOS the default is used.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

Log = Callable[[str], None]
EPISODE_STEPS = 50
GIF_SIZE = 320
RED = (227, 73, 72)


# ---------------------------------------------------------------------------- frame drawing


def annotate(frame: np.ndarray, lines: list[str], alarm: bool = False) -> np.ndarray:
    """Downscale to GIF_SIZE, write caption lines top-left, red border when alarm is on."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(frame).resize((GIF_SIZE, GIF_SIZE), Image.BILINEAR)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    y = 12
    for line in lines:
        box = draw.textbbox((14, y), line, font=font)
        draw.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=(0, 0, 0))
        draw.text((14, y), line, fill=(255, 255, 255), font=font)
        y = box[3] + 6
    if alarm:
        w = 8
        draw.rectangle((0, 0, GIF_SIZE - 1, GIF_SIZE - 1), outline=RED, width=w)
        label = "ASM ALARM"
        box = draw.textbbox((0, 0), label, font=font)
        tw = box[2] - box[0]
        x0, y0 = GIF_SIZE - tw - 18, GIF_SIZE - 30
        draw.rectangle((x0 - 4, y0 - 3, x0 + tw + 4, y0 + 17), fill=RED)
        draw.text((x0, y0), label, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def write_gif(path: Path, frames: list[np.ndarray], fps: int = 10) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, duration=1.0 / fps, loop=0)


# ---------------------------------------------------------------------------- rollouts


def render_rollout(
    act: Callable[[dict], np.ndarray],
    exp: dict,
    dr_spec: dict,
    ep_seed: int,
    shortcut_mode: str = "intact",
    steps: int = EPISODE_STEPS,
) -> tuple[list[np.ndarray], bool]:
    """Same env stack and reset seed as eval.rollout.run_episodes, plus rgb frames.
    Returns (frames before each action + final frame, final is_success)."""
    from projectv.envs.make_env import make_env

    env = make_env(
        exp, seed=ep_seed, dr_spec=dr_spec, shortcut_mode=shortcut_mode, render_mode="rgb_array"
    )
    obs, _ = env.reset(seed=int(ep_seed))
    frames, success = [], False
    for _ in range(steps):
        frames.append(env.render())
        obs, _, term, trunc, info = env.step(act(obs))
        success = bool(info["is_success"])
        if term or trunc:
            break
    frames.append(env.render())
    env.close()
    return frames, success


def sb3_actor(model) -> Callable[[dict], np.ndarray]:
    """Deterministic action tanh(mean), computed exactly as in eval.rollout.run_episodes."""
    import torch

    from projectv.xai.policy_fn import SB3Policy, flatten_obs

    pol = SB3Policy(model)

    def act(obs: dict) -> np.ndarray:
        x = torch.as_tensor(flatten_obs(obs)[None], dtype=torch.float64)
        with torch.no_grad():
            mean, _ = pol.dist(x)
        return torch.tanh(mean)[0].numpy().astype(np.float32)

    return act


def load_policy(path: Path, exp: dict):
    """projectv.eval.rollout.load_model, retried with an env: SB3 refuses to load a
    HerReplayBuffer model without one ("You must pass an environment ...")."""
    from projectv.eval.rollout import load_model

    try:
        return load_model(path)
    except AssertionError:
        from projectv.envs.make_env import make_env

        env = make_env(exp, seed=0)
        try:
            from stable_baselines3 import SAC

            return SAC.load(path, env=env, device="cpu")
        except Exception:
            from sb3_contrib import TQC

            return TQC.load(path, env=env, device="cpu")


def _ckpt(root: Path, exp: str, seed: int) -> Path | None:
    from projectv.eval.rollout import resolve_checkpoint

    try:
        return resolve_checkpoint(exp, seed, root)
    except FileNotFoundError:
        return None


def _captioned(
    frames: list[np.ndarray], header: str, success: bool, alarm_from: int | None = None
) -> list[np.ndarray]:
    out = []
    for t, f in enumerate(frames):
        lines = [*header.split(" | "), f"t = {min(t, EPISODE_STEPS)}/{EPISODE_STEPS}"]
        if t == len(frames) - 1:
            lines.append("outcome: SUCCESS" if success else "outcome: FAILURE")
        alarm = alarm_from is not None and t >= alarm_from
        out.append(annotate(f, lines, alarm))
    # hold the last frame for 1 s
    return out + [out[-1]] * 10


# ---------------------------------------------------------------------------- the 3 GIFs


def gif_nominal_success(root: Path, out: Path, log: Log, seed: int = 0, n_try: int = 5) -> None:
    from projectv.eval.cells import NOMINAL
    from projectv.eval.steplog import load_steplog
    from projectv.utils.config import load_exp

    name = "push_dr_nominal_success.gif"
    ckpt = _ckpt(root, "push_dr", seed)
    sl = root / "steplogs" / f"push_dr_s{seed}" / "cell00.npz"
    if ckpt is None or not sl.exists():
        return log(f"skip {name}: checkpoint or {sl} missing")
    rec = load_steplog(sl)
    ok = np.flatnonzero(rec["success"])
    if not len(ok):
        return log(f"skip {name}: no successful nominal episode in {sl}")
    exp = load_exp("configs/exp/push_dr.yaml")
    act = sb3_actor(load_policy(ckpt, exp))
    for ep in ok[:n_try]:
        sd = int(rec["ep_seed"][ep])
        frames, success = render_rollout(act, exp, NOMINAL.dr_spec(), sd)
        if not success:
            log(f"  warning: re-simulated episode {sd} did not reproduce the logged success; next")
            continue
        header = f"push_dr s{seed} | nominal dynamics | ep_seed {sd}"
        write_gif(out / name, _captioned(frames, header, success))
        return log(f"wrote {out / name} (ep_seed {sd})")
    log(f"skip {name}: no logged success reproduced in {min(len(ok), n_try)} tries")


def alarm_step(row: pd.Series) -> int | None:
    """First step whose EWMA ASM score exceeds the calib threshold: an explicit
    `detect_step`/`alarm_step` column, else per-step `e_00`..`e_49` columns vs `threshold`."""
    for c in ("detect_step", "alarm_step"):
        if c in row.index:
            return None if pd.isna(row[c]) else int(row[c])
    ecols = sorted(
        (c for c in row.index if re.fullmatch(r"e_?\d+", c)),
        key=lambda c: int(re.sub(r"\D", "", c)),
    )
    if ecols and "threshold" in row.index:
        above = np.flatnonzero(row[ecols].to_numpy(float) > float(row["threshold"]))
        return int(above[0]) if len(above) else None
    raise KeyError("episode scores need detect_step/alarm_step or e_* + threshold columns")


def gif_heldout_failure(
    root: Path, out: Path, log: Log, seed: int = 0, heldout: str | Path | None = None
) -> None:
    from projectv.eval.cells import HELDOUT_PATH, assert_frozen, load_cells
    from projectv.eval.steplog import load_steplog, steplog_path
    from projectv.utils.config import load_exp

    name = "push_dr_heldout_failure_asm.gif"
    heldout = Path(heldout or HELDOUT_PATH)
    ckpt = _ckpt(root, "push_dr", seed)
    scores_p = root / "c3" / "c3_push_dr_episode_scores.csv"
    if ckpt is None or not scores_p.exists() or not heldout.exists():
        return log(f"skip {name}: checkpoint, {scores_p} or {heldout} missing")
    try:
        assert_frozen(heldout, root / "heldout_hash.txt")
    except (AssertionError, FileNotFoundError) as e:
        return log(f"skip {name}: held-out grid not frozen/consistent ({e})")
    cells = {c.id: c for c in load_cells(heldout, check_hash=False)}
    df = pd.read_csv(scores_p)
    mcol = next((c for c in ("monitor", "method", "signal") if c in df.columns), None)
    sel = df[(df[mcol].astype(str).str.lower() == "asm")] if mcol else df
    if "seed" in sel.columns:
        sel = sel[sel["seed"] == seed]
    if "split" in sel.columns:
        sel = sel[sel["split"].astype(str) == "test"]
    failed = sel["failed"].astype(bool) if "failed" in sel.columns else ~sel["success"].astype(bool)
    sel = sel[failed & (sel["cell_id"] != 0)]
    sel = sel.assign(_alarm=[alarm_step(r) for _, r in sel.iterrows()])
    # Prefer an episode the monitor flagged, in file order (no cherry-picking by lead time).
    sel = pd.concat([sel[sel["_alarm"].notna()], sel[sel["_alarm"].isna()]])
    exp = load_exp("configs/exp/push_dr.yaml")
    act = sb3_actor(load_policy(ckpt, exp))
    for _, row in sel.iterrows():
        cid = int(row["cell_id"])
        sl = steplog_path(root, "push_dr", seed, cid)
        if not sl.exists() or cid not in cells:
            continue
        rec = load_steplog(sl)
        if "ep_seed" in row.index and not pd.isna(row["ep_seed"]):
            ep_seed = int(row["ep_seed"])
            ep = int(np.flatnonzero(rec["ep_seed"] == ep_seed)[0])
        else:
            ep = int(row["episode"])
            ep_seed = int(rec["ep_seed"][ep])
        alarm = None if pd.isna(row["_alarm"]) else int(row["_alarm"])
        frames, success = render_rollout(act, exp, cells[cid].dr_spec(), ep_seed)
        if success or bool(rec["success"][ep]):
            log(f"  warning: episode {ep_seed} logged/rendered outcome is not a failure; next")
            continue
        header = f"push_dr s{seed} | held-out: {cells[cid].name} | ep_seed {ep_seed}"
        if alarm is None:
            header += " | no ASM alarm"
        write_gif(out / name, _captioned(frames, header, success, alarm))
        return log(f"wrote {out / name} (cell {cid}, ep_seed {ep_seed}, alarm step {alarm})")
    log(f"skip {name}: no reproducible held-out failure with a step log found")


def gif_shortcut_failure(root: Path, out: Path, log: Log, seed: int = 0, n_try: int = 20) -> None:
    from projectv.eval.cells import NOMINAL
    from projectv.utils.config import load_exp
    from projectv.utils.seeding import eval_episode_seed

    name = "reach_shortcut_failure.gif"
    exp_name = "reach_shortcut_v2"  # v1 (HER) did not learn the hint, see CHANGELOG
    ckpt = _ckpt(root, exp_name, seed)
    if ckpt is None:
        return log(f"skip {name}: no {exp_name} checkpoint for seed {seed}")
    exp = load_exp(f"configs/exp/{exp_name}.yaml")
    act = sb3_actor(load_policy(ckpt, exp))
    for ep in range(n_try):
        sd = eval_episode_seed(seed, 0, ep)
        frames, success = render_rollout(act, exp, NOMINAL.dr_spec(), sd, shortcut_mode="random")
        if not success:
            header = f"{exp_name} s{seed} | hint = random point | ep {ep}"
            write_gif(out / name, _captioned(frames, header, success))
            return log(f"wrote {out / name} (first failing episode: {ep + 1} of {n_try} tried)")
    log(f"skip {name}: all {n_try} random-hint episodes succeeded (no shortcut failure to show)")


def check_render(path: Path, log: Log, steps: int = 10) -> None:
    """Render a few frames of the analytic controller on Reach (backend smoke test)."""
    from projectv.control.analytic_controller import PDController
    from projectv.eval.cells import NOMINAL

    frames, success = render_rollout(
        PDController("reach").act,
        {"task": "reach", "dr_cfg": {}},
        NOMINAL.dr_spec(),
        0,
        steps=steps,
    )
    assert frames[0].ndim == 3 and frames[0].shape[-1] == 3 and frames[0].std() > 0
    write_gif(path, [annotate(f, [f"analytic controller | t = {t}"]) for t, f in enumerate(frames)])
    log(
        f"wrote {path} ({len(frames)} frames {frames[0].shape}, MUJOCO_GL="
        f"{os.environ.get('MUJOCO_GL', '<default>')})"
    )


GIFS = (gif_nominal_success, gif_heldout_failure, gif_shortcut_failure)


def make_all(
    root: Path, out: Path, seed: int = 0, log: Log = print, heldout: str | Path | None = None
) -> None:
    for fn in GIFS:
        try:
            if fn is gif_heldout_failure:
                fn(root, out, log, seed, heldout)
            else:
                fn(root, out, log, seed)
        except Exception as e:  # one broken input must not stop the other GIFs
            log(f"FAILED {fn.__name__}: {type(e).__name__}: {e}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default="results")
    p.add_argument("--out", default=None, help="default: {results}/gifs")
    p.add_argument("--seed", type=int, default=0, help="training seed of the policies shown")
    p.add_argument("--heldout", default=None, help="default: configs/dr/heldout.yaml")
    p.add_argument("--check-render", default=None, help="only render the controller to this path")
    args = p.parse_args(argv)
    if args.check_render:
        return check_render(Path(args.check_render), print)
    root = Path(args.results)
    make_all(root, Path(args.out) if args.out else root / "gifs", args.seed, heldout=args.heldout)


if __name__ == "__main__":
    sys.exit(main())
