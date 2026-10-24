import json
import sys

import numpy as np

r = json.load(open(sys.argv[1]))
P = r["per_pair"]
n = len(P)
print(f"pairs={n} alpha={r['alpha']} ig_steps={r['ig_steps']}")

ig = np.array([p["ig_match_maxabs"] for p in P])
ce = np.array([p["completeness_err"] for p in P])
ct = np.array([abs(p["completeness_target"]) for p in P])
print(f"T1 IG reproduces signed-IG norms: max |diff| = {ig.max():.1e}")
print(f"T1 completeness |sum IG - (f(x)-f(0))|: median {np.median(ce):.4f}, max {ce.max():.4f} "
      f"(|f(x)-f(0)| median {np.median(ct):.3f}; rel err median {np.median(ce / ct):.1%})")

steps = sum(p["n_iter"] for p in P)
inc = sum(p["sim_increase_steps"] for p in P)
print(f"T2/T4 erasure steps that INCREASED similarity: {inc}/{steps} ({inc / steps:.1%}); "
      f"pairs with >=1 such step: {sum(p['sim_increase_steps'] > 0 for p in P)}/{n}")

conv = [p for p in P if p["converged"]]
print(f"T4 converged (sim < alpha within 20 iters): {len(conv)}/{n}")
sizes = np.array([p["set_size"] for p in conv])
print(f"T6 |C| (converged, after pruning): median {np.median(sizes):.0f}, "
      f"range {sizes.min()}-{sizes.max()}, in [3,8]: {np.mean((sizes >= 3) & (sizes <= 8)):.0%}")
print(f"T5 valid (masked set crosses alpha): {sum(p['valid'] for p in conv)}/{len(conv)}; "
      f"1-minimal: {sum(p['one_minimal'] for p in conv)}/{len(conv)}")


def drop(key, k):
    out = []
    for p in P:
        t = p[key]
        t = t + [t[-1]] * (6 - len(t))
        out.append(t[0] - t[k])
    return np.array(out)


print("T3 mean similarity drop after k erasures (higher = better search):")
print("   k   gwie_orig   signed_ig   random   greedy_oracle   gwie>random(pairs)")
for k in range(1, 6):
    g, s, rnd, o = (drop(x, k) for x in ("traj_gwie", "traj_signed", "traj_random", "traj_oracle"))
    print(f"   {k}   {g.mean():.4f}      {s.mean():.4f}      {rnd.mean():.4f}   {o.mean():.4f}"
          f"          {np.mean(g > rnd):.0%}")
g1, o1 = drop("traj_gwie", 1), drop("traj_oracle", 1)
print(f"   gwie step-1 pick == oracle pick quality (within 1e-4): {np.mean(np.abs(g1 - o1) < 1e-4):.0%}")
