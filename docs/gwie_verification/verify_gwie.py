"""Empirical verification of the ORIGINAL DePlag GWIE (unmodified code) before porting.

Usage: python verify_gwie.py <deplag_root> <out_json> [n_pairs]
"""

import json
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT, OUT = sys.argv[1], sys.argv[2]
N_PAIRS = int(sys.argv[3]) if len(sys.argv) > 3 else 40
sys.path.insert(0, ROOT)

from sentence_transformers import SentenceTransformer  # noqa: E402

from explainability import gwie as G  # noqa: E402
from explainability.integrated_gradients import aggregate_to_words  # noqa: E402

torch.manual_seed(0)
random.seed(0)
rng = np.random.default_rng(0)

model = SentenceTransformer(f"{ROOT}/models/deplag_brain", device="cpu")
model.eval()
ALPHA, IG_STEPS = 0.75, 30
gw = G.GWIE(model, alpha=ALPHA, max_iter=20, ig_steps=IG_STEPS)
ig = gw.ig


def enc(texts):
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                        show_progress_bar=False, batch_size=64)


def sim(text, e_src):
    return float(enc([text])[0] @ e_src)


def signed_ig(text, e_src_np):
    """Same path/forward as their IntegratedGradients.attribute, keeping the signed vector."""
    e_src = torch.tensor(e_src_np, dtype=torch.float32).unsqueeze(0)
    e = ig._tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=False)
    with torch.no_grad():
        x = ig._embed_table(e["input_ids"])
    base = torch.zeros_like(x)
    acc = torch.zeros_like(x)
    for k in range(1, IG_STEPS + 1):
        xi = (base + k / IG_STEPS * (x - base)).detach().requires_grad_(True)
        ig._cosine_sim_from_embeds(xi, e, e_src, "cpu").backward()
        acc += xi.grad
    attr = (x - base) * acc / IG_STEPS
    with torch.no_grad():
        f_x = ig._cosine_sim_from_embeds(x, e, e_src, "cpu").item()
        f_b = ig._cosine_sim_from_embeds(base, e, e_src, "cpu").item()
    toks = ig._tokenizer.convert_ids_to_tokens(e["input_ids"][0].tolist())
    return toks, attr.squeeze(0), f_x, f_b


def valid_words(text, chosen):
    return [w for w in dict.fromkeys(text.split()) if gw._is_valid_candidate(w, chosen)]


def trajectory(text, e_src, pick, k_max=5):
    """Erase k_max words chosen by `pick(cur_text, chosen_keys)`; return sims [s0..sk]."""
    cur, chosen, sims = text, set(), [sim(text, e_src)]
    for _ in range(k_max):
        w = pick(cur, chosen)
        if w is None:
            break
        chosen.add(G._canonical_word(w))
        cur = G._mask_word(cur, w)
        sims.append(sim(cur, e_src))
    return sims


def pick_random(cur, chosen):
    c = valid_words(cur, chosen)
    return random.choice(c) if c else None


def make_pick_oracle(e_src):
    def pick(cur, chosen):
        c = valid_words(cur, chosen)
        if not c:
            return None
        s = enc([G._mask_word(cur, w) for w in c]) @ e_src
        return c[int(np.argmin(s))]
    return pick


def make_pick_signed(e_src):
    """Variant: argmax of SIGNED word IG (sum of token-level signed contributions)."""
    def pick(cur, chosen):
        toks, attr, _, _ = signed_ig(cur, e_src)
        tok_scores = attr.sum(-1).tolist()
        words = aggregate_signed(toks, tok_scores)
        c = [(w, s) for w, s in words if gw._is_valid_candidate(w, chosen)]
        return max(c, key=lambda t: t[1])[0] if c else None
    return pick


def aggregate_signed(toks, scores):
    out, cw, cs = [], "", 0.0
    for t, s in zip(toks, scores):
        if t in ("[CLS]", "[SEP]", "<s>", "</s>", "<pad>", "[PAD]"):
            continue
        if t.startswith("##"):
            cw += t[2:]
            cs += s
        else:
            if cw:
                out.append((cw, cs))
            cw, cs = t, s
    if cw:
        out.append((cw, cs))
    best = {}
    for w, s in out:
        w = G._canonical_word(w)
        if w:
            best[w] = max(best.get(w, -1e9), s)
    return list(best.items())


def pick_gwie_original(e_src_np):
    """Their own selection step, exactly as in GWIE.explain."""
    def pick(cur, chosen):
        ta = ig.attribute(cur, e_src_np, device="cpu")
        wa = aggregate_to_words(ta, cur)
        c = [(w, s) for w, s in wa if gw._is_valid_candidate(w, chosen)]
        return max(c, key=lambda t: t[1])[0] if c else None
    return pick


# ---- data: positive pairs, initial sim >= alpha, <= 40 words ---------------------------------
df = pd.read_csv(f"{ROOT}/data/final_train_data.csv")
df = df[(df.label == 1) & df.text_b.str.split().str.len().between(8, 40)]
df = df.sample(n=min(len(df), 600), random_state=0)
pairs = []
for a, b in zip(df.text_a, df.text_b):
    e = enc([a])[0]
    if sim(b, e) >= ALPHA:
        pairs.append((a, b, e))
    if len(pairs) == N_PAIRS:
        break

res = {"n_pairs": len(pairs), "alpha": ALPHA, "ig_steps": IG_STEPS, "per_pair": []}
t0 = time.time()
for i, (a, b, e) in enumerate(pairs):
    row = {}
    # T1: their IG == norm of signed IG; completeness of signed IG
    toks, attr, f_x, f_b = signed_ig(b, e)
    theirs = np.array([s for _, s in ig.attribute(b, e, device="cpu")])
    mine = attr.norm(dim=-1).numpy()
    mine = mine / mine.max()
    row["ig_match_maxabs"] = float(np.abs(theirs - mine).max())
    row["completeness_err"] = float(abs(attr.sum().item() - (f_x - f_b)))
    row["completeness_target"] = float(f_x - f_b)
    # T2/T4/T5/T6: full original explain()
    out = gw.explain(b, e)
    it = out["iterations"]
    row["converged"] = out["converged"]
    row["n_iter"] = len(it)
    row["set_size"] = len(out["counterfactual_set"])
    row["sim_increase_steps"] = sum(s["sim_after"] > s["sim_before"] for s in it)
    row["final_sim"] = out["final_similarity"]
    # 1-minimality of pruned set
    C = out["counterfactual_set"]
    if out["converged"]:
        row["valid"] = sim(G._mask_words(b, C), e) < ALPHA
        row["one_minimal"] = all(
            sim(G._mask_words(b, [w for w in C if w != x]), e) >= ALPHA for x in C
        )
    # T3: first-5 trajectories
    row["traj_gwie"] = trajectory(b, e, pick_gwie_original(e))
    row["traj_signed"] = trajectory(b, e, make_pick_signed(e))
    row["traj_oracle"] = trajectory(b, e, make_pick_oracle(e))
    row["traj_random"] = np.mean(
        [np.pad(t := trajectory(b, e, pick_random), (0, 6 - len(t)), mode="edge")
         for _ in range(5)], axis=0).tolist()
    res["per_pair"].append(row)
    print(f"[{i + 1}/{len(pairs)}] {time.time() - t0:.0f}s conv={row['converged']} "
          f"|C|={row['set_size']} ig_match={row['ig_match_maxabs']:.1e}", flush=True)

with open(OUT, "w") as f:
    json.dump(res, f, indent=1)
print("DONE", OUT)
