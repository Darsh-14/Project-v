# GWIE: the original algorithm, its verification, and the continuous-control adaptation

## Source

| | |
|---|---|
| Repo | DePlag (`C:\Users\hp\OneDrive\Desktop\DePlag A Web-Based Explainable NLP System for Detecting Paraphrased Plagiarism`), git HEAD `5ba9fcb` (2026-01-30) |
| Implementation | `explainability/gwie.py` (sha256 `3bedc6bd…d38e1`) |
| Scoring dependency | `explainability/integrated_gradients.py` (sha256 `d8c68bf1…ce218`) |
| Paper description | `Paper_pdf/DePlag_Revised_Draft.md` §3.5, Algorithm 1 (sha256 `665a9076…d00`) |
| Older copy | `DePlag_Colab.py` writes an earlier `gwie.py` without stop-word filtering and without pruning. The file above is the current, deployed version (`server.py` imports it). |

## What GWIE computes

GWIE is a **counterfactual-set search**, not a scoring method. Given a suspicious sentence and a
source embedding, it greedily masks words until the SBERT cosine similarity drops below the
plagiarism threshold, then prunes the set. Its output is the set of erased words
(`counterfactual_set`), the masked text, and a per-iteration log. It never assigns a score to
every word.

### 1. Model output being explained
`S(text) = cos(SBERT(text), e_src)`. `e_src` is the (fixed) source-sentence embedding, so the
target is a similarity to a *different* input, not to the model's own output on the original
text.

### 2. Score / weighting formula (per iteration, recomputed on the current text)
- Integrated Gradients of `S` with respect to the token **input embeddings**: zero-embedding
  baseline, `ig_steps` right-endpoint Riemann steps (30 in GWIE; 50 is the standalone default).
- Per-token score = **L2 norm** of the token's IG vector over embedding dimensions. **The sign
  is discarded.**
- Token scores are max-normalised. Sub-word tokens (`##`) are **summed** into words, re-normalised,
  cleaned (alphanumerics, `-`, `'`), and duplicate words are collapsed by max.
- Candidates exclude stop-words, words of length ≤ 2 (unless upper-case), already-chosen words,
  and the mask token itself.
- **Pivot = argmax word score** among candidates.

### 3. Erasure operator / baseline
Replace the **first occurrence** of the pivot word (case-insensitive, word-boundary regex) with
the literal string `[ERASED]`, which the tokenizer then splits into `[`, `erased`, `]`. The IG
baseline (zero embedding) and the erasure value (`[ERASED]` token) are different things.

### 4. Iteration order
Greedy and adaptive. Each iteration recomputes IG on the partially erased text, so earlier
erasures change later scores.

### 5. Stopping rule
Stop when `S(current) < alpha` (`alpha` = 0.75, the deployed plagiarism threshold) → `converged`.
Also stop when no candidates remain, or after `max_iter` iterations (20 in the class default; 6 in
`server.py`). If masking does not change the text, the word is marked as chosen and the
iteration is still consumed.

### 6. Pruning (post-hoc minimisation)
For each chosen word in erasure order, re-mask the *original* text with the current set minus
that word. If the similarity is still below `alpha`, drop the word. One pass. `converged` is then
recomputed from the pruned set.

### 7. Final output ("attribution")
- `counterfactual_set`: pruned, in erasure order.
- `iterations`: a list of `{step, pivot_word, attribution (normalised IG score), sim_before, sim_after}`.
- Initial/final similarity, `converged`.

There is no per-unit attribution vector. The closest quantity is the per-step similarity drop
`sim_before − sim_after` of each erased word.

### Paper vs code
Algorithm 1 in the paper has no pruning step, no candidate filter and no `max_iter` inside the
loop. It states that convergence is guaranteed "because token removal is monotonically
non-increasing in cosine similarity". The code adds pruning, filters and caps. The monotonicity
claim is checked empirically below.

## Verification of the original (before porting)

Harness: `docs/gwie_verification/verify_gwie.py`. It imports DePlag's `gwie.py` and
`integrated_gradients.py` **unmodified** and runs them with the deployed fine-tuned SBERT
(`models/deplag_brain`) on 40 positive pairs from DePlag's `data/final_train_data.csv`
(label 1, 8–40 words, initial similarity ≥ 0.75, `random_state=0`), using alpha 0.75 and 30 IG
steps on CPU. Raw results: `results40.json`. Summary: `summary.txt` (regenerate with
`summarize.py`).

| Check | Result | Verdict |
|---|---|---|
| IG implementation reproduces a signed IG through the same forward pass | max abs diff 1.8e-7 | correct |
| IG completeness, sum IG vs f(x) − f(0), 30 steps | median relative error 1.9% (max abs 0.009) | correct (Riemann error only) |
| Search quality: mean similarity drop after k = 1..5 erasures | GWIE 0.071 / 0.137 / 0.187 / 0.220 / 0.239 | works |
| … uniformly random valid-word order | 0.029 / 0.063 / 0.094 / 0.128 / 0.158 | GWIE > random on 90–100% of pairs |
| … brute-force greedy oracle (best single erasure each step) | 0.084 / 0.153 / 0.203 / 0.232 / 0.255 | GWIE reaches 84–94% of the oracle |
| … signed-IG pivot variant | 0.072 / 0.136 / 0.178 / 0.199 / 0.213 | no better: sign-dropping is not a defect in practice |
| Paper claim: erasure monotonically non-increasing | **22/261 steps (8.4%) increased similarity**, in 14/40 pairs | **claim false** |
| Paper claim: convergence guaranteed | 33/40 converged within 20 iterations | **not guaranteed** (the `max_iter` cap is needed) |
| Pruned set is valid (crosses alpha) | 33/33 | correct |
| Pruned set is 1-minimal | 32/33 | near-minimal (single pass on a non-monotone function) |
| Paper claim: \|C\| about 3–8 | median 5, range 2–11, 85% within 3–8 | roughly holds |

**Conclusion.** The mechanism works as a guided greedy search. It is clearly better than random
and close to an exhaustive greedy oracle, at roughly 1/|vocab| of the oracle's forward-pass
cost. Two paper statements do not hold (monotonicity, guaranteed convergence). Neither affects
the code, which has the safety cap, but Project V must not repeat them. The port keeps
magnitude-based selection because the signed variant was not better.

## Continuous-control adaptation

Two variants are implemented, because the CLAUDE.md §8 provisional pseudocode differs from the
original in several non-trivial ways.

- **`gwie_orig_port`**: a direct port. Same target type (cosine to a fixed reference), IG
  scoring, threshold stopping, pruning, set output.
- **`gwie_c`**: the C2 method. Same guided-erasure mechanism (IG recomputed on the current
  erased state, magnitude pivot, iterative erasure), but it runs over all groups and assigns
  each a marginal divergence gain, so it produces an attribution for every group and a full
  ranking (needed for C1 τ and deletion curves).

### Deviations (original → `gwie_orig_port` / `gwie_c`)

| Aspect | Original (DePlag) | `gwie_orig_port` | `gwie_c` | Why |
|---|---|---|---|---|
| Unit erased | word (sub-word tokens summed) | feature group | feature group | modality |
| Model output | cos(SBERT(text), e_src), e_src a fixed other input | S(s') = cos(mu(s'), mu(s)), reference = action mean at the original state, pre-tanh | f(s') = <mu(s'), mu(s)/‖mu(s)‖> (CLAUDE.md §8) | modality: no second input exists; the reference is the policy's own decision. `gwie_c` uses the CLAUDE.md scalar target shared by all gradient methods |
| Pivot score | IG on current text; per-token L2 over embedding dims; sub-words summed | IG on current state; per-group Σ\|IG_i\| | same as port | per-feature IG is a scalar, so L2 over dims becomes \|·\|; the sub-word sum becomes the group sum |
| IG baseline | zero embedding | pool mean (marginal-mean) | pool mean | CLAUDE.md §8 forbids zero baselines for positions; 0 m is a meaningful position |
| IG steps / rule | 30, right-endpoint Riemann | 30, right-endpoint | 30, midpoint | port faithful; midpoint is exact for linear policies |
| Erasure value | literal `[ERASED]` token | marginal: M = 8 pool rows, one per chain; quantities averaged over chains | same | CLAUDE.md §8 erasure convention; there is no "mask token" for continuous features. Each chain erases every group from the same pool row, so jointly erased groups keep their joint distribution |
| Candidate filter | stop-words, length ≤ 2, mask token | none (all not-yet-erased groups) | none | no analogue |
| Stopping | S < alpha (0.75), or no candidates, or max_iter (20) | S < alpha (0.75; same constant, not calibrated for action space), or all groups erased, or max_iter (20) | no threshold: erase all groups | `gwie_c` needs a score for every group |
| Pruning | one pass, drop words not needed to stay below alpha | same | none | without a threshold there is no set to minimise |
| Output | ordered counterfactual set + per-step log | set C (pruned), erasure order, per-step S trajectory | A[g] = D(π(s)‖π(s_after g)) − D(π(s)‖π(s_before g)) (KL, chain-averaged), shares \|A\|/Σ\|A\|, erasure order | `gwie_c`: CLAUDE.md marginal divergence gain, the analogue of the original's logged `sim_before − sim_after` with KL instead of cosine |
| Ranking for C1 metrics | n/a | erasure order (pre-pruning), then not-erased groups by their last IG score | erasure order | evaluation adapter only; the port's group score = its per-step S drop (0 if never erased) |

The README and report may call `gwie_c` "adapted from GWIE (DePlag)" with a link to this
table. They must not say it *is* GWIE, and must not repeat the monotonicity or convergence claims.
