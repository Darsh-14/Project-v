"""Monitor scoring and metrics shared by ASM and all baselines (mvp.md §7).

Every monitor produces a per-step raw score `m` of shape (n_episodes, T). The same aggregation is
applied to all of them: EWMA smoothing, then a prefix score S(T) = max_{t<T} e_t.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def ewma(m: np.ndarray, alpha: float) -> np.ndarray:
    """e_0 = m_0, e_t = alpha * m_t + (1 - alpha) * e_{t-1}, along the last axis."""
    assert 0.0 < alpha <= 1.0, alpha
    m = np.asarray(m, dtype=float)
    e = np.empty_like(m)
    e[..., 0] = m[..., 0]
    for t in range(1, m.shape[-1]):
        e[..., t] = alpha * m[..., t] + (1.0 - alpha) * e[..., t - 1]
    return e


def prefix_score(e: np.ndarray, T: int) -> np.ndarray:
    """S(T) = max over the first T steps."""
    return np.max(np.asarray(e)[..., :T], axis=-1)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Labels: 1 = episode failed."""
    return float(roc_auc_score(labels, scores))


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(average_precision_score(labels, scores))


def tpr_at_fpr(
    calib_negative_scores: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    fpr: float = 0.05,
) -> tuple[float, float, float]:
    """Threshold = (1 - fpr) quantile of calib-split successful-episode scores.

    Returns (TPR on test, threshold, realised FPR on test).
    """
    thr = float(np.quantile(calib_negative_scores, 1.0 - fpr))
    test_labels = np.asarray(test_labels).astype(bool)
    alarm = np.asarray(test_scores) > thr
    tpr = float(alarm[test_labels].mean()) if test_labels.any() else float("nan")
    real_fpr = float(alarm[~test_labels].mean()) if (~test_labels).any() else float("nan")
    return tpr, thr, real_fpr


def detection_steps(e: np.ndarray, threshold: float) -> np.ndarray:
    """First step where e_t > threshold per episode; NaN if never."""
    above = np.asarray(e) > threshold
    first = np.argmax(above, axis=-1).astype(float)
    first[~above.any(axis=-1)] = np.nan
    return first


def cluster_bootstrap_ci(
    metric: Callable[[np.ndarray, np.ndarray], float],
    scores: np.ndarray,
    labels: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Two-level bootstrap (CLAUDE.md §7): resample clusters (training seeds), then episodes
    within each sampled cluster. Resamples with a single label class are skipped.

    Returns (point estimate, lower, upper).
    """
    scores, labels, clusters = map(np.asarray, (scores, labels, clusters))
    point = metric(scores, labels)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(clusters == c) for c in np.unique(clusters)]
    stats = []
    for _ in range(n_boot):
        picked = rng.integers(len(groups), size=len(groups))
        idx = np.concatenate([rng.choice(groups[g], size=len(groups[g])) for g in picked])
        if np.unique(labels[idx]).size < 2:
            continue
        stats.append(metric(scores[idx], labels[idx]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)
