"""Shared calibration-method library for the PromptShift-CRC revision.

Every experiment in `revision_v2` calls `run_stream`, so all methods see the same
streams, the same windows and the same update rule.

Changes from the June scripts (see REVISION_PLAN.md):
  * The online update is NOT projected: alpha_{t+1} = alpha_t + gamma * (alpha - miss_t),
    with the Gibbs and Candes (2021) quantile convention (level >= 1 gives +inf, level <= 0
    gives -inf). Their Proposition 4.1 then applies verbatim for the 0/1 miss indicator.
    `project=True` reproduces the June behaviour (clip to [0.02, 0.35]).
  * Adaptive CRC is run with two windows: the full window W of the June paper, and a window
    matched to PromptShift's recent window (`adaptive_crc_matched`). The matching rule is
    fixed in advance (window = recent window) and is never tuned per data set.
  * `adaptive_wK` / `rolling_wK` give any window K for the window sweep.
  * `correct=True` applies the finite-sample correction of Barber et al. (2023) used in the
    theory section: level (1 - alpha_t)(1 + max_s w_s). With the unprojected update the
    Gibbs-Candes guarantee still holds because the level is monotone in alpha_t and the
    edge cases (+inf for alpha_t <= 0, -inf for alpha_t >= 1) are unchanged.
"""
from __future__ import annotations

import numpy as np

PS_FAMILY = {
    "promptshift_crc",
    "no_representation_weighting",
    "no_drift_gate",
    "no_recent_fallback",
    "no_online_update",
}
ONLINE = {
    "adaptive_crc",
    "adaptive_crc_matched",
    "promptshift_crc",
    "no_representation_weighting",
    "no_drift_gate",
    "no_recent_fallback",
    "rep_aci",
}


def weighted_quantile(values: np.ndarray, weights: np.ndarray, level: float) -> float:
    """Weighted empirical quantile with the Gibbs-Candes convention at the extremes."""
    # Gibbs-Candes (2021): the set is everything for alpha_t < 0 (level > 1) and empty for
    # alpha_t > 1 (level < 0). Using strict inequalities keeps level == 1 equal to the
    # empirical maximum instead of "never miss".
    if level > 1.0:
        return float("inf")
    if level <= 0.0:
        return float("-inf")
    order = np.argsort(values, kind="stable")
    v = values[order]
    w = weights[order]
    cdf = np.cumsum(w) / np.sum(w)
    i = int(np.searchsorted(cdf, level, side="left"))
    return float(v[min(i, len(v) - 1)])


def is_online(method: str) -> bool:
    return method in ONLINE or method.startswith("adaptive_w")


def run_stream(emb: np.ndarray, risk: np.ndarray, method: str, cfg: dict, alpha: float = 0.1,
               project: bool = False, correct: bool = False) -> dict:
    """Run one calibration method over one stream. Returns per-step traces."""
    n = len(risk)
    cs, W, RW, bw = cfg["cal_start"], cfg["window"], cfg["recent"], cfg["bw"]
    gk, gc, gamma = cfg["gate_k"], cfg["gate_c"], cfg["gamma"]
    rep_mult, ps_mult = cfg.get("rep_mult", 3), cfg.get("ps_mult", 4)
    public_style = cfg.get("public_style", False)
    online = is_online(method)
    alpha_t = alpha
    m = n - cs
    miss_a = np.zeros(m)
    thr = np.zeros(m)
    ess = np.zeros(m)
    dr = np.zeros(m)
    drpre = np.zeros(m)
    alphas = np.zeros(m)
    infs = np.zeros(m)
    raw_q = np.zeros(m)
    for k, t in enumerate(range(cs, n)):
        if method == "static_crc":
            idx = np.arange(0, cs)
            w = np.ones(len(idx)) / len(idx)
            drift = float(np.linalg.norm(emb[t] - emb[idx].mean(axis=0)))
        elif method in ("rolling_crc", "adaptive_crc", "adaptive_crc_matched") or method.startswith(("adaptive_w", "rolling_w")):
            if method in ("rolling_crc", "adaptive_crc"):
                Wm = W
            elif method == "adaptive_crc_matched":
                Wm = RW
            else:
                Wm = int(method.split("_w")[1])
            idx = np.arange(max(0, t - Wm), t)
            w = np.ones(len(idx)) / len(idx)
            drift = float(np.linalg.norm(emb[t] - emb[idx].mean(axis=0)))
        elif method in ("representation_weighted_crc", "rep_aci"):
            idx = np.arange(max(0, t - rep_mult * W), t)
            dist = np.linalg.norm(emb[idx] - emb[t], axis=1)
            w = np.exp(-(dist ** 2) / bw ** 2) + 1e-8
            w = w / w.sum()
            drift = float(np.average(dist, weights=w))
        elif method in PS_FAMILY:
            idx = np.arange(max(0, t - ps_mult * W), t)
            dist = np.linalg.norm(emb[idx] - emb[t], axis=1)
            if method == "no_representation_weighting":
                sim = np.ones(len(idx)) / len(idx)
                local_drift = float(np.average(dist, weights=sim + 1e-12))
            elif public_style:
                sim = np.exp(-(dist ** 2) / bw ** 2) + 1e-8
                sim = sim / sim.sum()
                local_drift = float(np.average(dist, weights=sim))
            else:
                sim = np.exp(-(dist ** 2) / bw ** 2)
                sim = sim / max(sim.sum(), 1e-12)
                local_drift = float(np.average(dist, weights=sim + 1e-12))
            recent_idx = np.arange(max(0, t - RW), t)
            recent_mask = np.isin(idx, recent_idx).astype(float)
            recent = recent_mask / max(1.0, recent_mask.sum())
            gate = 0.5 if method == "no_drift_gate" else 1.0 / (1.0 + np.exp(gk * (local_drift - gc)))
            w = sim + 1e-8 if method == "no_recent_fallback" else gate * sim + (1.0 - gate) * recent + 1e-8
            w = w / w.sum()
            drift = local_drift
        else:
            raise ValueError(method)
        level = 1.0 - alpha_t
        if correct:
            # Barber et al. (2023) correction: the weighted quantile with an atom at +inf of mass
            # m/(1+m), m = largest normalised weight, equals the plain weighted quantile at level (1-alpha)(1+m).
            level = level * (1.0 + float(w.max()))
        q = weighted_quantile(risk[idx], w, level)
        miss = float(risk[t] > q)
        alphas[k] = alpha_t
        if online:
            alpha_t = round(alpha_t + gamma * (alpha - miss), 12)
            if project:
                alpha_t = float(np.clip(alpha_t, 0.02, 0.35))
        miss_a[k] = miss
        thr[k] = min(max(q, 0.0), 1.0)  # risks live in [0, 1]; q = +-inf is reported as 1 / 0
        infs[k] = float(np.isinf(q))
        raw_q[k] = q
        ess[k] = 1.0 / np.sum(w ** 2)
        drpre[k] = drift  # pre-gate drift, as in the June paper
        dr[k] = float(np.sum(w * np.linalg.norm(emb[idx] - emb[t], axis=1)))  # post-gate drift D_t used in the theory
    return {"miss": miss_a, "threshold": thr, "ess": ess, "drift": dr, "drift_pre": drpre, "alpha": alphas, "inf": infs, "raw_q": raw_q}


def summarise(tr: dict, tau: float = 0.5) -> dict:
    return {
        "coverage": 1.0 - float(tr["miss"].mean()),
        "intervention_rate": float(np.mean(tr["raw_q"] > tau)),  # decision quantity: intervene when q_t > tau
        "frac_inf": float(tr["inf"].mean()),
        "mean_threshold": float(tr["threshold"].mean()),
        "mean_ess": float(tr["ess"].mean()),
        "mean_drift": float(tr["drift"].mean()),
        "mean_drift_pre": float(tr["drift_pre"].mean()),
    }


def local_metrics(miss: np.ndarray, shift_idx: list[int], post: int = 150, win: int = 120) -> dict:
    roll = np.convolve(miss, np.ones(win) / win, mode="valid")
    return {
        "max_local_viol": float(roll.max()),
        "post_shift_viol": float(np.mean([miss[s: s + post].mean() for s in shift_idx])),
        "frac_windows_over_015": float(np.mean(roll > 0.15)),
    }


def gibbs_candes_bound(alpha_1: float, gamma: float, T: int, alpha: float = 0.1) -> float:
    """|T^-1 sum err_t - alpha| <= (max(alpha_1, 1 - alpha_1) + gamma) / (T * gamma)."""
    return (max(alpha_1, 1.0 - alpha_1) + gamma) / (T * gamma)


def block_bootstrap_se(miss: np.ndarray, block: int, reps: int = 1000, seed: int = 0) -> float:
    """Moving-block bootstrap sd of the mean miss indicator (dependence-aware)."""
    rng = np.random.default_rng(seed)
    n = len(miss)
    block = max(1, min(block, n))
    nb = int(np.ceil(n / block))
    starts_max = n - block
    means = np.empty(reps)
    for r in range(reps):
        st = rng.integers(0, starts_max + 1, size=nb)
        sample = np.concatenate([miss[s: s + block] for s in st])[:n]
        means[r] = sample.mean()
    return float(means.std(ddof=1))


CFG_PILOT = dict(cal_start=300, window=300, recent=160, bw=1.8, gate_k=2.0, gate_c=2.0, gamma=0.025, rep_mult=3, ps_mult=3)
CFG_PROTO = dict(cal_start=300, window=280, recent=150, bw=1.85, gate_k=2.2, gate_c=2.05, gamma=0.02, rep_mult=3, ps_mult=3)
CFG_ABL = dict(cal_start=300, window=300, recent=160, bw=1.85, gate_k=2.2, gate_c=2.05, gamma=0.02, rep_mult=3, ps_mult=3)
CFG_TQA = dict(bw=1.25, gate_k=2.5, gate_c=1.35, gamma=0.02, rep_mult=3, ps_mult=3, public_style=True)
CFG_RTP = dict(bw=0.9, gate_k=3.0, gate_c=0.95, gamma=0.015, rep_mult=3, ps_mult=3, public_style=True)
CFG_SUMM = dict(bw=0.75, gate_k=4.0, gate_c=0.85, gamma=0.018, rep_mult=3, ps_mult=3, public_style=True)
CFG_LHQA = dict(bw=0.55, gate_k=5.0, gate_c=0.55, gamma=0.018, rep_mult=3, ps_mult=3, public_style=True)
