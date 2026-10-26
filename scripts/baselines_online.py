"""Stronger online baselines for the revision (referee review item M1).

All return the same trace dictionary as crc_lib.run_stream, so crc_lib.summarise / local_metrics apply.

  pid_quantile_tracking  Angelopoulos, Candes, Tibshirani (2023), P term only (quantile tracking):
                         q_{t+1} = q_t + eta (err_t - alpha), eta = gamma * B with B = 1 (risk range). q_1 = the
                         (1 - alpha) quantile of the calibration prefix. Pre-declared; not tuned.
  dtaci                  Gibbs and Candes (2024), Algorithm 1 with their recommended constants (I = 500,
                         gamma_i = 0.001 * 2^i, i = 0..7), deterministic weighted-mean version, calibration
                         pool = the matched recent window (same rule as adaptive_crc_matched).
  dsr_aci                A DS-CP-style baseline (Lin et al. 2026 weight calibration scores by an estimated density ratio
                         between recent and older prompts): logistic regression on PCA-16 features separates the recent
                         window from the older pool, refit every 25 steps; weights = estimated odds; the online update of
                         the main method is used. This is a simplified stand-in, not the authors' code.
L-ARC (Zecchin and Simeone 2024) is not reimplemented: no reference code was used and a faithful version needs its
RKHS machinery. It is discussed in related work only.
"""
from __future__ import annotations

import numpy as np

from crc_lib import weighted_quantile

ALPHA = 0.1


def _trace(m):
    return dict(miss=np.zeros(m), threshold=np.zeros(m), ess=np.zeros(m), drift=np.zeros(m), drift_pre=np.zeros(m),
                alpha=np.zeros(m), inf=np.zeros(m), raw_q=np.zeros(m))


def _store(tr, k, q, miss, ess, alpha_t, drift=0.0):
    tr["miss"][k] = miss
    tr["threshold"][k] = min(max(q, 0.0), 1.0)
    tr["raw_q"][k] = q
    tr["inf"][k] = float(np.isinf(q))
    tr["ess"][k] = ess
    tr["alpha"][k] = alpha_t
    tr["drift"][k] = drift
    tr["drift_pre"][k] = drift


def pid_quantile_tracking(emb, risk, cfg, alpha=ALPHA):
    n, cs = len(risk), cfg["cal_start"]
    eta = cfg["gamma"] * 1.0
    q = float(np.quantile(risk[:cs], 1 - alpha))
    tr = _trace(n - cs)
    for k, t in enumerate(range(cs, n)):
        miss = float(risk[t] > q)
        _store(tr, k, q, miss, float(cs), alpha)
        q = q + eta * (miss - alpha)
    return tr


def dtaci(emb, risk, cfg, alpha=ALPHA, I=500):
    n, cs, RW = len(risk), cfg["cal_start"], cfg["recent"]
    gam = np.array([0.001 * 2 ** i for i in range(8)])
    k_ = len(gam)
    sigma = 1.0 / (2 * I)
    eta = np.sqrt(3.0 / I) * np.sqrt((np.log(I * k_) + 2) / ((1 - alpha) ** 2 * alpha ** 3 + alpha ** 2 * (1 - alpha) ** 3))
    w = np.ones(k_)
    a_i = np.full(k_, alpha)
    tr = _trace(n - cs)
    for k, t in enumerate(range(cs, n)):
        p = w / w.sum()
        a_t = float(np.dot(p, a_i))
        idx = np.arange(max(0, t - RW), t)
        wts = np.ones(len(idx)) / len(idx)
        level = (1 - a_t) * (1 + wts.max())
        q = weighted_quantile(risk[idx], wts, level)
        miss = float(risk[t] > q)
        # beta_t: the miscoverage level at which s_t is just covered
        beta = 1.0 - float(np.mean(risk[idx] < risk[t]))
        loss = alpha * (beta - a_i) - np.minimum(0.0, beta - a_i)
        wbar = w * np.exp(-eta * loss)
        w = (1 - sigma) * wbar + wbar.sum() * sigma / k_
        a_i = a_i + gam * (alpha - (beta < a_i).astype(float))
        _store(tr, k, q, miss, float(len(idx)), a_t)
    return tr


def dsr_aci(emb, risk, cfg, alpha=ALPHA, refit=25, ncomp=16):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    n, cs, W, RW, gamma = len(risk), cfg["cal_start"], cfg["window"], cfg["recent"], cfg["gamma"]
    pool_len = cfg.get("rep_mult", 3) * W
    pca = PCA(n_components=min(ncomp, emb.shape[1]), random_state=0).fit(emb[:cs])
    z = pca.transform(emb)
    alpha_t = alpha
    tr = _trace(n - cs)
    odds = None
    last_fit = -10 ** 9
    for k, t in enumerate(range(cs, n)):
        lo = max(0, t - pool_len)
        idx = np.arange(lo, t)
        if t - last_fit >= refit or odds is None:
            recent = np.arange(max(0, t - RW), t)
            older = np.arange(lo, max(lo, t - RW))
            if len(older) >= 20:
                X = np.vstack([z[older], z[recent]])
                y = np.r_[np.zeros(len(older)), np.ones(len(recent))]
                clf = LogisticRegression(C=1.0, max_iter=200).fit(X, y)
                s = clf.decision_function(z[lo:t])
                odds = np.exp(np.clip(s, -8, 8))
            else:
                odds = np.ones(t - lo)
            last_fit = t
            base_lo = lo
        # odds were computed on the pool at fit time; extend with 1.0 for newer points, trim for older ones
        cur = np.ones(len(idx))
        take = np.arange(lo, min(t, base_lo + len(odds)))
        cur[take - lo] = odds[take - base_lo] if len(take) else 1.0
        w = cur + 1e-8
        w = w / w.sum()
        level = (1 - alpha_t) * (1 + w.max())
        q = weighted_quantile(risk[idx], w, level)
        miss = float(risk[t] > q)
        _store(tr, k, q, miss, float(1.0 / np.sum(w ** 2)), alpha_t,
               float(np.sum(w * np.linalg.norm(emb[idx] - emb[t], axis=1))))
        alpha_t = round(alpha_t + gamma * (alpha - miss), 12)
    return tr


BASELINES = {"pid_quantile_tracking": pid_quantile_tracking, "dtaci": dtaci, "dsr_aci": dsr_aci}
