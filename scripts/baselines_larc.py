"""L-ARC baseline (Localized Adaptive Risk Control; Zecchin and Simeone, NeurIPS 2024, arXiv:2405.07976).

Standalone module for the revision_v2 harness. It returns the same trace dictionary as
baselines_online.pid_quantile_tracking / crc_lib.run_stream, so crc_lib.summarise and the local metrics apply.
See L_ARC_NOTES.md for the exact parameterisation, the paper-versus-code differences and what was not verified.

Method as implemented (harness orientation: miss_t = 1{risk_t > g_t(x_t)}, a larger threshold lowers the loss):

    g_t(x)  = f_t(x) + c_t,   f_t(x) = sum_{s<t} beta_s * kappa * exp(-||x - x_s||^2 / ell^2)
    miss_t  = 1{ risk_t > g_t(x_t) }
    eta_t   = eta1 / sqrt(t)                       t = 1, 2, ... counts post-prefix steps (t_offset shifts it)
    c_{t+1} = c_t - eta_t (alpha - miss_t)                         (paper Eq. 17)
    f_{t+1} = (1 - lam eta_t) f_t - eta_t (alpha - miss_t) k(x_t, .)   (paper Eq. 18)
              i.e. beta_s <- (1 - lam eta_t) beta_s for s < t, and beta_t = -eta_t (alpha - miss_t)

Initialisation: f_1 = 0; c_1 = (1 - alpha) empirical quantile of risk[:cal_start], the same warm start
pid_quantile_tracking uses (np.quantile, linear interpolation). Only post-prefix points carry coefficients.
No clipping of g_t is applied to the update or to the miss indicator (neither the paper's algorithm nor the authors'
code clips); only the reported `threshold` field is clipped to [0, 1] by the harness's _store, with `raw_q` unclipped.

Kernel length scale: `ell` enters as exp(-||x - x'||^2 / ell^2) (the form used in the authors' code), kernel_form="code".
The paper text writes exp(-||x - x'||^2 / l) with l unsquared; kernel_form="paper" uses that denominator instead
(then l = ell^2). At ell = 1 the two coincide. Cost: O(t) per step, one kernel row via one matrix-vector product.
"""
from __future__ import annotations

from functools import partial

import numpy as np

from baselines_online import _store, _trace  # identical neutral values to the other baselines

ALPHA = 0.1


def _kernel_denominator(ell: float, kernel_form: str) -> float:
    if ell <= 0:
        raise ValueError("ell must be positive")
    if kernel_form == "code":
        return float(ell) ** 2
    if kernel_form == "paper":
        return float(ell)
    raise ValueError("kernel_form must be 'code' or 'paper'")


def matched_length_scale(h: float, kernel_form: str = "code") -> float:
    """Value of `ell` for which the L-ARC kernel (with kappa = 1) equals exp(-||x - x'||^2 / h^2).

    kernel_form="code":  exp(-d^2 / ell^2) = exp(-d^2 / h^2)  =>  ell = h.
    kernel_form="paper": exp(-d^2 / l)     = exp(-d^2 / h^2)  =>  l = h^2.
    h is the PromptShift kernel bandwidth, cfg["bw"]."""
    h = float(h)
    if h <= 0:
        raise ValueError("h must be positive")
    return h if kernel_form == "code" else h * h


def larc(e, risk, cfg, alpha=ALPHA, eta1=1.0, ell=1.0, lam=1e-4, kappa=1.0, kernel_form="code", t_offset=0,
         warm_start=True):
    """Online L-ARC over the post-prefix part of one stream.

    e: (n, d) features in stream order; risk: (n,) in [0, 1]; cfg needs cal_start (and bw if ell == "matched").
    eta1: step-size scale, eta_t = eta1 / sqrt(t_offset + k + 1) at the k-th post-prefix step (k = 0, 1, ...).
    ell: float length scale (see module docstring), or the string "matched" meaning matched_length_scale(cfg["bw"]),
         resolved at call time so that it follows the per-stream bandwidth.
    lam: RKHS regularisation (shrink factor 1 - lam * eta_t); kappa: kernel scale (kappa = 1 in the paper's experiments).
    warm_start: c_1 = (1 - alpha) quantile of risk[:cal_start] (default); False gives the paper's c_1 = 0."""
    risk = np.asarray(risk, dtype=np.float64)
    n, cs = len(risk), cfg["cal_start"]
    m = n - cs
    if isinstance(ell, str):
        if ell != "matched":
            raise ValueError("ell must be a positive number or 'matched'")
        ell = matched_length_scale(cfg["bw"], kernel_form)
    inv_den = 1.0 / _kernel_denominator(ell, kernel_form)
    x = np.ascontiguousarray(np.asarray(e, dtype=np.float64)[cs:])
    sq = np.einsum("ij,ij->i", x, x)
    c = float(np.quantile(risk[:cs], 1 - alpha)) if warm_start else 0.0
    beta = np.zeros(m)
    tr = _trace(m)
    for k in range(m):
        if k:
            d2 = sq[:k] + sq[k] - 2.0 * (x[:k] @ x[k])  # one kernel row, vectorised over all past points
            np.maximum(d2, 0.0, out=d2)
            f = kappa * float(beta[:k] @ np.exp(-d2 * inv_den))
        else:
            f = 0.0
        q = f + c
        miss = float(risk[cs + k] > q)
        _store(tr, k, q, miss, float(cs), alpha)  # ess = float(cs), drift = 0, alpha_t = alpha, as pid_quantile_tracking
        eta = eta1 / np.sqrt(t_offset + k + 1.0)
        step = eta * (miss - alpha)  # = -eta (alpha - miss)
        c += step
        if k:
            beta[:k] *= 1.0 - lam * eta
        beta[k] = step
    return tr


def make_larc_configs(kernel_form="code"):
    """Four pre-declared configurations, each f(e, risk, cfg, alpha=...) like the other BASELINES entries.
    'Default' = the authors' defaults (kappa = 1, ell = 1, lam = 1e-4); 'Matched' = ell equal to the PromptShift
    bandwidth cfg['bw'] (the kernel exp(-d^2 / h^2)), read from cfg at call time. eta1 = 1 is the authors' code
    (electricity, tumor, beam, fruit scripts); eta1 = 0.1 is the paper's tumor-segmentation text."""
    out = {}
    for eta1, tag in ((0.1, "0.1"), (1.0, "1")):
        out[f"larc_eta{tag}_ellDefault"] = partial(larc, eta1=eta1, ell=1.0, lam=1e-4, kappa=1.0, kernel_form=kernel_form)
        out[f"larc_eta{tag}_ellMatched"] = partial(larc, eta1=eta1, ell="matched", lam=1e-4, kappa=1.0, kernel_form=kernel_form)
    return out


LARC_BASELINES = make_larc_configs()


def register(registry: dict) -> dict:
    """registry.update(LARC_BASELINES), e.g. baselines_larc.register(baselines_online.BASELINES)."""
    registry.update(LARC_BASELINES)
    return registry


# ---------------------------------------------------------------------------------------------------------------
# Self-tests (light). Run from the scripts directory, one core:
#   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B baselines_larc.py
# ---------------------------------------------------------------------------------------------------------------
def _unit(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def _reference_thresholds(e, risk, cs, alpha, eta1, ell, lam, kappa, c1):
    """Slow reference: explicit coefficient products instead of in-place shrinking, Python loops over past points."""
    m = len(risk) - cs
    steps, etas, qs = [], [], []
    c = c1
    for k in range(m):
        q = c
        for s in range(k):
            shrink = 1.0
            for u in range(s + 1, k):
                shrink *= 1.0 - lam * etas[u]
            d2 = float(np.sum((e[cs + k] - e[cs + s]) ** 2))
            q += kappa * steps[s] * shrink * np.exp(-d2 / ell ** 2)
        miss = float(risk[cs + k] > q)
        eta = eta1 / np.sqrt(k + 1.0)
        etas.append(eta)
        steps.append(eta * (miss - alpha))
        c += steps[-1]
        qs.append(q)
    return np.array(qs)


def _selftest():
    import time

    import baselines_online as B
    import crc_lib as C

    rng = np.random.default_rng(0)
    cfg = dict(cal_start=256, window=256, recent=128, bw=1.2, gate_k=2.0, gate_c=1.3, gamma=0.02, rep_mult=3, ps_mult=3)
    alpha = 0.1

    # (0) vectorised implementation equals the slow reference (shrinkage, kappa, ell, warm start all active)
    n0, d0, cs0 = 140, 8, 40
    e0 = _unit(rng.normal(size=(n0, d0)))
    r0 = rng.uniform(size=n0)
    c0 = dict(cfg, cal_start=cs0)
    tr0 = larc(e0, r0, c0, alpha=alpha, eta1=0.7, ell=0.8, lam=0.05, kappa=1.5)
    ref = _reference_thresholds(e0, r0, cs0, alpha, 0.7, 0.8, 0.05, 1.5, float(np.quantile(r0[:cs0], 1 - alpha)))
    assert np.allclose(tr0["raw_q"], ref, atol=1e-10), np.abs(tr0["raw_q"] - ref).max()
    # kernel_form / matched helper: the same kernel must come out
    a = larc(e0, r0, dict(c0, bw=0.9), alpha=alpha, eta1=1.0, ell="matched")["raw_q"]
    b = larc(e0, r0, dict(c0, bw=0.9), alpha=alpha, eta1=1.0, ell=0.9)["raw_q"]
    p = larc(e0, r0, dict(c0, bw=0.9), alpha=alpha, eta1=1.0, ell="matched", kernel_form="paper")["raw_q"]
    p2 = larc(e0, r0, dict(c0, bw=0.9), alpha=alpha, eta1=1.0, ell=0.81, kernel_form="paper")["raw_q"]
    assert np.allclose(a, b) and np.allclose(a, p) and np.allclose(p, p2)
    print("test 0 ok: vectorised == slow reference (max abs diff %.2e); matched ell == explicit ell, both forms" % np.abs(tr0["raw_q"] - ref).max())

    # (1) iid stream, random unit vectors in 384 dims, uniform risks: violation rate near alpha
    n1 = 3000
    e1 = _unit(rng.normal(size=(n1, 384)))
    r1 = rng.uniform(size=n1)
    for name, f in make_larc_configs().items():
        v = float(f(e1, r1, cfg, alpha=alpha)["miss"].mean())
        print(f"test 1 {name:26s} violation rate {v:.4f} (target {alpha})")
        assert abs(v - alpha) < 0.03, (name, v)

    # (2) two interleaved regimes: centres orthogonal, risk law depends on the regime; small ell localises
    n2 = 4500
    reg = rng.integers(0, 2, size=n2)
    centres = np.zeros((2, 384))
    centres[0, 0] = 1.0
    centres[1, 1] = 1.0
    e2 = _unit(centres[reg] + 0.3 * rng.normal(size=(n2, 384)) / np.sqrt(384))
    r2 = np.where(reg == 0, rng.uniform(0.0, 0.4, size=n2), rng.uniform(0.5, 1.0, size=n2))
    post = reg[cfg["cal_start"]:]
    res = {}
    for ell in (0.5, 100.0):
        tr = larc(e2, r2, cfg, alpha=alpha, eta1=1.0, ell=ell)
        va, vb = tr["miss"][post == 0].mean(), tr["miss"][post == 1].mean()
        ta, tb = tr["raw_q"][post == 0][-500:].mean(), tr["raw_q"][post == 1][-500:].mean()
        res[ell] = (va, vb, ta, tb)
        print(f"test 2 ell={ell:6.1f}: violation regime A {va:.3f}, regime B {vb:.3f}; "
              f"mean threshold (last 500 per regime) A {ta:.3f}, B {tb:.3f}")
    va, vb, ta, tb = res[0.5]
    assert abs(va - alpha) < 0.04 and abs(vb - alpha) < 0.04, (va, vb)
    assert tb - ta > 0.3, (ta, tb)
    assert abs(res[100.0][0] - res[100.0][1]) > abs(va - vb) + 0.05, "non-local kernel should leave the regime gap"
    mt = larc(e2, r2, dict(cfg, bw=1.2), alpha=alpha, eta1=1.0, ell="matched")
    print("test 2 info (not asserted): ell matched to bw=1.2: violation A %.3f, B %.3f"
          % (mt["miss"][post == 0].mean(), mt["miss"][post == 1].mean()))

    # (3) trace works with crc_lib.summarise, same keys as the other baselines, 4000 online steps in 384 dims
    n3 = 4000 + cfg["cal_start"]
    e3 = _unit(rng.normal(size=(n3, 384)))
    r3 = rng.uniform(size=n3)
    t0 = time.perf_counter()
    tr3 = larc(e3, r3, cfg, alpha=alpha, eta1=1.0, ell=1.0)
    dt = time.perf_counter() - t0
    ref_tr = B.pid_quantile_tracking(e3, r3, cfg, alpha=alpha)
    assert set(tr3) == set(ref_tr) and all(tr3[k].shape == ref_tr[k].shape for k in ref_tr)
    assert np.isfinite(tr3["raw_q"]).all() and tr3["threshold"].min() >= 0 and tr3["threshold"].max() <= 1
    s = C.summarise(tr3)
    print("test 3 summarise:", {k: round(v, 4) for k, v in s.items()})
    print(f"test 3 runtime: {dt:.2f} s for 4000 online steps, d = 384 (one core if the BLAS env vars are set to 1)")
    assert dt < 60.0
    print("all L-ARC self-tests passed")


if __name__ == "__main__":
    _selftest()
