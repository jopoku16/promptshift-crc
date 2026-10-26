"""Round 4, item B: does the weighting help when the kernel localises?

Pre-declared in PREREGISTRATION_ROUND4_2026-10-26.md (item B and Amendment 1). Nothing is selected; every bandwidth is reported.
  sweep : B1, the declared band-block (RealToxicityPrompts) or category-block (MMLU) orderings with the kernel bandwidth
          h' in {h/4, h/2, h, 2h}, gate centre 1.1 h' and slope 2.5 / h'.
  recur : B2, recurring-regime orderings: blocks of R = 128 items cycle through a permutation of the regimes (RealToxicityPrompts:
          7 blocks per band; MMLU: 4 blocks per category), 24 permutations, post-shift window L = R, same four bandwidths.
Only the probability risk is used. Writes results/round4_<mode>_<task>[_<tag>][_alpha<a>].csv.

Usage: python run_round4_sweep.py sweep|recur rtp --tag gpt2|qwen05b
       python run_round4_sweep.py sweep|recur mmlu [--alpha 0.2]
"""
from __future__ import annotations
import csv, itertools, sys, zlib
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
MODE, _task = sys.argv[1], sys.argv[2]
_rest = sys.argv[3:]
_orig_argv = list(sys.argv)
sys.argv = ["run_real_streams.py", _task] + _rest
import run_real_streams as R  # reads task, tag and alpha from sys.argv at import time
sys.argv = _orig_argv
import crc_lib as C
import baselines_online as B

ALPHA, TAG = R.ALPHA, R.RTP_TAG
MULTS = [0.25, 0.5, 1.0, 2.0]
KERNEL = ["promptshift_crc", "rep_aci", "no_representation_weighting", "no_drift_gate", "no_recent_fallback"]
REFS = ["static_crc", "rolling_crc", "adaptive_crc", "adaptive_crc_matched"] + [f"adaptive_w{k}" for k in (R.RW // 4, R.RW // 2, R.RW, 2 * R.RW, 4 * R.RW)] + list(B.BASELINES)
REFS_SWEEP = [m for m in REFS if m in ("static_crc", "rolling_crc", "adaptive_crc", "adaptive_crc_matched", "dtaci") or m.startswith("adaptive_w")]
G = {}


def recurrence_orders(recs):
    """Blocks of RW items cycle through a permutation of the regimes (a fixed random order inside each regime)."""
    grp = np.array([R.group_of(r) for r in recs])
    regimes = sorted(set(grp.tolist()), key=str)
    per = 7 if _task == "rtp" else 4
    orders = {}
    for pi, perm in enumerate(itertools.permutations(regimes)):
        seqs = {}
        for c in regimes:
            rng = np.random.default_rng(zlib.crc32(f"{_task}_{TAG}_recur_{c}".encode()))
            seqs[c] = rng.permutation(np.where(grp == c)[0])[: per * R.RW]
            assert len(seqs[c]) == per * R.RW, (c, len(seqs[c]))
        idx, bounds = [], []
        for j in range(per):
            for c in perm:
                idx += seqs[c][j * R.RW: (j + 1) * R.RW].tolist()
                bounds.append(len(idx))
        orders[f"recur_{pi:02d}"] = (np.array(idx), bounds[:-1])
    return orders


def job(args):
    name, idx, bounds, post = args
    recs, emb = G["recs"], G["emb"]
    risk = np.array([recs[i]["risk"] for i in idx])
    e = emb[idx]
    cfg = R.make_cfg(e)
    h = cfg["bw"]
    shifts = [b - R.CS for b in bounds if b - R.CS >= 0]
    win = 120 if _task == "mmlu" else 60
    rows = []

    def add(m, tr, mult):
        row = {"task": _task, "tag": TAG, "stream": name, "risk": "prob", "method": m, "bw_mult": mult, **C.summarise(tr),
               "abs_err": abs(1 - tr["miss"].mean() - (1 - ALPHA)), "alpha": ALPHA, "h": h,
               "gc_bound": C.gibbs_candes_bound(ALPHA, cfg["gamma"], len(tr["miss"])) if C.is_online(m) else float("nan")}
        roll = np.convolve(tr["miss"], np.ones(win) / win, mode="valid")
        row["max_local_viol"] = float(roll.max())
        row["post_shift_viol"] = float(np.mean([tr["miss"][s: s + post].mean() for s in shifts])) if shifts else float("nan")
        rows.append(row)

    for m in (REFS if MODE == "recur" else REFS_SWEEP):
        tr = B.BASELINES[m](e, risk, cfg, alpha=ALPHA) if m in B.BASELINES else C.run_stream(e, risk, m, cfg, alpha=ALPHA, correct=True)
        add(m, tr, float("nan"))
    for mult in MULTS:
        c2 = dict(cfg, bw=h * mult, gate_c=1.1 * h * mult, gate_k=2.5 / (h * mult))
        for m in KERNEL:
            add(m, C.run_stream(e, risk, m, c2, alpha=ALPHA, correct=True), mult)
    return rows


def _init(r, e):
    G.update(recs=r, emb=e)


if __name__ == "__main__":
    recs, emb = R.load()
    if MODE == "sweep":
        orders = R.build_orders(recs, emb)
        prefix = {"rtp": "bandblock", "mmlu": "catblock"}[_task]
        jobs = [(n, i, b, R.SETUP["post"]) for n, (i, b) in orders.items() if n.startswith(prefix)]
    else:
        jobs = [(n, i, b, R.RW) for n, (i, b) in recurrence_orders(recs).items()]
    print(MODE, _task, TAG, "alpha", ALPHA, "jobs", len(jobs), flush=True)
    with Pool(7, initializer=_init, initargs=(recs, emb)) as p:
        res = p.map(job, jobs, chunksize=1)
    rows = [r for rr in res for r in rr]
    sfx = (f"_{TAG}" if _task == "rtp" else "") + ("" if ALPHA == 0.1 else f"_alpha{ALPHA}")
    with (ROOT / "results" / f"round4_{MODE}_{_task}{sfx}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", len(rows), flush=True)
