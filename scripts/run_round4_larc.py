"""Round 4, item E: L-ARC (Zecchin and Simeone, 2024) on the paper's streams.

Pre-declared in PREREGISTRATION_ROUND4_2026-10-26.md (Amendment 2). Four configurations, all reported:
eta_1 in {0.1, 1} x kernel length scale in {1 (authors' default), matched to the paper's own kernel}. Primary: eta_1 = 1, length scale 1.
Writes results/round4_larc_<name>.csv in the schema of the runs files (method names larc_*), analysed by analyze_round4_larc.py.

Usage: python run_round4_larc.py synth
       python run_round4_larc.py real mmlu|truthfulqa|rtp [--tag qwen05b]
"""
from __future__ import annotations
import csv, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
MODE = sys.argv[1]
if MODE == "real":
    _task = sys.argv[2]
    _rest = sys.argv[3:]
    _orig_argv = list(sys.argv)
    sys.argv = ["run_real_streams.py", _task] + _rest
    import run_real_streams as R
    sys.argv = _orig_argv
import crc_lib as C
import baselines_larc as L

ALPHA = 0.1
G = {}


def _init(*objs):
    G["objs"] = objs


def synth_job(args):
    import run_phase_a_synthetic as S
    name, seed = args
    make, cfg, shifts = S.STREAMS[name]
    emb, risk = make(seed)
    rows = []
    for m, fn in L.LARC_BASELINES.items():
        tr = fn(emb, risk, cfg, alpha=ALPHA)
        rows.append({"stream": name, "seed": seed, "method": m, **C.summarise(tr), **C.local_metrics(tr["miss"], shifts),
                     "gc_bound": float("nan"), "abs_err": abs(1 - tr["miss"].mean() - 0.9)})
    return rows


def real_job(args):
    name, idx, bounds, risk_kind = args
    recs, emb = G["objs"]
    risk = np.array([recs[i]["risk"] if risk_kind == "prob" else float(recs[i]["error"]) for i in idx])
    e = emb[idx]
    cfg = R.make_cfg(e)
    shifts = [b - R.CS for b in bounds if b - R.CS >= 0]
    rows = []
    for m, fn in L.LARC_BASELINES.items():
        tr = fn(e, risk, cfg, alpha=ALPHA)
        row = {"task": R.TASK, "stream": name, "risk": risk_kind, "method": m, **C.summarise(tr),
               "abs_err": abs(1 - tr["miss"].mean() - (1 - ALPHA)), "alpha": ALPHA, "h": cfg["bw"], "gc_bound": float("nan")}
        win = 120 if R.TASK == "mmlu" else 60
        roll = np.convolve(tr["miss"], np.ones(win) / win, mode="valid")
        row["max_local_viol"] = float(roll.max())
        row["post_shift_viol"] = float(np.mean([tr["miss"][s: s + R.SETUP["post"]].mean() for s in shifts])) if shifts else float("nan")
        rows.append(row)
    return rows


if __name__ == "__main__":
    if MODE == "synth":
        import run_phase_a_synthetic as S
        jobs = [(n, s) for n in S.STREAMS for s in S.SEEDS]
        with Pool(7) as p:
            res = p.map(synth_job, jobs, chunksize=1)
        out_name = "synthetic"
    else:
        recs, emb = R.load()
        orders = R.build_orders(recs, emb)
        fams = {"mmlu": ("catblock", "subjectblock"), "rtp": ("bandblock",), "truthfulqa": ("categoryblock",)}[_task]
        jobs = [(n, i, b, rk) for n, (i, b) in orders.items() if n.startswith(fams) for rk in ("prob", "error")]
        print("real", _task, R.MODEL_TAG, "jobs", len(jobs), flush=True)
        with Pool(7, initializer=_init, initargs=(recs, emb)) as p:
            res = p.map(real_job, jobs, chunksize=1)
        out_name = f"{_task}_{R.MODEL_TAG}"
    rows = [r for rr in res for r in rr]
    with (ROOT / "results" / f"round4_larc_{out_name}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", out_name, len(rows), flush=True)
