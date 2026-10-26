"""30-seed run of the stronger online baselines on the five synthetic streams (same seeds as phase A)."""
from __future__ import annotations
import csv, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import crc_lib as C, baselines_online as B
from run_phase_a_synthetic import STREAMS, SEEDS


def job(a):
    name, seed = a
    make, cfg, shifts = STREAMS[name]
    emb, risk = make(seed)
    rows = []
    for m, f in B.BASELINES.items():
        tr = f(emb, risk, cfg)
        rows.append({"stream": name, "seed": seed, "method": m, **C.summarise(tr), **C.local_metrics(tr["miss"], shifts),
                     "abs_err": abs(1 - tr["miss"].mean() - 0.9)})
    return rows

if __name__ == "__main__":
    with Pool(6) as p:
        out = p.map(job, [(n, s) for n in STREAMS for s in SEEDS], chunksize=1)
    rows = [r for rr in out for r in rr]
    with (ROOT / "results" / "phase_a_baselines_runs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", len(rows))
