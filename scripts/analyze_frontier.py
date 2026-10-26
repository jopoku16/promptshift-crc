"""Validity/sharpness frontier. For each seed, interpolate post-shift violation (and max local
violation) of Adaptive CRC as a function of mean threshold across its window sweep, then ask whether
each method lies below that seed's frontier at its own mean threshold. Negative = better than the frontier."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
suffix = "" if len(sys.argv) < 2 else sys.argv[1]
runs = pd.read_csv(ROOT / "results" / f"phase_a_synthetic_runs{suffix}.csv")
bl = ROOT / "results" / "phase_a_baselines_runs.csv"
if bl.exists() and suffix == "":
    runs = pd.concat([runs, pd.read_csv(bl)], ignore_index=True)
adaptive_sweep = lambda m: m.startswith("adaptive_w") or m in ("adaptive_crc", "adaptive_crc_matched")
targets = ["promptshift_crc", "rep_aci", "no_drift_gate", "no_representation_weighting", "no_recent_fallback",
           "adaptive_crc", "adaptive_crc_matched", "pid_quantile_tracking", "dtaci", "dsr_aci"]
rows = []
for (stream, seed), g in runs.groupby(["stream", "seed"]):
    ad = g[g.method.map(adaptive_sweep)].sort_values("mean_threshold")
    x = ad.mean_threshold.values
    for ymet in ["post_shift_viol", "max_local_viol", "intervention_rate"]:
        y = ad[ymet].values
        # monotone (isotonic-free) piecewise-linear interpolation, clipped at the ends
        for m in targets:
            if not (g.method == m).any():
                continue
            r = g[g.method == m].iloc[0]
            yf = np.interp(r.mean_threshold, x, y)
            rows.append((stream, seed, m, ymet, r[ymet] - yf))
d = pd.DataFrame(rows, columns=["stream", "seed", "method", "metric", "delta"])
out = (d.groupby(["stream", "method", "metric"]).delta
       .agg(mean="mean", sd=lambda v: v.std(ddof=1), n_below=lambda v: int((v < 0).sum()), n="count").reset_index())
out["se"] = out["sd"] / np.sqrt(out["n"])
out.to_csv(ROOT / "results" / f"phase_a_frontier{suffix}.csv", index=False)
pd.set_option("display.width", 200)
print(out[out.metric == "post_shift_viol"].round(4).to_string(index=False))
