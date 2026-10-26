"""Round 4, item B: frontier gaps of every kernel variant at every bandwidth (bandwidth sweep and recurring-regime orderings).

Reads results/round4_sweep_*.csv and results/round4_recur_*.csv (written by run_round4_sweep.py), writes results/round4_sweep_summary.csv
and prints the pre-declared kill criterion (PREREGISTRATION_ROUND4_2026-10-26.md, item B): a kernel variant "passes" if its frontier gap is
more negative than -0.005 with |mean| > 2 SE.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
KERNEL = ["promptshift_crc", "rep_aci", "no_representation_weighting", "no_drift_gate", "no_recent_fallback"]
rows = []
for fn in sorted(RES.glob("round4_sweep_*.csv")) + sorted(RES.glob("round4_recur_*.csv")):
    if fn.name == "round4_sweep_summary.csv":
        continue
    r = pd.read_csv(fn)
    design = "bandwidth sweep" if fn.name.startswith("round4_sweep") else "recurring regimes"
    setting = fn.stem.replace("round4_sweep_", "").replace("round4_recur_", "")
    units = []
    for st, g in r.groupby("stream"):
        ref = g[g.bw_mult.isna()]
        sw = ref[ref.method.str.startswith("adaptive_w")].sort_values("mean_threshold")
        x, y = sw.mean_threshold.values, sw.post_shift_viol.values
        matched = ref[ref.method == "adaptive_crc_matched"].iloc[0]
        for _, q in g[g.bw_mult.notna()].iterrows():
            units.append(dict(stream=st, method=q.method, bw_mult=q.bw_mult, gap=q.post_shift_viol - np.interp(q.mean_threshold, x, y),
                              inside=bool(x.min() <= q.mean_threshold <= x.max()), vs_matched=q.post_shift_viol - matched.post_shift_viol,
                              thr_vs_matched=q.mean_threshold - matched.mean_threshold, ess=q.mean_ess, abs_err=q.abs_err, gc=q.gc_bound))
        d_ref = ref[ref.method == "dtaci"]
        if not d_ref.empty:
            q = d_ref.iloc[0]
            units.append(dict(stream=st, method="dtaci", bw_mult=np.nan, gap=q.post_shift_viol - np.interp(q.mean_threshold, x, y),
                              inside=bool(x.min() <= q.mean_threshold <= x.max()), vs_matched=q.post_shift_viol - matched.post_shift_viol,
                              thr_vs_matched=q.mean_threshold - matched.mean_threshold, ess=np.nan, abs_err=q.abs_err, gc=q.gc_bound))
    u = pd.DataFrame(units)
    u["bw_mult"] = u.bw_mult.fillna(-1.0)
    for (m, b), g in u.groupby(["method", "bw_mult"]):
        n = len(g); gi = g[g.inside]
        rows.append(dict(design=design, setting=setting, method=m, bw_mult=b, n=n, gap=g.gap.mean(), gap_se=g.gap.std(ddof=1) / np.sqrt(n),
                         n_inside=len(gi), gap_inside=gi.gap.mean() if len(gi) else np.nan,
                         gap_inside_se=gi.gap.std(ddof=1) / np.sqrt(len(gi)) if len(gi) > 1 else np.nan,
                         vs_matched=g.vs_matched.mean(), vs_matched_se=g.vs_matched.std(ddof=1) / np.sqrt(n), n_lower=int((g.vs_matched < 0).sum()),
                         thr_vs_matched=g.thr_vs_matched.mean(), mean_ess=g.ess.mean(), max_abs_err=g.abs_err.max(), min_bound=g.gc.min()))
out = pd.DataFrame(rows)
out.to_csv(RES / "round4_sweep_summary.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
if out.empty:
    print("no sweep files yet")
else:
    print(out[out.method.isin(["promptshift_crc", "rep_aci", "dtaci"])].round(4)[["design", "setting", "method", "bw_mult", "n", "gap", "gap_se", "n_inside", "gap_inside", "gap_inside_se", "vs_matched", "vs_matched_se", "n_lower", "mean_ess"]].to_string(index=False))
    k = out[out.method.isin(KERNEL)].copy()
    k["z"] = k.gap / k.gap_se
    k["passes"] = (k.gap < -0.005) & (k.gap.abs() > 2 * k.gap_se)
    k["passes_inside"] = k.passes & (k.gap_inside < -0.005) & (k.gap_inside.abs() > 2 * k.gap_inside_se)
    print("\nPRE-DECLARED RULE: cells with a frontier gap below -0.005 and |mean| > 2 SE:", int(k.passes.sum()), "of", len(k), "| also inside the sweep range:", int(k.passes_inside.sum()))
    print("most negative cells:")
    print(k.sort_values("gap").head(6).round(4)[["design", "setting", "method", "bw_mult", "gap", "gap_se", "gap_inside", "n_inside", "n"]].to_string(index=False))
    print("Gibbs-Candes: max abs_err over every kernel run with the update vs the bound:", round(k[k.method != "no_online_update"].max_abs_err.max(), 4), round(k.min_bound.min(), 4))
