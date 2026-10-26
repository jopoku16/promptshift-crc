"""Round 4, item C: does the computed drift predict violations? Analyses results/round4_trace_*.pkl.

Statistics pre-declared in PREREGISTRATION_ROUND4_2026-10-26.md (item C, with the amendment of the same date that fixes
PromptShift-CRC's violations as the primary outcome and the matched baseline's as a robustness check):
  (i)   violation rate in the steps whose pre-gate drift is in the top decile of the stream, minus the rate in the other steps
  (ii)  AUC of the pre-gate drift for the violation indicator
  (iii) Spearman correlation, across the shifts of a family, between the jump of the pre-gate drift at the boundary (mean over the
        20 steps after minus mean over the 20 steps before) and the post-shift violation (secondary, no threshold)
"Works": (i) >= 0.02 with |mean| > 2 SE, or (ii) AUC >= 0.55, on at least one real family.
"""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
FAMS = {"rtp_gpt2": "real", "rtp_qwen05b": "real", "mmlu": "real", "synthetic": "synthetic"}
JUMP = 20


def auc(score, y):
    y = y.astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


rows, pairs_rows = [], []
for fam, kind in FAMS.items():
    fn = ROOT / "results" / f"round4_trace_{fam}.pkl"
    if not fn.exists():
        print("missing", fn.name)
        continue
    res = pickle.load(fn.open("rb"))
    groups = {}
    for key, tr in res.items():
        grp = key[0] if isinstance(key, tuple) else "all"  # synthetic: one group per stream, orderings: one group
        groups.setdefault(grp, []).append((key, tr))
    for grp, items in groups.items():
        per = {m: {"top": [], "auc": []} for m in ("ps", "matched")}
        jumps, pv = [], {"ps": [], "matched": []}
        for key, tr in items:
            d = tr["dpre"]
            top = d >= np.quantile(d, 0.9)
            for m, e in (("ps", tr["err_ps"]), ("matched", tr["err_matched"])):
                per[m]["top"].append(e[top].mean() - e[~top].mean())
                per[m]["auc"].append(auc(d, e))
            for b in tr["shifts"]:
                if b - JUMP >= 0 and b + tr["post"] <= len(d):
                    jumps.append(d[b: b + JUMP].mean() - d[b - JUMP: b].mean())
                    pv["ps"].append(tr["err_ps"][b: b + tr["post"]].mean())
                    pv["matched"].append(tr["err_matched"][b: b + tr["post"]].mean())
        for m in ("ps", "matched"):
            for stat in ("top", "auc"):
                v = np.array(per[m][stat], dtype=float)
                v = v[~np.isnan(v)]
                rows.append(dict(family=fam, kind=kind, group=grp, method=m, stat={"top": "top_decile_minus_rest", "auc": "auc"}[stat],
                                 mean=float(v.mean()), se=float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan, n=len(v)))
            if len(jumps) > 3:
                rho = spearmanr(jumps, pv[m]).correlation
                rng = np.random.default_rng(0)
                bs = []
                for _ in range(1000):
                    ii = rng.integers(0, len(jumps), len(jumps))
                    bs.append(spearmanr(np.array(jumps)[ii], np.array(pv[m])[ii]).correlation)
                lo, hi = np.nanpercentile(bs, [2.5, 97.5])
                rows.append(dict(family=fam, kind=kind, group=grp, method=m, stat="jump_vs_postshift_spearman", mean=float(rho), se=np.nan, n=len(jumps), lo=float(lo), hi=float(hi)))
out = pd.DataFrame(rows)
out.to_csv(ROOT / "results" / "round4_trace_summary.csv", index=False)
pd.set_option("display.width", 220)
print(out.round(4).to_string(index=False))
real = out[(out.kind == "real") & (out.method == "ps")]
top = real[real.stat == "top_decile_minus_rest"]
au = real[real.stat == "auc"]
works_top = ((top["mean"] >= 0.02) & (top["mean"].abs() > 2 * top["se"])).any()
works_auc = (au["mean"] >= 0.55).any()
print("\nRULE (PromptShift-CRC outcome, real families): top-decile criterion", bool(works_top), "| AUC criterion", bool(works_auc), "=> works =", bool(works_top or works_auc))
rm = out[(out.kind == "real") & (out.method == "matched")]
print("robustness (matched outcome): top-decile", bool(((rm[rm.stat == 'top_decile_minus_rest']['mean'] >= 0.02) & (rm[rm.stat == 'top_decile_minus_rest']['mean'].abs() > 2 * rm[rm.stat == 'top_decile_minus_rest']['se'])).any()),
      "| AUC", bool((rm[rm.stat == 'auc']['mean'] >= 0.55).any()))
