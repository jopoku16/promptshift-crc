"""Round 4, item E: L-ARC against the Adaptive CRC frontier and against PromptShift-CRC.

Reads results/round4_larc_*.csv (run_round4_larc.py) and the existing runs files (the window sweep, PromptShift-CRC and DtACI are
already there for the same streams). Writes results/round4_larc_summary.csv. Pre-declared in PREREGISTRATION_ROUND4_2026-10-26.md, Amendment 2.
Primary configuration: larc_eta1_ellDefault. All four are reported.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SETS = [("synthetic", None), ("mmlu", "Qwen2.5-1.5B-Instruct"), ("rtp_gpt2", "gpt2"), ("rtp_qwen05b", "qwen05b"), ("truthfulqa", "Qwen2.5-1.5B-Instruct")]
rows = []


def summarise(label, risk, fam, g_units):
    for cfg, gu in g_units.groupby("config"):
        n = len(gu); gi = gu[gu.inside]
        rows.append(dict(set=label, family=fam, risk=risk, config=cfg, n=n, coverage=gu.coverage.mean(), gap=gu.gap.mean(), gap_se=gu.gap.std(ddof=1) / np.sqrt(n),
                         n_inside=len(gi), gap_inside=gi.gap.mean() if len(gi) else np.nan,
                         gap_inside_se=gi.gap.std(ddof=1) / np.sqrt(len(gi)) if len(gi) > 1 else np.nan,
                         ps_gap=gu.ps_gap.mean(), minus_ps=gu.post_minus_ps.mean(), minus_ps_se=gu.post_minus_ps.std(ddof=1) / np.sqrt(n), n_lower_than_ps=int((gu.post_minus_ps < 0).sum()),
                         thr_minus_ps=gu.thr_minus_ps.mean(), rank_among_windows=gu["rank"].mean()))


for label, tag in SETS:
    fn = RES / (f"round4_larc_synthetic.csv" if label == "synthetic" else f"round4_larc_{label.split('_')[0]}_{tag if label != 'rtp_gpt2' and label != 'rtp_qwen05b' else tag}.csv")
    if label.startswith("rtp"):
        fn = RES / f"round4_larc_rtp_{tag}.csv"
    if not fn.exists():
        print("missing", fn.name)
        continue
    L = pd.read_csv(fn)
    if label == "synthetic":
        base = pd.concat([pd.read_csv(RES / "phase_a_synthetic_runs.csv"), pd.read_csv(RES / "phase_a_baselines_runs.csv")], ignore_index=True)
        key = ["stream", "seed"]
        L["risk"] = "prob"; base["risk"] = "prob"
    else:
        base = pd.read_csv(RES / (f"real_{label.split('_')[0]}_{tag}_runs.csv" if not label.startswith("rtp") else f"real_rtp_{tag}_runs.csv"))
        key = ["stream", "risk"]
    base["fam"] = base.stream if label == "synthetic" else base.stream.str.replace(r"_\d+$", "", regex=True)
    L["fam"] = L.stream if label == "synthetic" else L.stream.str.replace(r"_\d+$", "", regex=True)
    units = []
    for k, g in base.groupby(key):
        sw = g[g.method.str.startswith("adaptive_w") | ((label == "synthetic") & g.method.isin(["adaptive_crc", "adaptive_crc_matched"]))].sort_values("mean_threshold")
        x, y = sw.mean_threshold.values, sw.post_shift_viol.values
        if not np.isfinite(y).all() or len(x) < 2:
            continue
        ps = g[g.method == "promptshift_crc"].iloc[0]
        ps_gap = ps.post_shift_viol - np.interp(ps.mean_threshold, x, y)
        win = g[g.method.str.startswith("adaptive_w")]
        lm = L[(L[key[0]] == k[0]) & (L[key[1]] == k[1])]
        for _, q in lm.iterrows():
            gap = q.post_shift_viol - np.interp(q.mean_threshold, x, y)
            rank = 1 + (win.post_shift_viol.values < q.post_shift_viol - 1e-12).sum() + 0.5 * (np.abs(win.post_shift_viol.values - q.post_shift_viol) <= 1e-12).sum()
            units.append(dict(fam=q.fam, risk=q.get("risk", "prob"), config=q.method, coverage=q.coverage, gap=gap, inside=bool(x.min() <= q.mean_threshold <= x.max()), ps_gap=ps_gap,
                              post_minus_ps=q.post_shift_viol - ps.post_shift_viol, thr_minus_ps=q.mean_threshold - ps.mean_threshold, rank=rank))
    u = pd.DataFrame(units)
    for (fam, risk), g in u.groupby(["fam", "risk"]):
        summarise(label, risk, fam, g)
out = pd.DataFrame(rows)
out.to_csv(RES / "round4_larc_summary.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
if not out.empty:
    show = out[(out.risk == "prob") & ~out.family.str.contains("iid|pc1|length|prompt_tox", regex=True)]
    print(show.round(4)[["set", "family", "config", "n", "coverage", "gap", "gap_se", "n_inside", "gap_inside", "ps_gap", "minus_ps", "minus_ps_se", "n_lower_than_ps", "thr_minus_ps", "rank_among_windows"]].to_string(index=False))
