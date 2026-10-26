"""Frontier gaps and paired contrasts for real-output streams. Usage: python analyze_real.py mmlu|truthfulqa|rtp

The frontier is the Adaptive CRC window-sweep curve of post-shift violation against mean threshold. A method's gap is read off that
curve by linear interpolation at the method's own mean threshold; outside the range of the sweep the curve is clipped to its nearest
end (np.interp). `inside` marks the units whose mean threshold lies inside the sweep range; the *_inside columns use only those."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
# forms: mmlu | truthfulqa | rtp | rtp:qwen05b | rtp:gpt2:0.05
parts = sys.argv[1].split(":")
task = parts[0]
tag = (parts[1] if len(parts) > 1 else ("gpt2" if task == "rtp" else "Qwen2.5-1.5B-Instruct"))
asfx = f"_alpha{parts[2]}" if len(parts) > 2 else ""
default = len(parts) == 1
name = task if default else f"{task}_{tag}{asfx}"
r = pd.read_csv(ROOT / "results" / f"real_{task}_{tag}{asfx}_runs.csv")
r["family"] = r.stream.str.replace(r"_\d+$", "", regex=True)
sweep = [m for m in r.method.unique() if m.startswith("adaptive_w")]
targets = ["promptshift_crc", "no_representation_weighting", "no_drift_gate", "no_recent_fallback", "rep_aci",
           "adaptive_crc", "adaptive_crc_matched", "dtaci", "dsr_aci", "pid_quantile_tracking"]
rows, con = [], []
for (fam, risk, st), g in r.groupby(["family", "risk", "stream"]):
    ymets = ["post_shift_viol"] if g.post_shift_viol.notna().any() else []
    ymets += ["max_local_viol"]
    ad = g[g.method.isin(sweep)].sort_values("mean_threshold")
    x = ad.mean_threshold.values
    for ymet in ymets:
        y = ad[ymet].values
        for m in targets:
            q = g[g.method == m].iloc[0]
            rows.append((fam, risk, st, m, ymet, q[ymet] - np.interp(q.mean_threshold, x, y), bool(x.min() <= q.mean_threshold <= x.max())))
    p = g[g.method == "promptshift_crc"].iloc[0]; c = g[g.method == "adaptive_crc_matched"].iloc[0]
    for met in ["abs_err", "post_shift_viol", "max_local_viol", "mean_threshold"]:
        con.append((fam, risk, st, met, p[met] - c[met]))
d = pd.DataFrame(rows, columns=["family", "risk", "stream", "method", "metric", "delta", "inside"])
out = d.groupby(["family", "risk", "method", "metric"]).delta.agg(mean="mean", sd=lambda v: v.std(ddof=1) if len(v) > 1 else np.nan,
                                                               n_below=lambda v: int((v < 0).sum()), n="count").reset_index()
out["se"] = out["sd"] / np.sqrt(out["n"])
ins = d[d.inside].groupby(["family", "risk", "method", "metric"]).delta.agg(mean_inside="mean", sd_inside=lambda v: v.std(ddof=1) if len(v) > 1 else np.nan,
                                                                          n_inside="count").reset_index()
out = out.merge(ins, how="left", on=["family", "risk", "method", "metric"])
out["se_inside"] = out["sd_inside"] / np.sqrt(out["n_inside"])
out["n_inside"] = out["n_inside"].fillna(0).astype(int)
out.to_csv(ROOT / "results" / f"real_{name}_frontier.csv", index=False)
cdf = pd.DataFrame(con, columns=["family", "risk", "stream", "metric", "delta"])
c2 = cdf.groupby(["family", "risk", "metric"]).delta.agg(mean="mean", sd=lambda v: v.std(ddof=1) if len(v) > 1 else np.nan,
                                                        n_lower=lambda v: int((v < 0).sum()), n="count").reset_index()
c2["se"] = c2["sd"] / np.sqrt(c2["n"])
c2.to_csv(ROOT / "results" / f"real_{name}_ps_vs_matched.csv", index=False)
pd.set_option("display.width", 200)
print("== frontier gaps (negative = better than Adaptive CRC at the same sharpness), prob risk")
print(out[(out.risk == "prob") & (out.metric != "max_local_viol") & out.method.isin(["promptshift_crc", "no_representation_weighting", "dtaci", "dsr_aci", "no_drift_gate"])].round(4).to_string(index=False))
print("\n== PromptShift minus matched Adaptive, prob risk")
print(c2[c2.risk == "prob"].round(4).to_string(index=False))
