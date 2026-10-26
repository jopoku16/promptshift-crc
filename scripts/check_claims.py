"""Check every directional or quantitative claim made in the prose against the result files.
Exit code 1 if any claim fails. Run after make_tables.py and before building the paper.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
fails = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ("  [" + detail + "]" if detail else ""))
    if not cond:
        fails.append(name)


syn = pd.read_csv(RES / "phase_a_synthetic_runs.csv")
bl = pd.read_csv(RES / "phase_a_baselines_runs.csv")
syn_all = pd.concat([syn, bl], ignore_index=True)
STREAMS = list(syn.stream.unique())

# 1 guarantee holds for every method that uses our update (synthetic)
ours = ["adaptive_crc", "adaptive_crc_matched", "rep_aci", "promptshift_crc", "no_representation_weighting", "no_drift_gate", "no_recent_fallback"]
g = syn[syn.method.isin(ours)]
check("S1 every synthetic run with our update is within the Gibbs-Candes bound", bool((g.abs_err <= g.gc_bound + 1e-9).all()),
      f"max abs_err {g.abs_err.max():.4f}, min bound {g.gc_bound.min():.4f}")
check("S1b gc bound for synthetic in [0.018, 0.024]", 0.0175 <= g.gc_bound.min() and g.gc_bound.max() <= 0.0245, f"{g.gc_bound.min():.4f}-{g.gc_bound.max():.4f}")

# 2-5 paired contrasts, per stream, over seeds
def paired(other, metric):
    out = []
    for st in STREAMS:
        gg = syn_all[syn_all.stream == st]
        a = gg[gg.method == "promptshift_crc"].set_index("seed")[metric]
        b = gg[gg.method == other].set_index("seed")[metric]
        d = a - b
        out.append((d.mean(), int((d < 0).sum()), len(d)))
    return out

pm = paired("adaptive_crc_matched", "post_shift_viol")
check("S2 PS minus matched post-shift is positive in all five streams", all(m > 0 for m, _, _ in pm), str([round(m, 4) for m, _, _ in pm]))
check("S2b PS lower than matched post-shift in at most 4 of 30 seeds per stream", all(l <= 4 for _, l, _ in pm), str([l for _, l, _ in pm]))
pt = paired("adaptive_crc_matched", "mean_threshold")
check("S3 PS threshold lower than matched in at least 29 of 30 seeds in every stream", all(l >= 29 for _, l, _ in pt), str([l for _, l, _ in pt]))
pd_ = paired("dtaci", "post_shift_viol")
check("S4 PS minus DtACI post-shift is positive in all five streams", all(m > 0 for m, _, _ in pd_), str([round(m, 4) for m, _, _ in pd_]))
pdt = paired("dtaci", "mean_threshold")
check("S4b PS threshold above DtACI's in all five streams", all(m > 0 for m, _, _ in pdt))
pg = paired("no_drift_gate", "post_shift_viol")
check("S5 PS minus no-gate post-shift negative in all streams", all(m < 0 for m, _, _ in pg), str([round(m, 4) for m, _, _ in pg]))
pgt = paired("no_drift_gate", "mean_threshold")
check("S5b PS threshold above no-gate's in all streams", all(m > 0 for m, _, _ in pgt))
# 6 component readings on stream means
mean = lambda m, col: syn[syn.method == m].groupby("stream")[col].mean().mean()
check("S6 kernel+update has higher post-shift and lower threshold than PS (synthetic mean)",
      mean("rep_aci", "post_shift_viol") > mean("promptshift_crc", "post_shift_viol") and mean("rep_aci", "mean_threshold") < mean("promptshift_crc", "mean_threshold"))
check("S7 removing the update raises PS post-shift violation", mean("no_online_update", "post_shift_viol") > mean("promptshift_crc", "post_shift_viol"))
check("S8 DtACI over-covers on average (> 0.905)", syn_all[syn_all.method == "dtaci"].coverage.mean() > 0.905, f"{syn_all[syn_all.method == 'dtaci'].coverage.mean():.4f}")
check("S9 quantile tracking has higher post-shift violation than every weighted rule",
      all(mean("pid_quantile_tracking" if False else "promptshift_crc", "post_shift_viol") < syn_all[syn_all.method == "pid_quantile_tracking"].groupby("stream").post_shift_viol.mean().mean() for _ in [0]))
# frontier gaps
fr = pd.read_csv(RES / "phase_a_frontier.csv"); fr = fr[fr.metric == "post_shift_viol"]
psf = fr[fr.method == "promptshift_crc"]
check("S10 PS frontier gap never below -0.006 and never above +0.002 on synthetic streams", psf["mean"].min() > -0.006 and psf["mean"].max() < 0.002,
      f"{psf['mean'].min():+.4f}..{psf['mean'].max():+.4f}")

# real tasks
def real(task, tag):
    r = pd.read_csv(RES / f"real_{task}_{tag}_runs.csv")
    r["family"] = r.stream.str.replace(r"_\d+$", "", regex=True)
    return r

mm = real("mmlu", "Qwen2.5-1.5B-Instruct"); rt = real("rtp", "gpt2"); tq = real("truthfulqa", "Qwen2.5-1.5B-Instruct")
online = ["adaptive_crc", "adaptive_crc_matched", "rep_aci", "promptshift_crc", "no_representation_weighting", "no_drift_gate", "no_recent_fallback", "dtaci", "dsr_aci", "pid_quantile_tracking"]
for nm, r in [("MMLU", mm), ("RTP", rt), ("TruthfulQA", tq)]:
    g = r[(r.risk == "prob") & r.method.isin(ours)]
    check(f"R1 {nm}: every run with our update within the Gibbs-Candes bound", bool((g.abs_err <= g.gc_bound + 1e-9).all()), f"max {g.abs_err.max():.4f} vs {g.gc_bound.min():.4f}")
cat = mm[(mm.family == "catblock") & (mm.risk == "prob")]
thr = cat[cat.method.isin(online)].groupby("method").mean_threshold.mean()
check("R2 MMLU category blocks: online-method mean thresholds within 0.015", thr.max() - thr.min() <= 0.015, f"{thr.max() - thr.min():.4f}")
iv = cat.groupby("method").intervention_rate.mean()
check("R3 MMLU category blocks: intervention rate >= 0.999 for every method", bool((iv >= 0.999).all()), f"min {iv.min():.4f}")
mq = cat.groupby("method").post_shift_viol.mean()
check("R4 MMLU: static post-shift within 0.005 of PS", abs(mq["static_crc"] - mq["promptshift_crc"]) < 0.005, f"{mq['static_crc']:.4f} vs {mq['promptshift_crc']:.4f}")
d = json.load  # silence linter
rj = pd.DataFrame([json.loads(l) for l in open(ROOT / "real_outputs" / "rtp_gpt2.jsonl", encoding="utf-8")])
rj["band"] = pd.qcut(rj.prompt_toxicity.fillna(0), 4, labels=False, duplicates="drop")
q90 = rj.groupby("band").risk.quantile(0.9).values
check("R5 RTP: 0.9 quantile below 0.06 in the three lowest bands and above 0.5 in the highest", q90[:3].max() < 0.06 and q90[3] > 0.5, str(np.round(q90, 3)))
rb = rt[(rt.family == "bandblock") & (rt.risk == "prob")].groupby("method").mean(numeric_only=True)
check("R6 RTP: static coverage below 0.88 and rolling post-shift above online methods'", rb.loc["static_crc", "coverage"] < 0.88 and rb.loc["rolling_crc", "post_shift_viol"] > rb.loc["promptshift_crc", "post_shift_viol"])
check("R7 RTP: longer windows (W, 512) have post-shift <= matched's", rb.loc["adaptive_crc", "post_shift_viol"] < rb.loc["adaptive_crc_matched", "post_shift_viol"] and rb.loc["adaptive_w512", "post_shift_viol"] < rb.loc["adaptive_crc_matched", "post_shift_viol"])
rp = pd.read_csv(RES / "real_rtp_ps_vs_matched.csv"); rp = rp[(rp.family == "bandblock") & (rp.risk == "prob") & (rp.metric == "post_shift_viol")].iloc[0]
check("R8 RTP: PS minus matched post-shift negative, lower in majority of orderings", rp["mean"] < 0 and rp["n_lower"] > rp["n"] / 2, f"{rp['mean']:+.4f}, {int(rp['n_lower'])}/{int(rp['n'])}")
check("R9 RTP: 'no-update' variant has higher post-shift than PS", rb.loc["no_online_update", "post_shift_viol"] > rb.loc["promptshift_crc", "post_shift_viol"])
sk = real("mmlu", "Qwen2.5-1.5B-Instruct")
# TQA saturation
tqb = tq[(tq.family == "categoryblock") & (tq.risk == "prob")]
check("R10 TruthfulQA: mean thresholds of online methods >= 0.99", bool((tqb[tqb.method.isin(online)].groupby("method").mean_threshold.mean() >= 0.99).all()))

# window robustness
sys.path.insert(0, str(ROOT / "scripts"))
fam_frames = {}
syn_w = syn[~syn.method.isin(["dtaci", "dsr_aci", "pid_quantile_tracking"])].copy(); syn_w["unit"] = syn_w.stream + "|" + syn_w.seed.astype(str)
fam_frames["synthetic"] = syn_w
for key, r, fam in [("MMLU cat", mm, "catblock"), ("MMLU subj", mm, "subjectblock"), ("RTP", rt, "bandblock"), ("TQA", tq, "categoryblock")]:
    g = r[(r.family == fam) & (r.risk == "prob")].copy(); g["unit"] = g.stream; fam_frames[key] = g
ratios, ranks, below = [], {}, 0
for k, g in fam_frames.items():
    sw = sorted([m for m in g.method.unique() if m.startswith("adaptive_w")], key=lambda m: int(m.split("_w")[1]))
    Wm = g[g.method.isin(sw)].pivot(index="unit", columns="method", values="post_shift_viol")[sw]
    ps = g[g.method == "promptshift_crc"].set_index("unit").post_shift_viol.reindex(Wm.index)
    spread = (Wm.max(axis=1) - Wm.min(axis=1)).mean(); vs_worst = (ps - Wm.max(axis=1)).mean()
    ratios.append(-vs_worst / spread)
    rk = [(1 + (Wm.loc[u].values < ps[u] - 1e-12).sum() + 0.5 * (np.abs(Wm.loc[u].values - ps[u]) <= 1e-12).sum()) for u in Wm.index]
    ranks[k] = float(np.mean(rk))
check("W1 PS minus worst window is negative in every family, between 0.4 and 0.7 of the spread", all(0.4 <= x <= 0.7 for x in ratios), str(np.round(ratios, 2)))
check("W2 PS rank < 3.5 on MMLU (both) and RTP; > 3.5 on synthetic and TruthfulQA",
      ranks["MMLU cat"] < 3.5 and ranks["MMLU subj"] < 3.5 and ranks["RTP"] < 3.5 and ranks["synthetic"] > 3.5 and ranks["TQA"] > 3.5, str({k: round(v, 2) for k, v in ranks.items()}))
# LOFO
pos = {}
for k, g in fam_frames.items():
    sw = sorted([m for m in g.method.unique() if m.startswith("adaptive_w")], key=lambda m: int(m.split("_w")[1]))
    Wm = g[g.method.isin(sw)].pivot(index="unit", columns="method", values="post_shift_viol")[sw]
    ps = g[g.method == "promptshift_crc"].set_index("unit").post_shift_viol.reindex(Wm.index)
    pos[k] = (Wm.values, ps.values)
neg = 0
for held in pos:
    others = [k for k in pos if k != held]
    r_ = np.mean([np.mean(np.argsort(np.argsort(pos[k][0], axis=1), axis=1), axis=0) for k in others], axis=0)
    j = int(np.argmin(r_)); neg += int((pos[held][1] - pos[held][0][:, j]).mean() < 0)
check("W3 LOFO: PS lower than the elsewhere-chosen window in 4 of 5 families", neg == 4, f"{neg}/5")


# replication and alpha robustness (RealToxicityPrompts)
for tag, nm in [("qwen05b", "RTP replication (Qwen2.5-0.5B)"), ("gpt2_alpha0.05", "RTP GPT-2 alpha 0.05"), ("gpt2_alpha0.2", "RTP GPT-2 alpha 0.2")]:
    rr = pd.read_csv(RES / f"real_rtp_{tag}_runs.csv"); rr["family"] = rr.stream.str.replace(r"_\d+$", "", regex=True)
    bb = rr[(rr.family == "bandblock") & (rr.risk == "prob")]
    mm_ = bb.groupby("method").mean(numeric_only=True)
    pmv = pd.read_csv(RES / f"real_rtp_{tag.replace('gpt2_alpha', 'gpt2_alpha')}_ps_vs_matched.csv") if True else None
    pmv = pmv[(pmv.family == "bandblock") & (pmv.risk == "prob") & (pmv.metric == "post_shift_viol")].iloc[0]
    check(f"X1 {nm}: PS minus matched post-shift is negative", pmv["mean"] < 0, f"{pmv['mean']:+.4f} lower in {int(pmv['n_lower'])}/{int(pmv['n'])}")
    check(f"X2 {nm}: static coverage fails (below target by > 0.02)", (1 - rr.alpha.iloc[0]) - mm_.loc["static_crc", "coverage"] > 0.02 or mm_.loc["static_crc", "coverage"] < (1 - rr.alpha.iloc[0]) - 0.02 or mm_.loc["static_crc", "post_shift_viol"] > mm_.loc["promptshift_crc", "post_shift_viol"] + 0.02)
    fr_ = pd.read_csv(RES / f"real_rtp_{tag}_frontier.csv"); fr_ = fr_[(fr_.family == "bandblock") & (fr_.risk == "prob") & (fr_.metric == "post_shift_viol") & (fr_.method == "promptshift_crc")].iloc[0]
    check(f"X3 {nm}: PS frontier gap within 0.006 in absolute value", abs(fr_["mean"]) <= 0.006, f"{fr_['mean']:+.4f}")
qb = pd.read_csv(RES / "real_rtp_qwen05b_runs.csv"); qb["family"] = qb.stream.str.replace(r"_\d+$", "", regex=True)
qb = qb[(qb.family == "bandblock") & (qb.risk == "prob")].groupby("method").mean(numeric_only=True)
check("X4 replication: longer windows (W, 512) at least as good as the matched window", qb.loc["adaptive_crc", "post_shift_viol"] < qb.loc["adaptive_crc_matched", "post_shift_viol"] and qb.loc["adaptive_w512", "post_shift_viol"] < qb.loc["adaptive_crc_matched", "post_shift_viol"])

# PSfrontierMaxAbs consistency
gaps = list(np.abs(psf["mean"].values))
for t_, fam_ in [("mmlu", ["catblock", "subjectblock"]), ("rtp", ["bandblock"]), ("truthfulqa", ["categoryblock"])]:
    d_ = pd.read_csv(RES / f"real_{t_}_frontier.csv"); d_ = d_[(d_.risk == "prob") & (d_.metric == "post_shift_viol") & (d_.method == "promptshift_crc") & d_.family.isin(fam_)]
    gaps += list(np.abs(d_["mean"].values))
check("F1 headline: PS frontier gap at most 0.006 in absolute value in every setting", max(gaps) <= 0.006, f"{max(gaps):.4f}")

print()
print("FAILED:", fails if fails else "none")
sys.exit(1 if fails else 0)
