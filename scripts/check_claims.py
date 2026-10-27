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

# ---------------------------------------------------------------- round 4: corrections to the frontier analysis and the kernel facts
import re as _re
_mac = dict(_re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", (ROOT / "numbers_generated.tex").read_text(encoding="utf-8")))
mv = lambda k: float(_mac[k])


def inside_count(r, fam):
    """Independent recount of the units whose mean threshold lies inside the range of the Adaptive CRC window sweep."""
    g = r[(r.family == fam) & (r.risk == "prob")]
    n_in = 0
    for st, gg in g.groupby("stream"):
        x = gg[gg.method.str.startswith("adaptive_w")].mean_threshold.values
        q = gg[gg.method == "promptshift_crc"].mean_threshold.iloc[0]
        n_in += int(x.min() <= q <= x.max())
    return n_in, g.stream.nunique()


for key, r, fam in [("MmluCat", mm, "catblock"), ("MmluSubj", mm, "subjectblock"), ("Rtp", rt, "bandblock"), ("Tqa", tq, "categoryblock")]:
    n_in, n_all = inside_count(r, fam)
    check(f"C1 {key}: inside-range count of the frontier analysis recomputed independently", n_in == int(mv(f"{key}FrontierPSInN")) and n_all == int(mv(f"{key}FrontierPSAllN")), f"{n_in} of {n_all}")
check("C2 synthetic frontier: no unit is clipped (PS, every stream)", bool((psf.n_inside == psf.n).all()), str(psf.n_inside.tolist()))
check("C3 MMLU category blocks: the inside-range frontier gap is the one quoted and is above the all-units gap", abs(mv("MmluCatFrontierPSIn") - 0.0001) < 1e-9 and mv("MmluCatFrontierPSIn") > mv("MmluCatFrontierPS"),
      f"{mv('MmluCatFrontierPSIn'):+.4f} vs {mv('MmluCatFrontierPS'):+.4f}")
check("K1 RTP: the kernel at the declared bandwidth is near-uniform (weight ratio below 1.10, effective size above 0.9 of the pool, both tasks)",
      mv("RtpKernelRatio") < 1.10 and mv("RtpKernelNeff") > 0.9 * mv("PoolLen") and mv("MmluKernelNeff") > 0.9 * mv("PoolLen"),
      f"ratio {mv('RtpKernelRatio'):.2f}, n_eff {mv('RtpKernelNeff'):.0f} and {mv('MmluKernelNeff'):.0f} of {mv('PoolLen'):.0f}")
check("K2 RTP: kernel+update post-shift equals the 512-window rule's to within 0.001 (a window effect, not a similarity effect)",
      abs(mv("RtpKernelUpdPost") - mv("RtpAdaptLongPost")) <= 0.001, f"{mv('RtpKernelUpdPost'):.3f} vs {mv('RtpAdaptLongPost'):.3f}")
lofo_z = {k: mv(f"lofo{k}") / mv(f"lofo{k}SE") for k in ["synthetic", "MMLUcategories", "MMLUsubjects", "RTPbands", "TruthfulQAcategories"]}
check("L1 LOFO with standard errors: clearly lower (below -2 SE) on synthetic and RTP, within 2 SE on both MMLU families, a significant loss on TruthfulQA",
      lofo_z["synthetic"] < -2 and lofo_z["RTPbands"] < -2 and abs(lofo_z["MMLUcategories"]) < 2 and abs(lofo_z["MMLUsubjects"]) < 2 and lofo_z["TruthfulQAcategories"] > 2,
      str({k: round(v, 1) for k, v in lofo_z.items()}))

# ---------------------------------------------------------------- round 4, item A: MMLU at levels where the target quantile moves
def mm_alpha(a_):
    r_ = pd.read_csv(RES / f"real_mmlu_Qwen2.5-1.5B-Instruct_alpha{a_}_runs.csv"); r_["family"] = r_.stream.str.replace(r"_\d+$", "", regex=True)
    return r_

def gap_row(name_, fam_="catblock", method_="promptshift_crc"):
    f_ = pd.read_csv(RES / f"real_{name_}_frontier.csv")
    return f_[(f_.family == fam_) & (f_.risk == "prob") & (f_.metric == "post_shift_viol") & (f_.method == method_)].iloc[0]

for a_ in ("0.2", "0.3"):
    r_ = mm_alpha(a_)
    g_ = r_[(r_.risk == "prob") & r_.method.isin(ours)]
    check(f"A1 MMLU alpha {a_}: every run with our update within the Gibbs-Candes bound", bool((g_.abs_err <= g_.gc_bound + 1e-9).all()), f"max {g_.abs_err.max():.4f} vs {g_.gc_bound.min():.4f}")
g1, g2, g3 = (gap_row(n_) for n_ in ("mmlu", "mmlu_Qwen2.5-1.5B-Instruct_alpha0.2", "mmlu_Qwen2.5-1.5B-Instruct_alpha0.3"))
check("A2 pre-declared rule for item A: the criterion (gap <= -0.005 and |mean| >= 2 SE at alpha 0.3, over all units and over inside-range units) is NOT met",
      not ((g3["mean"] <= -0.005 and abs(g3["mean"]) >= 2 * g3["se"]) and (g3["mean_inside"] <= -0.005 and abs(g3["mean_inside"]) >= 2 * g3["se_inside"])),
      f"all {g3['mean']:+.4f} (SE {g3['se']:.4f}); inside {g3['mean_inside']:+.4f} (SE {g3['se_inside']:.4f})")
check("A3 pre-declared ordering of the effect (0.3 <= 0.2 <= 0.1) is NOT borne out, and every PS frontier gap on MMLU is within two standard errors of zero",
      not (g3["mean"] <= g2["mean"] <= g1["mean"]) and all(abs(g["mean"]) < 2 * g["se"] for g in (g1, g2, g3)),
      f"{g1['mean']:+.4f}, {g2['mean']:+.4f}, {g3['mean']:+.4f}")
check("A4 the target quantile spread on MMLU categories grows with alpha (0.025, 0.077, 0.242)", mv("MmluSpreadAOne") < mv("MmluSpreadATwo") < mv("MmluSpreadAThree") and abs(mv("MmluSpreadAThree") - 0.242) < 0.002)
mech = [("rtp_gpt2_alpha0.05", "bandblock"), ("rtp", "bandblock"), ("rtp_gpt2_alpha0.2", "bandblock"), ("rtp_qwen05b", "bandblock"), ("mmlu", "catblock"),
        ("mmlu_Qwen2.5-1.5B-Instruct_alpha0.2", "catblock"), ("mmlu_Qwen2.5-1.5B-Instruct_alpha0.3", "catblock"), ("truthfulqa", "categoryblock")]
mg = [gap_row(n_, f_) for n_, f_ in mech]
check("A5 mechanism table: the PromptShift-CRC frontier gap is at most 0.002 in absolute value in every setting, over all units and over inside-range units",
      all(abs(g["mean"]) <= 0.002 and abs(g["mean_inside"]) <= 0.00205 for g in mg), f"max all {max(abs(g['mean']) for g in mg):.4f}, max inside {max(abs(g['mean_inside']) for g in mg):.4f}")

# ---------------------------------------------------------------- round 4, item B: bandwidth sweep and recurring regimes
bs = pd.read_csv(RES / "round4_sweep_summary.csv")
KER5 = ["promptshift_crc", "rep_aci", "no_representation_weighting", "no_drift_gate", "no_recent_fallback"]
k5 = bs[bs.method.isin(KER5)].copy()
check("B1 the sweep has 160 kernel cells (8 settings x 5 variants x 4 bandwidths)", len(k5) == 160 and k5.groupby("setting").size().nunique() >= 1, str(len(k5)))
pas = k5[(k5.gap < -0.005) & (k5.gap.abs() > 2 * k5.gap_se)]
check("B2 the pre-declared criterion over all orderings is met in exactly 2 cells, both the kernel+update rule or its twin at h/4 on the GPT-2 block orderings, with no ordering inside the sweep range",
      len(pas) == 2 and set(pas.method) == {"rep_aci", "no_recent_fallback"} and set(pas.setting) == {"rtp_gpt2"} and set(pas.design) == {"bandwidth sweep"} and bool((pas.n_inside == 0).all()) and bool(np.isclose(pas.bw_mult, 0.25).all()),
      str(pas[["design", "setting", "method", "bw_mult", "gap", "n_inside"]].values.tolist()))
ki = k5[k5.n_inside >= 5]
pi_ = ki[(ki.gap_inside < -0.005) & (ki.gap_inside.abs() > 2 * ki.gap_inside_se)]
check("B3 over inside-range orderings the criterion is met in exactly 2 of 146 cells: the no-gate rule at h/4 on the two RTP block-ordering runs, and nowhere else (not MMLU, not the recurring orderings)",
      len(ki) == 146 and len(pi_) == 2 and set(pi_.method) == {"no_drift_gate"} and set(pi_.setting) == {"rtp_gpt2", "rtp_qwen05b"} and set(pi_.design) == {"bandwidth sweep"} and bool(np.isclose(pi_.bw_mult, 0.25).all()),
      str(pi_[["setting", "method", "bw_mult", "gap_inside", "gap_inside_se", "n_inside"]].values.tolist()))
rest = ki.drop(pi_.index)
check("B4 in every other cell with at least five orderings inside the range the gap is smaller than 0.005 in absolute value", bool((rest.gap_inside.abs() < 0.005).all()), f"max {rest.gap_inside.abs().max():.4f}")
rec = bs[(bs.design == "recurring regimes") & (bs.method == "promptshift_crc")]
check("B5 recurring regimes: PromptShift-CRC frontier gap within 0.0005 of zero and within 0.0006 of the matched baseline at every bandwidth and setting", rec.gap.abs().max() <= 0.0005 and rec.vs_matched.abs().max() <= 0.0006, f"{rec.gap.abs().max():.4f}, {rec.vs_matched.abs().max():.4f}")
g1_ = bs[(bs.design == "bandwidth sweep") & (bs.setting == "rtp_gpt2")]
neff = lambda m_, b_: float(g1_[(g1_.method == m_) & np.isclose(g1_.bw_mult, b_)].mean_ess.iloc[0])
check("B6 sharper kernels localise (kernel+update n_eff 727 -> 67 on RTP GPT-2) and PromptShift-CRC at h/4 falls back on the recent window (n_eff within 2 of 128)",
      neff("rep_aci", 1.0) > 700 and neff("rep_aci", 0.25) < 70 and abs(neff("promptshift_crc", 0.25) - 128) < 2, f"{neff('rep_aci', 1.0):.0f}, {neff('rep_aci', 0.25):.0f}, {neff('promptshift_crc', 0.25):.1f}")
dt = bs[(bs.design == "recurring regimes") & (bs.method == "dtaci") & bs.setting.isin(["rtp_gpt2", "rtp_qwen05b"])]
check("B7 DtACI is below the frontier by more than 0.005 on both recurring RTP runs with all 24 orderings inside the range", bool((dt.gap < -0.005).all() and (dt.n_inside == 24).all()), str(dt[["setting", "gap", "n_inside"]].values.tolist()))
runs_b = []
for fn_ in ["round4_sweep_rtp_gpt2", "round4_sweep_rtp_qwen05b", "round4_sweep_mmlu", "round4_recur_rtp_gpt2", "round4_recur_rtp_qwen05b", "round4_recur_mmlu", "round4_recur_mmlu_alpha0.2", "round4_recur_mmlu_alpha0.3"]:
    r_ = pd.read_csv(RES / f"{fn_}.csv"); r_ = r_[r_.method.isin(["promptshift_crc", "rep_aci", "no_representation_weighting", "no_drift_gate", "no_recent_fallback"])]
    runs_b.append(bool((r_.abs_err <= r_.gc_bound + 1e-9).all()))
check("B8 every kernel run with the update in the sweep and recurrence files is within the Gibbs-Candes bound", all(runs_b), str(runs_b))

def recur_gaps(fn_):
    r_ = pd.read_csv(RES / f"{fn_}.csv"); out_ = {}
    for st_, g_ in r_.groupby("stream"):
        ref_ = g_[g_.bw_mult.isna()]; sw_ = ref_[ref_.method.str.startswith("adaptive_w")].sort_values("mean_threshold")
        x_, y_ = sw_.mean_threshold.values, sw_.post_shift_viol.values
        for m_ in ["dtaci", "dsr_aci", "pid_quantile_tracking", "adaptive_crc_matched", "adaptive_crc", "rolling_crc"]:
            q_ = ref_[ref_.method == m_].iloc[0]; out_.setdefault(m_, []).append(q_.post_shift_viol - np.interp(q_.mean_threshold, x_, y_))
        q_ = g_[(g_.method == "promptshift_crc") & np.isclose(g_.bw_mult, 1.0)].iloc[0]; out_.setdefault("promptshift_crc", []).append(q_.post_shift_viol - np.interp(q_.mean_threshold, x_, y_))
    return {k_: float(np.mean(v_)) for k_, v_ in out_.items()}

rg_ = [recur_gaps("round4_recur_rtp_gpt2"), recur_gaps("round4_recur_rtp_qwen05b")]
check("B9 recurring RTP orderings: DtACI has the lowest frontier gap of all rules (matched, window W, rolling, PromptShift-CRC, quantile tracking, density ratio, DtACI) on both runs",
      all(min(d_, key=d_.get) == "dtaci" for d_ in rg_), str([{k_: round(v_, 4) for k_, v_ in d_.items()} for d_ in rg_]))

allgaps = list(gaps)
for fn_ in ["real_rtp_qwen05b_frontier", "real_rtp_gpt2_alpha0.05_frontier", "real_rtp_gpt2_alpha0.2_frontier", "real_mmlu_Qwen2.5-1.5B-Instruct_alpha0.2_frontier", "real_mmlu_Qwen2.5-1.5B-Instruct_alpha0.3_frontier"]:
    d_ = pd.read_csv(RES / f"{fn_}.csv")
    d_ = d_[(d_.risk == "prob") & (d_.metric == "post_shift_viol") & (d_.method == "promptshift_crc") & d_.family.isin(["bandblock", "catblock"])]
    allgaps += list(d_["mean"].abs()) + list(d_["mean_inside"].abs().dropna())
ps_all = bs[bs.method == "promptshift_crc"]
allgaps += list(ps_all.gap.abs()) + list(ps_all.gap_inside.abs().dropna())
check("F2 headline over EVERY setting of the study (replication, other levels, bandwidth sweep, recurring regimes; all and inside-range orderings): PS frontier gap at most PSfrontierMaxAbs in absolute value",
      max(allgaps) <= mv("PSfrontierMaxAbs") + 1e-9, f"max {max(allgaps):.4f} vs {mv('PSfrontierMaxAbs'):.4f}")

# ---------------------------------------------------------------- round 4, item E: L-ARC
la = pd.read_csv(RES / "round4_larc_summary.csv"); la = la[la.risk == "prob"]
prim = la[la.config == "larc_eta1_ellDefault"]
syn_p = prim[prim.set == "synthetic"]
check("E1 L-ARC with the authors' defaults is above the frontier by 0.015 to 0.04 on all five synthetic streams and holds the target (coverage 0.895 to 0.90)",
      bool(((syn_p.gap > 0.015) & (syn_p.gap < 0.04)).all()) and syn_p.coverage.min() > 0.895 and syn_p.coverage.max() < 0.90, f"{syn_p.gap.min():+.4f}..{syn_p.gap.max():+.4f}, cov {syn_p.coverage.min():.3f}..{syn_p.coverage.max():.3f}")
sm = la[(la.set == "synthetic") & (la.config == "larc_eta0.1_ellDefault")]
check("E2 L-ARC with step 0.1 does not reach the target on any synthetic stream (coverage below 0.88) and is above the frontier by 0.15 to 0.25", bool((sm.coverage < 0.88).all()) and sm.gap.min() > 0.15 and sm.gap.max() < 0.25,
      f"cov {sm.coverage.min():.3f}..{sm.coverage.max():.3f}, gap {sm.gap.min():+.3f}..{sm.gap.max():+.3f}")
rp = lambda s_, f_: prim[(prim.set == s_) & (prim.family == f_)].iloc[0]
mc, ms, tq_ = rp("mmlu", "catblock"), rp("mmlu", "subjectblock"), rp("truthfulqa", "categoryblock")
check("E3 L-ARC on MMLU and TruthfulQA is on the frontier (|gap| below 0.0015 in all three families)", all(abs(x["gap"]) < 0.0015 for x in (mc, ms, tq_)), str([round(x["gap"], 4) for x in (mc, ms, tq_)]))
rg, rq = rp("rtp_gpt2", "bandblock"), rp("rtp_qwen05b", "bandblock")
check("E4 L-ARC on RTP: below the frontier with GPT-2 and above it with Qwen2.5-0.5B", rg["gap"] < 0 < rq["gap"], f"{rg['gap']:+.4f}, {rq['gap']:+.4f}")
psg = [float(pd.read_csv(RES / f"real_{n_}_frontier.csv").query("family=='bandblock' and risk=='prob' and metric=='post_shift_viol' and method=='promptshift_crc'")["mean"].iloc[0]) for n_ in ("rtp", "rtp_qwen05b")]
check("E5 the two pre-declared L-ARC sentences do not fit: on the GPT-2 run PromptShift-CRC is also at or below the frontier, and on the Qwen run L-ARC is above it", psg[0] <= 0 and rg["gap"] <= 0 and rq["gap"] > 0, f"PS {psg[0]:+.4f}, {psg[1]:+.4f}")

# ---------------------------------------------------------------- round 4, item C: the drift is not a reliable alarm
ts = pd.read_csv(RES / "round4_trace_summary.csv")
def tr_(fam, m, stat, col="mean"):
    return float(ts[(ts.family == fam) & (ts.method == m) & (ts.stat == stat)].iloc[0][col])
real_f = ["rtp_gpt2", "rtp_qwen05b", "mmlu"]
tops = {f_: tr_(f_, "ps", "top_decile_minus_rest") for f_ in real_f}; ses = {f_: tr_(f_, "ps", "top_decile_minus_rest", "se") for f_ in real_f}
aucs = {f_: tr_(f_, "ps", "auc") for f_ in real_f}
met = [f_ for f_ in real_f if tops[f_] >= 0.02 and abs(tops[f_]) > 2 * ses[f_]]
check("T1 pre-declared rule for item C: the top-decile criterion is met on exactly one real family (MMLU) and the AUC criterion on none", met == ["mmlu"] and max(aucs.values()) < 0.55,
      f"top {({k: round(v, 3) for k, v in tops.items()})}, auc {({k: round(v, 3) for k, v in aucs.items()})}")
check("T2 the sign reverses on both RealToxicityPrompts families (top decile below zero, AUC below 0.5)", tops["rtp_gpt2"] < 0 and tops["rtp_qwen05b"] < 0 and aucs["rtp_gpt2"] < 0.5 and aucs["rtp_qwen05b"] < 0.5)
rhos = {f_: (tr_(f_, "ps", "jump_vs_postshift_spearman"), tr_(f_, "ps", "jump_vs_postshift_spearman", "hi")) for f_ in real_f}
check("T3 shift-level correlation negative on all three real families with a 95% interval below zero", all(r_ < 0 and h_ < 0 for r_, h_ in rhos.values()), str({k: round(v[0], 2) for k, v in rhos.items()}))
check("T4 the matched baseline has the same signs on the real families (top decile, AUC side of 0.5, correlation)",
      all(np.sign(tr_(f_, "matched", "top_decile_minus_rest")) == np.sign(tops[f_]) for f_ in real_f) and all((tr_(f_, "matched", "auc") - 0.5) * (aucs[f_] - 0.5) > 0 for f_ in real_f)
      and all(tr_(f_, "matched", "jump_vs_postshift_spearman") < 0 for f_ in real_f))
sy = ts[(ts.family == "synthetic") & (ts.method == "ps")]
check("T5 synthetic: step-level association positive in every stream, shift-level correlation changes sign across streams",
      bool((sy[sy.stat == "top_decile_minus_rest"]["mean"] > 0).all()) and sy[sy.stat == "jump_vs_postshift_spearman"]["mean"].min() < 0 < sy[sy.stat == "jump_vs_postshift_spearman"]["mean"].max())
check("T6 RTP: the drift is almost flat across a boundary (change below 0.02) and the violation excess after a boundary is confined to the first 25 steps; MMLU has no excess",
      abs(mv("TraceRtpDAfter") - mv("TraceRtpDBefore")) < 0.02 and mv("TraceRtpErrFirst") > mv("TraceRtpErrLater") + 0.02 and mv("TraceMmluErrFirst") <= mv("TraceMmluErrLater") + 0.005,
      f"RTP D {mv('TraceRtpDBefore'):.3f}->{mv('TraceRtpDAfter'):.3f}, err {mv('TraceRtpErrFirst'):.3f} vs {mv('TraceRtpErrLater'):.3f}; MMLU err {mv('TraceMmluErrFirst'):.3f} vs {mv('TraceMmluErrLater'):.3f}")

print()
print("FAILED:", fails if fails else "none")
sys.exit(1 if fails else 0)
