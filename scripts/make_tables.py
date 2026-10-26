"""Generate LaTeX tables and number macros for the revised paper from results/*.csv.
Nothing in the paper's results sections is typed by hand: every number comes from here.
Writes tables_generated.tex and numbers_generated.tex in revision_v2/.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
LAB = {
    "static_crc": "Static", "rolling_crc": "Rolling", "adaptive_crc": "Adaptive CRC, window $W$",
    "adaptive_crc_matched": "Adaptive CRC, matched window", "representation_weighted_crc": "Kernel weights, no update",
    "rep_aci": "Kernel weights + update", "promptshift_crc": "PromptShift-CRC",
    "no_representation_weighting": "Gate + recency only", "no_drift_gate": "No gate (fixed mix)",
    "no_recent_fallback": "No recency fallback", "no_online_update": "No online update",
    "dtaci": "DtACI", "dsr_aci": "Density-ratio weights", "pid_quantile_tracking": "Quantile tracking",
}
MAIN = ["static_crc", "rolling_crc", "adaptive_crc", "adaptive_crc_matched", "rep_aci", "representation_weighted_crc",
        "promptshift_crc", "no_representation_weighting", "no_drift_gate", "no_recent_fallback", "no_online_update",
        "dtaci", "dsr_aci", "pid_quantile_tracking"]
STREAM_LAB = {"pilot": "pilot", "ablation": "ablation stream", "proto:qa_topic_shift": "QA topic shift",
              "proto:summarization_domain_shift": "summarization shift", "proto:safety_intent_shift": "safety-intent shift"}
out_tab, out_num = [], []


def f(x, d=3):
    return "--" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.{d}f}"


def pm(m, s, d=3):
    return "--" if np.isnan(m) else (f"{m:.{d}f}" if np.isnan(s) else f"{m:.{d}f} $\\pm$ {s:.{d}f}")


def table(label, caption, header, body, note=None):
    cols = "l" + "r" * (len(header) - 1)
    wide = len(header) >= 6
    s = ["\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}"]
    if wide:
        s.append("\\resizebox{\\textwidth}{!}{%")
    s += [f"\\begin{{tabular}}{{{cols}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    s += [" & ".join(r) + " \\\\" for r in body]
    s += ["\\bottomrule", "\\end{tabular}"]
    if wide:
        s.append("}")
    if note:
        s += [f"\\par\\smallskip{{\\footnotesize {note}}}"]
    s += ["\\end{table}", ""]
    out_tab.append("\n".join(s))


def macro(name, val):
    out_num.append(f"\\newcommand{{\\{name}}}{{{val}}}")


# ---------------------------------------------------------------- synthetic
syn = pd.read_csv(RES / "phase_a_synthetic_runs.csv")
bl = RES / "phase_a_baselines_runs.csv"
if bl.exists():
    syn = pd.concat([syn, pd.read_csv(bl)], ignore_index=True)
nseeds = syn.seed.nunique()
macro("nSeeds", str(nseeds))
rows = []
for m in MAIN:
    g = syn[syn.method == m]
    if g.empty:
        continue
    per = g.groupby("stream").agg(covm=("coverage", "mean"), covsd=("coverage", lambda v: v.std(ddof=1)), err=("abs_err", "mean"),
                                  post=("post_shift_viol", "mean"), postsd=("post_shift_viol", lambda v: v.std(ddof=1)),
                                  ml=("max_local_viol", "mean"), thr=("mean_threshold", "mean"))
    rows.append([LAB[m], pm(per.covm.mean(), per.covsd.mean(), 3), pm(per.post.mean(), per.postsd.mean(), 3),
                 f(per.ml.mean(), 3), f(per.thr.mean(), 3)])
table("tab:synthetic", f"Synthetic streams, mean over five generators of the {nseeds}-seed means (and mean seed-to-seed sd). "
      "Every online method has coverage near $0.90$ because of the update; local violation and sharpness separate the methods.",
      ["Method", "Coverage", "Post-shift violation", "Max local violation", "Mean threshold"], rows,
      note="Post-shift violation is the miss rate over the 150 steps after each regime boundary (target $0.10$).")

# paired contrasts vs matched adaptive, per stream
rows = []
for st, nm in STREAM_LAB.items():
    g = syn[syn.stream == st]
    p = g[g.method == "promptshift_crc"].set_index("seed")
    for other, olab in [("adaptive_crc_matched", "matched Adaptive CRC"), ("no_drift_gate", "no gate"),
                        ("no_representation_weighting", "gate + recency only"), ("dtaci", "DtACI")]:
        c = g[g.method == other].set_index("seed")
        d1 = (p.post_shift_viol - c.post_shift_viol); d2 = (p.mean_threshold - c.mean_threshold)
        rows.append([nm if other == "adaptive_crc_matched" else "", olab, f"{d1.mean():+.4f}", f"{d1.std(ddof=1) / np.sqrt(len(d1)):.4f}",
                     f"{int((d1 < 0).sum())}/{len(d1)}", f"{d2.mean():+.4f}", f"{int((d2 < 0).sum())}/{len(d2)}"])
table("tab:paired", "PromptShift-CRC minus a comparator, paired by seed. Negative means PromptShift-CRC is lower (better for violation; sharper for threshold).",
      ["Stream", "Comparator", "$\\Delta$ post-shift", "SE", "PS lower in", "$\\Delta$ threshold", "PS lower in"], rows)

# frontier gaps
fr = pd.read_csv(RES / "phase_a_frontier.csv")
fr = fr[fr.metric == "post_shift_viol"]
rows = []
for st, nm in STREAM_LAB.items():
    r_ = [nm]
    for m in ["promptshift_crc", "no_representation_weighting", "rep_aci", "dtaci", "dsr_aci"]:
        x = fr[(fr.stream == st) & (fr.method == m)].iloc[0]
        r_.append(f"{x['mean']:+.4f} ({x['se']:.4f})")
    rows.append(r_)
table("tab:frontier_syn", "Frontier gap on the synthetic streams: post-shift violation minus the Adaptive-CRC window-sweep curve interpolated at the method's own mean "
      "threshold, mean over seeds with standard error. Negative means better than Adaptive CRC at the same sharpness.",
      ["Stream", "PromptShift-CRC", "Gate + recency only", "Kernel + update", "DtACI", "Density ratio"], rows)
pf = fr[fr.method == "promptshift_crc"]
lo, hi = pf["mean"].min(), pf["mean"].max()
macro("PSfrontierSynLo", f"{lo:+.4f}"); macro("PSfrontierSynHi", f"{hi:+.4f}")
macro("PSfrontierSynSEmax", f"{pf['se'].max():.4f}")
macro("PSfrontierSynBelowCount", str(int((pf['mean'] < 0).sum())))


# ---------------------------------------------------------------- macros for prose (synthetic)
def rng_macro(prefix, vals, d=4, signed=True):
    fmt = (lambda v: f"{v:+.{d}f}") if signed else (lambda v: f"{v:.{d}f}")
    macro(prefix + "Lo", fmt(min(vals))); macro(prefix + "Hi", fmt(max(vals)))

def contrasts(other, metric):
    vals, lows = [], []
    for st in STREAM_LAB:
        g = syn[syn.stream == st]
        a = g[g.method == "promptshift_crc"].set_index("seed")[metric]; b = g[g.method == other].set_index("seed")[metric]
        d = a - b; vals.append(d.mean()); lows.append(int((d < 0).sum()))
    return vals, lows

for other, tag in [("adaptive_crc_matched", "Matched"), ("no_drift_gate", "NoGate"), ("no_representation_weighting", "GateRec"), ("dtaci", "DtACI")]:
    v, l = contrasts(other, "post_shift_viol"); rng_macro(f"synPost{tag}", v)
    macro(f"synPost{tag}LowerMin", str(min(l))); macro(f"synPost{tag}LowerMax", str(max(l)))
    v, l = contrasts(other, "mean_threshold"); rng_macro(f"synThr{tag}", v)
    macro(f"synThr{tag}LowerMin", str(min(l))); macro(f"synThr{tag}LowerMax", str(max(l)))
for m, tag in [("promptshift_crc", "PS"), ("adaptive_crc_matched", "Matched"), ("dtaci", "DtACI"), ("rep_aci", "KernelUpd"), ("no_online_update", "NoUpd"),
               ("rolling_crc", "Rolling"), ("static_crc", "Static"), ("no_drift_gate", "NoGate"), ("no_representation_weighting", "GateRec")]:
    g = syn[syn.method == m]
    macro(f"syn{tag}Post", f"{g.groupby('stream').post_shift_viol.mean().mean():.3f}")
    macro(f"syn{tag}Cov", f"{g.groupby('stream').coverage.mean().mean():.3f}")
    macro(f"syn{tag}Thr", f"{g.groupby('stream').mean_threshold.mean().mean():.3f}")

# ---------------------------------------------------------------- real streams
def real_table(task, tag, fams, title_prefix, risk="prob", label_sfx=""):
    f_ = RES / f"real_{task}_{tag}_runs.csv"
    if not f_.exists():
        return None
    r = pd.read_csv(f_)
    r["family"] = r.stream.str.replace(r"_\d+$", "", regex=True)
    for fam, famlab in fams:
        g = r[(r.family == fam) & (r.risk == risk)]
        if g.empty:
            continue
        nstream = g.stream.nunique()
        rows = []
        use_post = g.post_shift_viol.notna().any()
        for m in MAIN:
            h = g[g.method == m]
            if h.empty:
                continue
            row = [LAB[m], pm(h.coverage.mean(), h.coverage.std(ddof=1) if nstream > 1 else np.nan, 3)]
            if use_post:
                row.append(pm(h.post_shift_viol.mean(), h.post_shift_viol.std(ddof=1) if nstream > 1 else np.nan, 3))
            row += [f(h.max_local_viol.mean(), 3), f(h.mean_threshold.mean(), 3), f(h.intervention_rate.mean(), 3)]
            rows.append(row)
        hdr = ["Method", "Coverage"] + (["Post-shift violation"] if use_post else []) + ["Max local violation", "Mean threshold", "Intervention rate"]
        table(f"tab:{task}_{fam}{label_sfx}", f"{title_prefix}, {famlab} ({nstream} stream{'s' if nstream > 1 else ''}; mean and sd over streams). Risk: "
              f"{('toxicity probability of the continuation' if task == 'rtp' else 'one minus the probability of the correct answer') if risk == 'prob' else '0/1 error'}.", hdr, rows)
    return r


real_table("mmlu", "Qwen2.5-1.5B-Instruct", [("catblock", "category blocks"), ("subjectblock", "subject blocks"), ("iid", "random order, no shift"),
                                              ("pc1_forward", "embedding-direction order (forward)"), ("pc1_reverse", "embedding-direction order (reversed)"), ("length_ascending", "question-length order")],
           "MMLU with Qwen2.5-1.5B-Instruct")
real_table("rtp", "gpt2", [("bandblock", "prompt-toxicity bands in blocks"), ("iid", "random order, no shift"), ("pc1_forward", "embedding-direction order (forward)"),
                           ("pc1_reverse", "embedding-direction order (reversed)"), ("length_ascending", "prompt-length order"),
                           ("prompt_toxicity_ascending", "ascending prompt toxicity"), ("prompt_toxicity_descending", "descending prompt toxicity")],
           "RealToxicityPrompts with GPT-2 continuations")
real_table("rtp", "qwen05b", [("bandblock", "prompt-toxicity bands in blocks")], "RealToxicityPrompts with Qwen2.5-0.5B continuations (replication)", label_sfx="_qwen")
real_table("truthfulqa", "Qwen2.5-1.5B-Instruct", [("categoryblock", "category blocks"), ("iid", "random order, no shift"), ("pc1_forward", "embedding-direction order (forward)"),
                                                  ("pc1_reverse", "embedding-direction order (reversed)"), ("length_ascending", "question-length order")],
           "TruthfulQA with Qwen2.5-1.5B-Instruct")

# real frontier gaps, one table per task
for task, tag, famlist in [("mmlu", "Qwen2.5-1.5B-Instruct", ["catblock", "subjectblock"]), ("rtp", "gpt2", ["bandblock"]),
                           ("truthfulqa", "Qwen2.5-1.5B-Instruct", ["categoryblock"])]:
    f_ = RES / f"real_{task}_frontier.csv"
    if not f_.exists():
        continue
    d = pd.read_csv(f_)
    d = d[(d.risk == "prob") & (d.metric == "post_shift_viol") & d.family.isin(famlist)]
    rows = []
    for fam in famlist:
        r_ = [fam.replace("block", " blocks")]
        for m in ["promptshift_crc", "no_representation_weighting", "rep_aci", "dtaci", "dsr_aci"]:
            x = d[(d.family == fam) & (d.method == m)]
            r_.append("--" if x.empty else f"{x.iloc[0]['mean']:+.4f} ({x.iloc[0]['se']:.4f})")
        rows.append(r_)
    table(f"tab:frontier_{task}", f"Frontier gap on {task.upper()} streams (post-shift violation minus the Adaptive-CRC window-sweep curve at the same mean threshold; "
          "mean over orderings with standard error). Negative is better than Adaptive CRC.",
          ["Ordering", "PromptShift-CRC", "Gate + recency only", "Kernel + update", "DtACI", "Density ratio"], rows)

# proxy streams (appendix)
pub = RES / "phase_a_public_summary.csv"
if pub.exists():
    pub = pd.read_csv(pub)
    p = pub[pub.method.isin(["static_crc", "rolling_crc", "adaptive_crc", "adaptive_crc_matched", "promptshift_crc", "no_representation_weighting", "no_drift_gate"])]
    rows = []
    for ds in p.dataset.unique():
        for m in p[p.dataset == ds].method.unique():
            x = p[(p.dataset == ds) & (p.method == m)].iloc[0]
            rows.append([ds if m == "static_crc" else "", LAB[m], f(x.coverage, 3), f(x.block_se, 3), f(x.mean_threshold, 3), f(x.max_local_120, 3)])
    table("tab:proxy", "Proxy streams built from benchmark fields (no model outputs). Coverage with moving-block-bootstrap standard error.",
          ["Stream", "Method", "Coverage", "SE", "Mean threshold", "Max local violation"], rows)



# ---------------------------------------------------------------- headline frontier summary
_gaps = [abs(v) for v in pf["mean"].values]
for task_, fams_ in [("mmlu", ["catblock", "subjectblock"]), ("rtp", ["bandblock"]), ("truthfulqa", ["categoryblock"])]:
    f_ = RES / f"real_{task_}_frontier.csv"
    if f_.exists():
        d_ = pd.read_csv(f_); d_ = d_[(d_.risk == "prob") & (d_.metric == "post_shift_viol") & (d_.method == "promptshift_crc") & d_.family.isin(fams_)]
        _gaps += [abs(v) for v in d_["mean"].values]
macro("PSfrontierMaxAbs", f"{max(_gaps):.4f}")

# ---------------------------------------------------------------- robustness to the window choice
def regret_rows():
    rows, store = [], {}
    def one(df, keys, label, tag):
        res = []
        for _, g in df.groupby(keys):
            sw = g[g.method.str.startswith("adaptive_w")]
            ps = g[g.method == "promptshift_crc"].post_shift_viol.iloc[0]
            allv = np.append(sw.post_shift_viol.values, ps)
            res.append((ps - sw.post_shift_viol.min(), sw.post_shift_viol.max() - sw.post_shift_viol.min(),
                        ps - sw.post_shift_viol.max(), 1 + int((sw.post_shift_viol.values < ps - 1e-12).sum()) + 0.5 * int((np.abs(sw.post_shift_viol.values - ps) <= 1e-12).sum()), len(sw) + 1))
        a = np.array(res, dtype=float)
        rows.append([label, str(len(a)), f"{a[:,0].mean():+.4f}", f"{a[:,1].mean():.4f}", f"{a[:,2].mean():+.4f}", f"{a[:,3].mean():.1f} of {int(a[0,4])}"])
        store[tag] = a
    one(syn[~syn.method.isin(["dtaci", "dsr_aci", "pid_quantile_tracking"])], ["stream", "seed"], "synthetic (5 streams x 30 seeds)", "Syn")
    for task, tag, fam, lab, key in [("mmlu", "Qwen2.5-1.5B-Instruct", "catblock", "MMLU, category blocks", "MmluCat"),
                                     ("mmlu", "Qwen2.5-1.5B-Instruct", "subjectblock", "MMLU, subject blocks", "MmluSubj"),
                                     ("rtp", "gpt2", "bandblock", "RealToxicityPrompts, toxicity bands", "Rtp"),
                                     ("truthfulqa", "Qwen2.5-1.5B-Instruct", "categoryblock", "TruthfulQA, category blocks", "Tqa"),
                                     ("rtp", "qwen05b", "bandblock", "RealToxicityPrompts replication (Qwen2.5-0.5B)", "RtpRep")]:
        f_ = RES / f"real_{task}_{tag}_runs.csv"
        if not f_.exists():
            continue
        r = pd.read_csv(f_); r["family"] = r.stream.str.replace(r"_\d+$", "", regex=True)
        one(r[(r.family == fam) & (r.risk == "prob")], ["stream"], lab, key)
    return rows, store

rows, store = regret_rows()
table("tab:regret", "Robustness to the window choice. For each stream, Adaptive CRC is run with a sweep of windows. PromptShift-CRC (PS) has no window to choose. "
      "Entries are means over streams of post-shift violation: PS minus the best window in hindsight, the spread of the window family (worst minus best), "
      "PS minus the worst window, and the rank of PS among the window family and PS (1 = lowest violation). The best window in hindsight is selected on the same data, so the first column is biased in PS's disfavour.",
      ["Setting", "Streams", "PS $-$ best", "Spread", "PS $-$ worst", "Rank of PS"], rows)
for k, a in store.items():
    macro(f"reg{k}Best", f"{a[:,0].mean():+.4f}"); macro(f"reg{k}Spread", f"{a[:,1].mean():.4f}"); macro(f"reg{k}Worst", f"{a[:,2].mean():+.4f}")
    macro(f"reg{k}Rank", f"{a[:,3].mean():.1f}")


# ---------------------------------------------------------------- leave-one-family-out window choice
def _family_frames():
    fr = {}
    base = syn[~syn.method.isin(["dtaci", "dsr_aci", "pid_quantile_tracking"])].copy()
    base["unit"] = base.stream + "|" + base.seed.astype(str)
    fr["synthetic"] = base
    for task, tag, fam, key in [("mmlu", "Qwen2.5-1.5B-Instruct", "catblock", "MMLU categories"), ("mmlu", "Qwen2.5-1.5B-Instruct", "subjectblock", "MMLU subjects"),
                                ("rtp", "gpt2", "bandblock", "RTP bands"), ("truthfulqa", "Qwen2.5-1.5B-Instruct", "categoryblock", "TruthfulQA categories")]:
        f_ = RES / f"real_{task}_{tag}_runs.csv"
        if f_.exists():
            r = pd.read_csv(f_); r["family"] = r.stream.str.replace(r"_\d+$", "", regex=True)
            g = r[(r.family == fam) & (r.risk == "prob")].copy(); g["unit"] = g.stream
            fr[key] = g
    return fr

fam = _family_frames()
posmat = {}   # family -> (units x 5 window violations), PS violations
for k, g in fam.items():
    sw = sorted([m for m in g.method.unique() if m.startswith("adaptive_w")], key=lambda m: int(m.split("_w")[1]))
    W_ = g[g.method.isin(sw)].pivot(index="unit", columns="method", values="post_shift_viol")[sw]
    PS_ = g[g.method == "promptshift_crc"].set_index("unit").post_shift_viol.reindex(W_.index)
    posmat[k] = (W_.values, PS_.values)
rows = []
for held in posmat:
    others = [k for k in posmat if k != held]
    ranks = np.mean([np.mean(np.argsort(np.argsort(posmat[k][0], axis=1), axis=1), axis=0) for k in others], axis=0)
    jstar = int(np.argmin(ranks))
    Wh, PSh = posmat[held]
    d = PSh - Wh[:, jstar]
    rows.append([held, f"{jstar + 1} of 5", f"{d.mean():+.4f}", f"{Wh.mean(axis=0).argmin() + 1} of 5", f"{(PSh - Wh.min(axis=1)).mean():+.4f}"])
    macro("lofo" + "".join(ch for ch in held if ch.isalpha()), f"{d.mean():+.4f}")
table("tab:lofo", "Choosing the Adaptive CRC window without looking at the held-out family. For each family of streams, the window (position in the five-window sweep, short to long) with the best mean rank "
      "over the other four families is chosen; the entry is PromptShift-CRC's post-shift violation minus that window's on the held-out family (negative: PromptShift-CRC is lower). "
      "The last two columns give the window that is best on the held-out family itself and PromptShift-CRC's gap to that window in hindsight.",
      ["Held-out family", "Window chosen elsewhere", "PS $-$ chosen", "Best here", "PS $-$ best here"], rows)

# ---------------------------------------------------------------- macros for the real-output prose
def realmac(task, tag, fam, key, asfx="", default=True):
    f_ = RES / f"real_{task}_{tag}{asfx}_runs.csv"
    name_ = task if default else f"{task}_{tag}{asfx}"
    if not f_.exists():
        return
    r = pd.read_csv(f_); r["family"] = r.stream.str.replace(r"_\d+$", "", regex=True)
    g = r[(r.family == fam) & (r.risk == "prob")]
    n = g.stream.nunique(); macro(f"n{key}", str(n))
    for m, mt in [("static_crc", "Static"), ("rolling_crc", "Rolling"), ("adaptive_crc", "AdaptW"), ("adaptive_crc_matched", "Matched"), ("promptshift_crc", "PS"),
                  ("rep_aci", "KernelUpd"), ("no_representation_weighting", "GateRec"), ("no_drift_gate", "NoGate"), ("no_online_update", "NoUpd"),
                  ("dtaci", "DtACI"), ("dsr_aci", "DensRatio"), ("pid_quantile_tracking", "Pid"), ("adaptive_w512", "AdaptLong"), ("adaptive_w32", "AdaptShort")]:
        h = g[g.method == m]
        if h.empty:
            continue
        macro(f"{key}{mt}Cov", f"{h.coverage.mean():.3f}")
        macro(f"{key}{mt}Thr", f"{h.mean_threshold.mean():.3f}")
        macro(f"{key}{mt}Local", f"{h.max_local_viol.mean():.3f}")
        if h.post_shift_viol.notna().any():
            macro(f"{key}{mt}Post", f"{h.post_shift_viol.mean():.3f}")
    pm_ = RES / f"real_{name_}_ps_vs_matched.csv"
    if pm_.exists():
        c = pd.read_csv(pm_); c = c[(c.family == fam) & (c.risk == "prob")]
        for met, mt in [("post_shift_viol", "Post"), ("mean_threshold", "Thr"), ("max_local_viol", "Local")]:
            x = c[c.metric == met]
            if not x.empty and x.iloc[0]["n"] > 0:
                macro(f"{key}PSvsMatched{mt}", f"{x.iloc[0]['mean']:+.4f}"); macro(f"{key}PSvsMatched{mt}SE", f"{x.iloc[0]['se']:.4f}")
                macro(f"{key}PSvsMatched{mt}Lower", f"{int(x.iloc[0]['n_lower'])}"); macro(f"{key}PSvsMatched{mt}N", f"{int(x.iloc[0]['n'])}")
    fr_ = RES / f"real_{name_}_frontier.csv"
    if fr_.exists():
        d = pd.read_csv(fr_); d = d[(d.family == fam) & (d.risk == "prob") & (d.metric == "post_shift_viol")]
        for m, mt in [("promptshift_crc", "PS"), ("dtaci", "DtACI"), ("rep_aci", "KernelUpd"), ("dsr_aci", "DensRatio"), ("no_representation_weighting", "GateRec")]:
            x = d[d.method == m]
            if not x.empty:
                macro(f"{key}Frontier{mt}", f"{x.iloc[0]['mean']:+.4f}"); macro(f"{key}Frontier{mt}SE", f"{x.iloc[0]['se']:.4f}")

realmac("mmlu", "Qwen2.5-1.5B-Instruct", "catblock", "MmluCat")
realmac("mmlu", "Qwen2.5-1.5B-Instruct", "subjectblock", "MmluSubj")
realmac("rtp", "gpt2", "bandblock", "Rtp")
realmac("truthfulqa", "Qwen2.5-1.5B-Instruct", "categoryblock", "Tqa")
realmac("rtp", "qwen05b", "bandblock", "RtpQ", default=False)
realmac("rtp", "gpt2", "bandblock", "RtpAlphaLow", asfx="_alpha0.05", default=False)
realmac("rtp", "gpt2", "bandblock", "RtpAlphaHigh", asfx="_alpha0.2", default=False)
import json as _json
def _jl0(n_):
    return pd.DataFrame([_json.loads(l_) for l_ in open(ROOT / "real_outputs" / n_, encoding="utf-8")])
_g2 = _jl0("rtp_gpt2.jsonl")
_q = _jl0("rtp_qwen05b.jsonl"); _q["band"] = pd.qcut(_q.prompt_toxicity.fillna(0), 4, labels=False, duplicates="drop")
macro("RtpQFracHigh", f"{(_q.risk > 0.5).mean():.3f}")
_qq = _q.groupby("band").risk.quantile(0.9)
macro("RtpQQnineLo", f"{_qq.min():.3f}"); macro("RtpQQnineHi", f"{_qq.max():.3f}")
macro("RtpQCorr", f"{np.corrcoef(_q.risk.values, _g2.risk.values)[0, 1]:.2f}")
# robustness table for RealToxicityPrompts
_rob = []
for lab_, nm_, dflt_, asfx_, tag_ in [("GPT-2, toxic-bert, $\\alpha=0.1$ (main)", "rtp", True, "", "gpt2"), ("Qwen2.5-0.5B, unbiased-toxic-roberta, $\\alpha=0.1$ (replication)", "rtp_qwen05b", False, "", "qwen05b"),
                                       ("GPT-2, toxic-bert, $\\alpha=0.05$", "rtp_gpt2_alpha0.05", False, "", "gpt2"), ("GPT-2, toxic-bert, $\\alpha=0.2$", "rtp_gpt2_alpha0.2", False, "", "gpt2")]:
    pmf = RES / f"real_{nm_}_ps_vs_matched.csv"; frf = RES / f"real_{nm_}_frontier.csv"
    if not (pmf.exists() and frf.exists()):
        continue
    c_ = pd.read_csv(pmf); c_ = c_[(c_.family == "bandblock") & (c_.risk == "prob")]
    f2_ = pd.read_csv(frf); f2_ = f2_[(f2_.family == "bandblock") & (f2_.risk == "prob") & (f2_.metric == "post_shift_viol")]
    a_ = c_[c_.metric == "post_shift_viol"].iloc[0]; b_ = c_[c_.metric == "mean_threshold"].iloc[0]
    gps = f2_[f2_.method == "promptshift_crc"].iloc[0]; gdt = f2_[f2_.method == "dtaci"].iloc[0]
    _rob.append([lab_, f"{a_['mean']:+.4f} ({a_['se']:.4f})", f"{int(a_['n_lower'])}/{int(a_['n'])}", f"{b_['mean']:+.4f}", f"{int(b_['n_lower'])}/{int(b_['n'])}",
                 f"{gps['mean']:+.4f} ({gps['se']:.4f})", f"{gdt['mean']:+.4f} ({gdt['se']:.4f})"])
table("tab:rtp_robust", "RealToxicityPrompts across generators, classifiers and target levels (24 band orderings each). PromptShift-CRC minus the matched baseline, "
      "post-shift violation with standard error and number of orderings in which PromptShift-CRC is lower, and mean threshold; frontier gaps of PromptShift-CRC and DtACI.",
      ["Configuration", "PS $-$ matched post-shift", "PS lower in", "PS $-$ matched threshold", "PS lower in", "Frontier gap PS", "Frontier gap DtACI"], _rob)


# ---------------------------------------------------------------- facts about the model-output data
import json, re as _re
def _jl(name):
    return pd.DataFrame([json.loads(l) for l in open(ROOT / "real_outputs" / name, encoding="utf-8")])
_src = (ROOT / "scripts" / "run_real_streams.py").read_text(encoding="utf-8")
_cats = eval(_src[_src.index("CATS = ") + 7:_src.index("SUBJ2CAT")])
_s2c = {s_: c_ for c_, ss_ in _cats.items() for s_ in ss_}
mm = _jl("mmlu_Qwen2.5-1.5B-Instruct.jsonl"); mm["cat"] = mm.subject.map(_s2c)
macro("MmluN", str(len(mm))); macro("MmluAcc", f"{1 - mm.error.mean():.3f}")
q90 = mm.groupby("cat").risk.quantile(0.9); q50 = mm.groupby("cat").risk.quantile(0.5)
macro("MmluQnineLo", f"{q90.min():.3f}"); macro("MmluQnineHi", f"{q90.max():.3f}")
macro("MmluMedLo", f"{q50.min():.3f}"); macro("MmluMedHi", f"{q50.max():.3f}")
tq = _jl("truthfulqa_Qwen2.5-1.5B-Instruct.jsonl")
macro("TqaN", str(len(tq))); macro("TqaAcc", f"{1 - tq.error.mean():.3f}"); macro("TqaQnine", f"{tq.risk.quantile(0.9):.3f}")
rt = _jl("rtp_gpt2.jsonl"); rt["band"] = pd.qcut(rt.prompt_toxicity.fillna(0), 4, labels=False, duplicates="drop")
macro("RtpN", str(len(rt)))
qb = rt.groupby("band").risk.quantile(0.9)
macro("RtpQnineLo", f"{qb.min():.3f}"); macro("RtpQnineHi", f"{qb.max():.3f}")
macro("RtpMeanLo", f"{rt.groupby('band').risk.mean().min():.3f}"); macro("RtpMeanHi", f"{rt.groupby('band').risk.mean().max():.3f}")
macro("RtpFracHigh", f"{(rt.risk > 0.5).mean():.3f}")


# ---------------------------------------------------------------- hyperparameter table (read from crc_lib, never retyped)
import sys as _sys
_sys.path.insert(0, str(ROOT / "scripts"))
import crc_lib as _C
hp_rows = []
for nm, cfg in [("Synthetic pilot", _C.CFG_PILOT), ("Synthetic ablation stream", _C.CFG_ABL), ("Synthetic prototype tasks (3)", _C.CFG_PROTO)]:
    _T = {"Synthetic pilot": 2400, "Synthetic ablation stream": 2400, "Synthetic prototype tasks (3)": 2200}[nm] - cfg["cal_start"]
    hp_rows.append([nm, str(cfg["cal_start"]), str(cfg["window"]), str(cfg["recent"]), f"{cfg['bw']}", f"{cfg['gate_c']}", f"{cfg['gate_k']}", f"{cfg['gamma']}", "3W", f"{_C.gibbs_candes_bound(0.1, cfg['gamma'], _T):.3f}"])
hp_rows.append(["MMLU", "256", "256", "128", "$h$ (median)", "$1.1h$", "$2.5/h$", "0.02", "3W", f"{_C.gibbs_candes_bound(0.1, 0.02, 2565 - 256):.3f}"])
hp_rows.append(["RealToxicityPrompts (both generators)", "256", "256", "128", "$h$ (median)", "$1.1h$", "$2.5/h$", "0.02", "3W", f"{_C.gibbs_candes_bound(0.1, 0.02, 4000 - 256):.3f}"])
hp_rows.append(["TruthfulQA", "100", "100", "50", "$h$ (median)", "$1.1h$", "$2.5/h$", "0.03", "3W", f"{_C.gibbs_candes_bound(0.1, 0.03, 817 - 100):.3f}"])
for nm, cfg, w_ in [("Proxy: TruthfulQA fields", _C.CFG_TQA, "see text"), ("Proxy: SummEval", _C.CFG_SUMM, ""), ("Proxy: LongHalQA", _C.CFG_LHQA, ""), ("Proxy: RealToxicityPrompts fields", _C.CFG_RTP, "")]:
    _pb = pd.read_csv(RES / "phase_a_public_summary.csv")
    _key = {"Proxy: TruthfulQA fields": "TruthfulQA", "Proxy: SummEval": "SummEval", "Proxy: LongHalQA": "LongHalQA", "Proxy: RealToxicityPrompts fields": "RealToxicityPrompts"}[nm]
    _gb = _pb[(_pb.dataset == _key) & (_pb.method == "promptshift_crc")].gc_bound.iloc[0]
    hp_rows.append([nm, "rule", "rule", "rule", f"{cfg['bw']}", f"{cfg['gate_c']}", f"{cfg['gate_k']}", f"{cfg['gamma']}", "3W", f"{_gb:.3f}"])
table("tab:hyper", "Settings used in every experiment. $W$ is the full window, ``recent'' the recency window, $h$ the kernel bandwidth, gate centre $c$ and slope $\kappa$ as in Section~\\ref{sec:method}, "
      "$\gamma$ the step of the online update, and ``pool'' the length of the history used by the weighted rules. All use $\\alpha=0.1$. The synthetic constants are those of the earlier version of this work and were not changed; "
      "the real-stream constants follow rules fixed in advance. ``Bound'' is the Theorem~\\ref{thm:longrun} bound on the distance between coverage and $0.90$. For the proxy streams the window lengths follow the rule in the released script (a fraction of the stream length with caps).",
      ["Experiment", "Prefix", "$W$", "Recent", "$h$", "Gate $c$", "Slope $\kappa$", "$\gamma$", "Pool", "Bound"], hp_rows)

import re
MAINTAB = {"tab:rtp_robust", "tab:lofo", "tab:synthetic", "tab:paired", "tab:frontier_syn", "tab:mmlu_catblock", "tab:rtp_bandblock", "tab:regret", "tab:frontier_mmlu", "tab:frontier_rtp"}
def _label(t_):
    return re.search(r"label[{]([^}]*)[}]", t_).group(1)
NL = chr(10)
(ROOT / "tables_generated.tex").write_text(NL.join(t_ for t_ in out_tab if _label(t_) in MAINTAB), encoding="utf-8")
(ROOT / "tables_appendix_generated.tex").write_text(NL.join(t_ for t_ in out_tab if _label(t_) not in MAINTAB), encoding="utf-8")
(ROOT / "numbers_generated.tex").write_text("\n".join(out_num) + "\n", encoding="utf-8")
print("tables:", len(out_tab), "macros:", len(out_num))
