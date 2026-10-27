"""Figures for the revised paper. Reads results/*.csv, writes figures/*.pdf (vector) and .png.
Usage: python make_figures.py synthetic | real_mmlu | real_truthfulqa | all
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures_v2"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})

STYLE = {  # semantic: grey = Adaptive CRC family, crimson = PromptShift-CRC, others one colour each
    "promptshift_crc": ("PromptShift-CRC", "#b3202a", "o"),
    "no_representation_weighting": ("gate + recency only", "#e08a1e", "s"),
    "no_drift_gate": ("no gate (fixed mix)", "#1f6f8b", "^"),
    "rep_aci": ("kernel weights + update", "#6a4c93", "v"),
    "dtaci": ("DtACI", "#2a9d8f", "D"),
    "dsr_aci": ("density-ratio weights", "#8a6f3b", "P"),
    "pid_quantile_tracking": ("quantile tracking", "#555555", "X"),
}


def sweep_methods(df):
    return sorted([m for m in df.method.unique() if m.startswith("adaptive_w")], key=lambda m: int(m.split("_w")[1]))


def frontier_panel(ax, g, ymet, title, ylabel=True, annotate=True):
    sw = sweep_methods(g)
    pts = g.groupby("method")[["mean_threshold", ymet]].mean()
    xs = [pts.loc[m, "mean_threshold"] for m in sw]
    ys = [pts.loc[m, ymet] for m in sw]
    order = np.argsort(xs)
    ax.plot(np.array(xs)[order], np.array(ys)[order], color="#999999", lw=1.2, marker="o", ms=3, zorder=1)
    if annotate:
        for m, x, y in zip(sw, xs, ys):
            ax.annotate(m.split("_w")[1], (x, y), textcoords="offset points", xytext=(3, 3), fontsize=6, color="#777777")
    for m, (lab, col, mk) in STYLE.items():
        if m in pts.index:
            ax.scatter(pts.loc[m, "mean_threshold"], pts.loc[m, ymet], s=22, color=col, marker=mk, zorder=3, label=lab)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("mean threshold")
    if ylabel:
        ax.set_ylabel(ymet.replace("_", " "))


def synthetic():
    r = pd.read_csv(RES / "phase_a_synthetic_runs.csv")
    b = RES / "phase_a_baselines_runs.csv"
    if b.exists():
        r = pd.concat([r, pd.read_csv(b)], ignore_index=True)
    names = {"pilot": "pilot", "ablation": "ablation stream", "proto:qa_topic_shift": "QA topic shift",
             "proto:summarization_domain_shift": "summarization shift", "proto:safety_intent_shift": "safety intent shift"}
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.6), sharey=False)  # drawn at the printed width, so the 8 pt text stays 8 pt
    axes = axes.ravel()
    for ax, (st, nm) in zip(axes, names.items()):
        frontier_panel(ax, r[r.stream == st], "post_shift_viol", nm, ylabel=(st in ("pilot", "proto:qa_topic_shift")), annotate=(st == "pilot"))
    h, l = axes[0].get_legend_handles_labels()
    axes[5].legend(h, l, loc="center", frameon=False, fontsize=7)
    axes[5].axis("off")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_frontier_synthetic.{ext}")
    plt.close(fig)


def real(task, model="Qwen2.5-1.5B-Instruct"):
    f = RES / f"real_{task}_{model}_runs.csv"
    if not f.exists():
        print("missing", f); return
    r = pd.read_csv(f)
    r["family"] = r.stream.str.replace(r"_\d+$", "", regex=True)
    fams = ["catblock", "subjectblock", "categoryblock", "bandblock", "iid", "pc1_forward", "pc1_reverse", "length_ascending", "prompt_toxicity_ascending", "prompt_toxicity_descending"]
    fams = [x for x in fams if x in set(r.family)]
    for risk in ("prob", "error"):
        sub = r[r.risk == risk]
        ncols = 4
        nrows = int(np.ceil((len(fams) + 1) / ncols))  # one spare cell holds the legend
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.5, 2.35 * nrows))  # drawn at the printed width, so the 8 pt text stays 8 pt
        axes = axes.ravel()
        for ax, fam in zip(axes, fams):
            g = sub[sub.family == fam]
            met = "post_shift_viol" if g.post_shift_viol.notna().any() else "max_local_viol"
            frontier_panel(ax, g, met, fam.replace("prompt_toxicity_", "toxicity ").replace("_", " "), ylabel=(fam in fams[::ncols]), annotate=(fam == fams[0]))
        h, l = axes[0].get_legend_handles_labels()
        axes[len(fams)].legend(h, l, loc="center", frameon=False, fontsize=6.5)
        for ax in axes[len(fams):]:
            ax.axis("off")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(FIG / f"fig_frontier_{task}_{risk}.{ext}")
        plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("synthetic", "all"):
        synthetic()
    if which in ("real_mmlu", "all"):
        real("mmlu")
    if which in ("real_truthfulqa", "all"):
        real("truthfulqa")
    if which in ("real_rtp", "all"):
        real("rtp", "gpt2")
    print("figures in", FIG)
