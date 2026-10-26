"""Phase A, proxy public streams: TruthfulQA, SummEval, LongHalQA, RealToxicityPrompts.

These four streams use hand-built features and risk scores derived from public benchmark
fields (see the paper); they contain NO model outputs. Revision changes: repeated tail
segments removed (SummEval, LongHalQA), matched-window Adaptive CRC, unprojected update,
moving-block-bootstrap standard errors (block = recent window).
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import crc_lib as C
import run_longhalqa_promptshift as lhqa
import run_realtoxicity_promptshift as rtp
import run_summeval_promptshift as summ
import run_truthfulqa_promptshift as tqa


def windows(n, cs, w, rw):
    return dict(cal_start=min(cs[0], max(cs[1], n // cs[2])), window=min(w[0], max(w[1], n // w[2])),
                recent=min(rw[0], max(rw[1], n // rw[2])))


def load():
    out = []
    raw, src = tqa.load_truthfulqa_rows(); e, r, _, _ = tqa.build_stream(raw)
    out.append(("TruthfulQA", e, r, {**C.CFG_TQA, **windows(len(r), (160, 40, 4), (160, 40, 4), (80, 25, 8))}, src))
    raw, src = summ.load_rows(); e, r = summ.build_stream(raw)
    out.append(("SummEval", e, r, {**C.CFG_SUMM, **windows(len(r), (320, 120, 5), (260, 100, 6), (120, 60, 12))}, src))
    raw, src = lhqa.load_rows(); e, r = lhqa.build_stream(raw)
    out.append(("LongHalQA", e, r, {**C.CFG_LHQA, **windows(len(r), (300, 120, 5), (240, 100, 6), (120, 60, 12))}, src))
    raw, src = rtp.load_rows(); e, r = rtp.build_stream(raw)
    out.append(("RealToxicityPrompts", e, r, {**C.CFG_RTP, **windows(len(r), (600, 160, 5), (500, 150, 6), (200, 80, 12))}, src))
    return out


if __name__ == "__main__":
    rows = []
    for name, emb, risk, cfg, src in load():
        n = len(risk); rw, w = cfg["recent"], cfg["window"]
        methods = ["static_crc", "rolling_crc", "adaptive_crc", "adaptive_crc_matched", "representation_weighted_crc",
                   "rep_aci", "promptshift_crc", "no_representation_weighting", "no_drift_gate", "no_recent_fallback",
                   "no_online_update", f"adaptive_w{rw // 2}", f"adaptive_w{2 * w}"]
        print(f"== {name}: n={n} cal_start={cfg['cal_start']} window={w} recent={rw} source={src}", flush=True)
        for m in methods:
            tr = C.run_stream(emb, risk, m, cfg)
            s = C.summarise(tr)
            se = C.block_bootstrap_se(tr["miss"], block=rw, reps=1000, seed=0)
            lm = C.local_metrics(tr["miss"], [], post=1) if False else None
            roll = np.convolve(tr["miss"], np.ones(120) / 120, mode="valid")
            row = {"dataset": name, "method": m, "n_test": len(tr["miss"]), **s, "abs_err": abs(s["coverage"] - 0.9),
                   "block_se": se, "max_local_120": float(roll.max()),
                   "gc_bound": C.gibbs_candes_bound(0.1, cfg["gamma"], len(tr["miss"])) if C.is_online(m) else float("nan")}
            rows.append(row)
            print(f"  {m:30s} cov={s['coverage']:.4f} (se {se:.4f}) |err|={row['abs_err']:.4f} ess={s['mean_ess']:.1f}", flush=True)
    with (ROOT / "results" / "phase_a_public_summary.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
