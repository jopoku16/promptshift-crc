"""Phase A: 30-seed study of the five synthetic streams with the revised baselines.

Streams: pilot, ablation stream, three prototype tasks. Methods include Adaptive CRC with
the June window W, Adaptive CRC matched to PromptShift's recent window (rule fixed in
advance, never tuned), a window sweep, the unprojected online update, and all ablations.
Writes results/phase_a_synthetic_runs.csv and results/phase_a_synthetic_summary.csv.
"""
from __future__ import annotations
import csv, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import crc_lib as C
import run_ablation_study as abl
import run_prompt_shift_simulation as pilot
import run_real_promptshift_benchmarks as proto

SEEDS = list(range(30))
RES = ROOT / "results"; RES.mkdir(exist_ok=True)


SWEEP = [32, 64, 128, 256, 512]  # Adaptive CRC window sweep for the validity/sharpness frontier


def methods_for(cfg):
    return ["static_crc", "rolling_crc", "adaptive_crc", "adaptive_crc_matched",
            "representation_weighted_crc", "rep_aci", "promptshift_crc",
            "no_representation_weighting", "no_drift_gate", "no_recent_fallback", "no_online_update"] + [f"adaptive_w{k}" for k in SWEEP]


STREAMS = {
    "pilot": (lambda s: pilot.make_stream(seed=s, n=2400)[:2], C.CFG_PILOT, [300, 900, 1500]),
    "ablation": (lambda s: abl.make_stream(seed=s, n=2400), C.CFG_ABL, [300, 900, 1500]),
}
for task in ["qa_topic_shift", "summarization_domain_shift", "safety_intent_shift"]:
    STREAMS[f"proto:{task}"] = ((lambda s, task=task: proto.make_task_stream(task, seed=s)[:2]), C.CFG_PROTO, [200, 700, 1300])


CORRECT = True  # Barber correction; set by main from argv


def job(args):
    name, seed, correct = args
    make, cfg, shifts = STREAMS[name]
    emb, risk = make(seed)
    rows = []
    for m in methods_for(cfg):
        tr = C.run_stream(emb, risk, m, cfg, correct=correct)
        T = len(tr["miss"])
        row = {"stream": name, "seed": seed, "method": m, **C.summarise(tr), **C.local_metrics(tr["miss"], shifts),
               "gc_bound": C.gibbs_candes_bound(0.1, cfg["gamma"], T) if C.is_online(m) else float("nan"),
               "abs_err": abs(1 - tr["miss"].mean() - 0.9)}
        rows.append(row)
    return rows


if __name__ == "__main__":
    CORRECT = (sys.argv[1] != "nocorrect") if len(sys.argv) > 1 else True
    SUFFIX = "" if CORRECT else "_nocorrect"
    if not CORRECT:
        SEEDS = SEEDS[:10]
    jobs = [(n, s, CORRECT) for n in STREAMS for s in SEEDS]
    with Pool(7) as p:
        out = p.map(job, jobs, chunksize=1)
    rows = [r for rr in out for r in rr]
    with (RES / f"phase_a_synthetic_runs{SUFFIX}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # summary: mean and sd over seeds per stream x method, plus paired contrasts
    summ = []
    for name in STREAMS:
        ms = methods_for(STREAMS[name][1])
        sub = [r for r in rows if r["stream"] == name]
        for m in ms:
            v = [r for r in sub if r["method"] == m]
            cov = np.array([r["coverage"] for r in v]); err = np.array([r["abs_err"] for r in v])
            summ.append({"stream": name, "method": m, "n_seeds": len(v),
                         "coverage_mean": cov.mean(), "coverage_sd": cov.std(ddof=1),
                         "abs_err_mean": err.mean(), "abs_err_sd": err.std(ddof=1),
                         "max_local_viol": np.mean([r["max_local_viol"] for r in v]),
                         "intervention_rate": np.mean([r["intervention_rate"] for r in v]),
                         "frac_inf": np.mean([r["frac_inf"] for r in v]),
                         "post_shift_viol": np.mean([r["post_shift_viol"] for r in v]),
                         "mean_ess": np.mean([r["mean_ess"] for r in v]),
                         "mean_threshold": np.mean([r["mean_threshold"] for r in v])})
    with (RES / f"phase_a_synthetic_summary{SUFFIX}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)
    print("wrote", len(rows), "runs")
