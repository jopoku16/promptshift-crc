"""Check crc_lib against the June results (project=True) and show the unprojected update."""
import csv, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import crc_lib as C
import run_prompt_shift_simulation as pilot
import run_ablation_study as abl
import run_real_promptshift_benchmarks as proto

def read(p):
    with open(HERE.parent / "results" / p, newline="") as f:
        return {r.get("method"): r for r in csv.DictReader(f)}

def check(name, emb, risk, cfg, csvname, methods, key=None):
    ref = read(csvname)
    print(f"\n== {name}")
    worst = 0
    for m in methods:
        mm = {"promptshift_crc": "full_promptshift_crc"}.get(m, m) if "ablation" in csvname else m
        tr = C.run_stream(emb, risk, m, cfg, project=True)
        cov = C.summarise(tr)["coverage"]
        un = C.summarise(C.run_stream(emb, risk, m, cfg, project=False))["coverage"]
        r = ref.get(mm) or ref.get(m)
        refcov = float(r["coverage"]) if r else float("nan")
        worst = max(worst, abs(cov - refcov)) if r else worst
        print(f"  {m:28s} June CSV {refcov:.4f} | lib(project) {cov:.4f} | lib(unprojected) {un:.4f}")
    print("  max |lib(project) - CSV| =", round(worst, 6))

emb, risk, _ = pilot.make_stream(seed=7, n=2400)
check("pilot seed 7", emb, risk, C.CFG_PILOT, "prompt_shift_pilot_summary.csv",
      ["static_crc", "rolling_crc", "adaptive_crc", "representation_weighted_crc", "promptshift_crc"])
emb, risk, _ = proto.make_task_stream("qa_topic_shift", seed=11)
ref = read("real_promptshift_summary.csv")
print("(prototype CSV is multi-task; checked separately in the 30-seed driver)")
