"""Round 4, item C: per-step traces of PromptShift-CRC and the matched baseline.

Pre-declared in PREREGISTRATION_ROUND4_2026-10-26.md (item C). Nothing here is tuned; every constant is the
declared one. Writes results/round4_trace_<family>.pkl, analysed by analyze_round4_trace.py.

Usage:
  python run_round4_trace.py real rtp gpt2        # band-block orderings, GPT-2 continuations
  python run_round4_trace.py real rtp qwen05b     # band-block orderings, Qwen2.5-0.5B continuations
  python run_round4_trace.py real mmlu            # category-block orderings, alpha = 0.1
  python run_round4_trace.py synth                # five synthetic streams, 30 seeds
"""
from __future__ import annotations
import pickle, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
MODE = sys.argv[1]
if MODE == "real":
    _task = sys.argv[2]
    _tag = sys.argv[3] if len(sys.argv) > 3 else None
    _orig_argv = list(sys.argv)
    sys.argv = ["run_real_streams.py", _task] + (["--tag", _tag] if _tag else [])
    import run_real_streams as R  # reads its task and tag from sys.argv at import time
    sys.argv = _orig_argv
import crc_lib as C

G = {}
ALPHA = 0.1


def _init(*objs):
    G["objs"] = objs


def real_job(name):
    recs, emb, orders = G["objs"]
    idx, bounds = orders[name]
    risk = np.array([recs[i]["risk"] for i in idx])
    e = emb[idx]
    cfg = R.make_cfg(e)
    shifts = [b - R.CS for b in bounds if b - R.CS >= 0]
    ps = C.run_stream(e, risk, "promptshift_crc", cfg, alpha=ALPHA, correct=True)
    mt = C.run_stream(e, risk, "adaptive_crc_matched", cfg, alpha=ALPHA, correct=True)
    return name, dict(err_ps=ps["miss"], err_matched=mt["miss"], dpre=ps["drift_pre"], d=ps["drift"], ess=ps["ess"], shifts=np.array(shifts), post=R.SETUP["post"])


def synth_job(args):
    import run_phase_a_synthetic as S
    name, seed = args
    make, cfg, shifts = S.STREAMS[name]
    emb, risk = make(seed)
    ps = C.run_stream(emb, risk, "promptshift_crc", cfg, alpha=ALPHA, correct=True)
    mt = C.run_stream(emb, risk, "adaptive_crc_matched", cfg, alpha=ALPHA, correct=True)
    return (name, seed), dict(err_ps=ps["miss"], err_matched=mt["miss"], dpre=ps["drift_pre"], d=ps["drift"], ess=ps["ess"], shifts=np.array(shifts), post=150)


if __name__ == "__main__":
    if MODE == "real":
        recs, emb = R.load()
        orders = R.build_orders(recs, emb)
        prefix = {"rtp": "bandblock", "mmlu": "catblock"}[_task]
        names = [n for n in orders if n.startswith(prefix)]
        out_name = f"{_task}_{_tag}" if _tag else _task
        with Pool(2, initializer=_init, initargs=(recs, emb, orders)) as p:
            res = dict(p.map(real_job, names, chunksize=1))
    else:
        import run_phase_a_synthetic as S
        jobs = [(n, s) for n in S.STREAMS for s in S.SEEDS]
        out_name = "synthetic"
        with Pool(2) as p:
            res = dict(p.map(synth_job, jobs, chunksize=1))
    with (ROOT / "results" / f"round4_trace_{out_name}.pkl").open("wb") as f:
        pickle.dump(res, f)
    print("wrote", out_name, len(res), flush=True)
