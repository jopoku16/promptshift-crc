"""Real-model-output streams: MMLU and TruthfulQA MC1 with Qwen2.5-1.5B-Instruct and MiniLM embeddings.

Pre-declared design (fixed before any result was seen; nothing is tuned on these streams):
  risk        = 1 - p(correct option) (primary) and the 0/1 error (secondary)
  features    = unit-norm all-MiniLM-L6-v2 embedding of the question text
  MMLU        cal_start = W = 256, recent = 128, gamma = 0.02, post-shift window 150
  TruthfulQA  cal_start = W = 100, recent = 50,  gamma = 0.03, post-shift window 50  (n = 817, so a larger step)
  common      rep pool = PromptShift pool = 3W, unprojected update, alpha = 0.1, Barber correction on,
              bandwidth h = median pairwise distance among the first cal_start questions of the stream,
              gate centre = 1.1 h, gate slope = 2.5 / h
  orders      MMLU: 24 permutations of the 4 broad categories (block shift), 20 random subject orders, 20 iid orders,
              PC1 forward / reverse, question length.  TruthfulQA: 20 random category orders (block shift),
              20 iid orders, PC1 forward / reverse, question length.
  All orderings use only the question text or the benchmark's own metadata, never the model's output.
Usage: python run_real_streams.py mmlu|truthfulqa
"""
from __future__ import annotations
import csv, itertools, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import crc_lib as C
import baselines_online as B

TASK = sys.argv[1] if len(sys.argv) > 1 else "mmlu"


def _opt(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


ALPHA = float(_opt("--alpha", 0.1))          # target level (declared 0.1; 0.05 and 0.2 are robustness runs)
RTP_TAG = _opt("--tag", "gpt2")              # RTP generator tag: gpt2 (main) or qwen05b (replication)
ONLY = _opt("--only", None)                  # restrict to one ordering family (robustness runs)
MODEL_TAG = "Qwen2.5-1.5B-Instruct"
RO = ROOT / "real_outputs"
SETUP = {"mmlu": dict(W=256, RW=128, CS=256, gamma=0.02, post=150),
         "truthfulqa": dict(W=100, RW=50, CS=100, gamma=0.03, post=50),
         "rtp": dict(W=256, RW=128, CS=256, gamma=0.02, post=150)}[TASK]
if TASK == "rtp":
    MODEL_TAG = RTP_TAG
W, RW, CS = SETUP["W"], SETUP["RW"], SETUP["CS"]

CATS = {
    "STEM": "abstract_algebra anatomy astronomy college_biology college_chemistry college_computer_science college_mathematics college_physics computer_security conceptual_physics electrical_engineering elementary_mathematics high_school_biology high_school_chemistry high_school_computer_science high_school_mathematics high_school_physics high_school_statistics machine_learning".split(),
    "humanities": "formal_logic high_school_european_history high_school_us_history high_school_world_history international_law jurisprudence logical_fallacies moral_disputes moral_scenarios philosophy prehistory professional_law world_religions".split(),
    "social": "econometrics high_school_geography high_school_government_and_politics high_school_macroeconomics high_school_microeconomics high_school_psychology human_sexuality professional_psychology public_relations security_studies sociology us_foreign_policy".split(),
    "other": "business_ethics clinical_knowledge college_medicine global_facts human_aging management marketing medical_genetics miscellaneous nutrition professional_accounting professional_medicine virology".split(),
}
SUBJ2CAT = {s: c for c, ss in CATS.items() for s in ss}
METHODS = ["static_crc", "rolling_crc", "adaptive_crc", "adaptive_crc_matched", "representation_weighted_crc", "rep_aci",
           "promptshift_crc", "no_representation_weighting", "no_drift_gate", "no_recent_fallback", "no_online_update"] + \
          [f"adaptive_w{k}" for k in (RW // 4, RW // 2, RW, 2 * RW, 4 * RW)] + list(B.BASELINES)
# duplicates (e.g. adaptive_w{RW} equals adaptive_crc_matched) are kept so the sweep is a clean curve


def group_of(r):
    if TASK == "mmlu":
        return SUBJ2CAT[r["subject"]]
    if TASK == "rtp":
        return r["band"]
    return r["category"]


def load():
    fn = RO / {"mmlu": f"mmlu_{MODEL_TAG}.jsonl", "truthfulqa": f"truthfulqa_{MODEL_TAG}.jsonl", "rtp": f"rtp_{RTP_TAG}.jsonl"}[TASK]
    recs = [json.loads(l) for l in fn.open(encoding="utf-8")]
    recs.sort(key=lambda r: r["idx"])
    if TASK == "rtp":  # bands = quartiles of the PROMPT toxicity annotation (a property of the input, not of the model output)
        pt = np.array([r["prompt_toxicity"] if r["prompt_toxicity"] is not None else 0.0 for r in recs])
        edges = np.quantile(pt, [0.25, 0.5, 0.75])
        for r, v in zip(recs, pt):
            r["band"] = int(np.searchsorted(edges, v, side="right"))
            r["error"] = int(r["risk"] > 0.5)
    for r in recs:  # text may be absent in the public release (embeddings are cached); question length is stored
        r.setdefault("qlen", len(r["question"]) if "question" in r else 0)
    cache = RO / f"{TASK}_{MODEL_TAG}_emb_cache.npz"
    have = {}
    if cache.exists():
        z = np.load(cache); have = dict(zip(z["idx"].tolist(), z["emb"]))
    need = [r for r in recs if r["idx"] not in have]
    if need:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        e = st.encode([r["question"] for r in need], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        for r, v in zip(need, e):
            have[r["idx"]] = v
        np.savez(cache, idx=np.array(list(have.keys())), emb=np.vstack(list(have.values())))
    return recs, np.vstack([have[r["idx"]] for r in recs])


def make_cfg(e):
    sub = e[:CS][:: max(1, CS // 128)]
    d = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=2)
    h = float(np.median(d[np.triu_indices(len(sub), 1)]))
    return dict(cal_start=CS, window=W, recent=RW, bw=h, gate_k=2.5 / h, gate_c=1.1 * h, gamma=SETUP["gamma"],
                rep_mult=3, ps_mult=3, public_style=True)


def build_orders(recs, emb):
    n = len(recs)
    grp = np.array([group_of(r) for r in recs])
    orders = {}
    if TASK == "rtp":
        for pi, perm in enumerate(itertools.permutations(range(4))):
            rng = np.random.default_rng(1000 + pi)
            idx, bounds = [], []
            for c in perm:
                idx += rng.permutation(np.where(grp == c)[0]).tolist(); bounds.append(len(idx))
            orders[f"bandblock_{pi:02d}"] = (np.array(idx), bounds[:-1])
        pt = np.array([r["prompt_toxicity"] if r["prompt_toxicity"] is not None else 0.0 for r in recs])
        orders["prompt_toxicity_ascending"] = (np.argsort(pt, kind="stable"), [])
        orders["prompt_toxicity_descending"] = (np.argsort(-pt, kind="stable"), [])
    elif TASK == "mmlu":
        for pi, perm in enumerate(itertools.permutations(CATS.keys())):
            rng = np.random.default_rng(1000 + pi)
            idx, bounds = [], []
            for c in perm:
                idx += rng.permutation(np.where(grp == c)[0]).tolist(); bounds.append(len(idx))
            orders[f"catblock_{pi:02d}"] = (np.array(idx), bounds[:-1])
        subj = np.array([r["subject"] for r in recs])
        for s in range(20):
            rng = np.random.default_rng(2000 + s)
            idx, bounds = [], []
            for sj in rng.permutation(sorted(set(subj.tolist()))):
                idx += rng.permutation(np.where(subj == sj)[0]).tolist(); bounds.append(len(idx))
            orders[f"subjectblock_{s:02d}"] = (np.array(idx), bounds[:-1])
    else:
        cats = sorted(set(grp.tolist()))
        for s in range(20):
            rng = np.random.default_rng(2000 + s)
            idx, bounds = [], []
            for c in rng.permutation(cats):
                idx += rng.permutation(np.where(grp == c)[0]).tolist(); bounds.append(len(idx))
            orders[f"categoryblock_{s:02d}"] = (np.array(idx), bounds[:-1])
    for s in range(20):
        orders[f"iid_{s:02d}"] = (np.random.default_rng(3000 + s).permutation(n), [])
    _, _, vt = np.linalg.svd(emb - emb.mean(0), full_matrices=False)
    pc1 = (emb - emb.mean(0)) @ vt[0]
    orders["pc1_forward"] = (np.argsort(pc1), [])
    orders["pc1_reverse"] = (np.argsort(-pc1), [])
    orders["length_ascending"] = (np.argsort(np.array([r["qlen"] for r in recs]), kind="stable"), [])
    return orders


G = {}


def job(args):
    name, idx, bounds, risk_kind = args
    recs, emb = G["recs"], G["emb"]
    risk = np.array([recs[i]["risk"] if risk_kind == "prob" else float(recs[i]["error"]) for i in idx])
    e = emb[idx]
    cfg = make_cfg(e)
    shifts = [b - CS for b in bounds if b - CS >= 0]
    out = []
    for m in METHODS:
        tr = B.BASELINES[m](e, risk, cfg, alpha=ALPHA) if m in B.BASELINES else C.run_stream(e, risk, m, cfg, alpha=ALPHA, correct=True)
        row = {"task": TASK, "stream": name, "risk": risk_kind, "method": m, **C.summarise(tr),
               "abs_err": abs(1 - tr["miss"].mean() - (1 - ALPHA)), "alpha": ALPHA, "h": cfg["bw"],
               "gc_bound": C.gibbs_candes_bound(ALPHA, cfg["gamma"], len(tr["miss"])) if C.is_online(m) else float("nan")}
        roll = np.convolve(tr["miss"], np.ones(120 if TASK == "mmlu" else 60) / (120 if TASK == "mmlu" else 60), mode="valid")
        row["max_local_viol"] = float(roll.max())
        row["post_shift_viol"] = float(np.mean([tr["miss"][s: s + SETUP["post"]].mean() for s in shifts])) if shifts else float("nan")
        out.append(row)
    return out


def _init(r, e):
    G.update(recs=r, emb=e)


if __name__ == "__main__":
    recs, emb = load()
    print("task", TASK, "items", len(recs), flush=True)
    orders = build_orders(recs, emb)
    jobs = [(nm, idx, bd, rk) for nm, (idx, bd) in orders.items() for rk in ("prob", "error")
            if ONLY is None or (nm.startswith(ONLY) and rk == "prob")]
    with Pool(7, initializer=_init, initargs=(recs, emb)) as p:
        res = p.map(job, jobs, chunksize=1)
    rows = [r for rr in res for r in rr]
    with (ROOT / "results" / f"real_{TASK}_{MODEL_TAG}{'' if ALPHA == 0.1 else f'_alpha{ALPHA}'}_runs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", len(rows), flush=True)
