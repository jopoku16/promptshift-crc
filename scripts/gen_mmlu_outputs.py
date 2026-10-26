"""Real model outputs for a domain-shift stream: MMLU with an open model on CPU.

Model: Qwen/Qwen2.5-1.5B-Instruct (default). For each question the model sees the standard
4-option prompt and we read the next-token probabilities of ' A'..' D'. The loss is
risk = 1 - p(correct option), a bounded score in [0, 1]; the 0/1 error is also stored.
Subsample: up to 45 questions per subject, fixed seed 0, so the stream is reproducible.
Embeddings: sentence-transformers/all-MiniLM-L6-v2 of the question text (unit norm, 384-d).
Output: real_outputs/mmlu_<model>.jsonl (resumable), real_outputs/mmlu_<model>_embeddings.npy
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "real_outputs"; OUT.mkdir(exist_ok=True)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
PER_SUBJECT = int(sys.argv[2]) if len(sys.argv) > 2 else 45
tag = MODEL.split("/")[-1]
jl = OUT / f"mmlu_{tag}.jsonl"

from datasets import load_dataset
ds = load_dataset("cais/mmlu", "all", split="test")
by_subj: dict[str, list[int]] = {}
for i, s in enumerate(ds["subject"]):
    by_subj.setdefault(s, []).append(i)
rng = np.random.default_rng(0)
items = []
for subj in sorted(by_subj):
    idx = by_subj[subj]
    take = sorted(rng.choice(idx, size=min(PER_SUBJECT, len(idx)), replace=False).tolist())
    items += take
print("questions:", len(items), "subjects:", len(by_subj), flush=True)

done = set()
if jl.exists():
    for line in jl.open(encoding="utf-8"):
        done.add(json.loads(line)["idx"])

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(8)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
model.eval()
letters = ["A", "B", "C", "D"]
letter_ids = [tok.encode(" " + l, add_special_tokens=False)[-1] for l in letters]


def prompt(ex):
    subj = ex["subject"].replace("_", " ")
    s = f"The following is a multiple choice question about {subj}. Answer with the letter of the correct option.\n\n{ex['question']}\n"
    for l, c in zip(letters, ex["choices"]):
        s += f"{l}. {c}\n"
    return s + "Answer:"

t0 = time.time()
with jl.open("a", encoding="utf-8") as f:
    for k, i in enumerate(items):
        if i in done:
            continue
        ex = ds[i]
        ids = tok(prompt(ex), return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            logits = model(**ids).logits[0, -1]
        lp = torch.log_softmax(logits[letter_ids].float(), dim=-1)
        p = lp.exp().numpy()
        y = int(ex["answer"])
        rec = {"idx": i, "subject": ex["subject"], "question": ex["question"], "answer": y,
               "probs": [float(x) for x in p], "p_correct": float(p[y]), "risk": float(1 - p[y]),
               "error": int(np.argmax(p) != y)}
        f.write(json.dumps(rec) + "\n"); f.flush()
        if k % 50 == 0:
            print(f"{k}/{len(items)} elapsed {time.time()-t0:.0f}s", flush=True)

# embeddings in the same item order
from sentence_transformers import SentenceTransformer
recs = {}
for line in jl.open(encoding="utf-8"):
    r = json.loads(line); recs[r["idx"]] = r
order = [i for i in items if i in recs]
st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
emb = st.encode([recs[i]["question"] for i in order], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
np.save(OUT / f"mmlu_{tag}_embeddings.npy", emb)
with (OUT / f"mmlu_{tag}_order.json").open("w") as f:
    json.dump(order, f)
print("done", len(order), "mean risk", np.mean([recs[i]["risk"] for i in order]), "acc", 1 - np.mean([recs[i]["error"] for i in order]), flush=True)
