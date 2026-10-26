"""Real model outputs for a second task: TruthfulQA MC1 with the same open model on CPU.

For each question the model scores every answer choice by the summed log-probability of the choice text after
"Q: <question>\nA:"; p(choice) is the softmax over the choices; risk = 1 - p(correct choice) in [0, 1] and the 0/1
error is the argmax miss. Categories come from the generation split of the same 817 questions (join on question text).
Output: real_outputs/truthfulqa_<model>.jsonl (resumable).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "real_outputs"; OUT.mkdir(exist_ok=True)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
tag = MODEL.split("/")[-1]
jl = OUT / f"truthfulqa_{tag}.jsonl"

from datasets import load_dataset
mc = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
gen = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
cat = {r["question"]: r["category"] for r in gen}
done = set()
if jl.exists():
    done = {json.loads(l)["idx"] for l in jl.open(encoding="utf-8")}

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(8)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
t0 = time.time()
with jl.open("a", encoding="utf-8") as f:
    for i, ex in enumerate(mc):
        if i in done:
            continue
        choices = ex["mc1_targets"]["choices"]; labels = ex["mc1_targets"]["labels"]
        prefix = f"Q: {ex['question']}\nA:"
        pids = tok(prefix, add_special_tokens=False)["input_ids"]
        seqs, spans = [], []
        for c in choices:
            cids = tok(" " + c, add_special_tokens=False)["input_ids"]
            seqs.append(pids + cids); spans.append((len(pids), len(pids) + len(cids)))
        L = max(len(s) for s in seqs)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs])
        att = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs])
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=att).logits.float()
        lp = torch.log_softmax(logits, dim=-1)
        scores = []
        for r, (a, b) in enumerate(spans):
            tgt = ids[r, a:b]
            scores.append(float(lp[r, a - 1:b - 1].gather(1, tgt[:, None]).sum()))
        sc = np.array(scores); p = np.exp(sc - sc.max()); p /= p.sum()
        y = int(np.argmax(labels))
        rec = {"idx": i, "question": ex["question"], "category": cat.get(ex["question"], "Unknown"), "n_choices": len(choices),
               "p_correct": float(p[y]), "risk": float(1 - p[y]), "error": int(np.argmax(p) != y)}
        f.write(json.dumps(rec) + "\n"); f.flush()
        if i % 50 == 0:
            print(f"{i}/{len(mc)} elapsed {time.time()-t0:.0f}s", flush=True)
print("done", flush=True)
