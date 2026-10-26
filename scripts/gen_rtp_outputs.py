"""Real model outputs for a task whose risk has a shifting UPPER tail: RealToxicityPrompts with GPT-2 continuations.

Design (fixed before any result): 4000 prompts drawn at random (seed 0) from the 99,442 prompts; GPT-2 (124M) continues each
prompt with nucleus sampling (p = 0.9, temperature 1, 20 new tokens, one sample, seeds fixed per batch of 16); the risk is the
toxicity probability of the continuation alone from unitary/toxic-bert (the 'toxic' output). Arguments: generator, classifier, tag. Default: gpt2, unitary/toxic-bert, gpt2. Replication: Qwen/Qwen2.5-0.5B, unitary/unbiased-toxic-roberta, qwen05b.
Output: real_outputs/rtp_<tag>.jsonl
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "real_outputs"; OUT.mkdir(exist_ok=True)
N, BATCH = 4000, 16
GEN = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
CLF = sys.argv[2] if len(sys.argv) > 2 else "unitary/toxic-bert"
TAG = sys.argv[3] if len(sys.argv) > 3 else "gpt2"
jl = OUT / f"rtp_{TAG}.jsonl"

from datasets import load_dataset
ds = load_dataset("allenai/real-toxicity-prompts", split="train")
rng = np.random.default_rng(0)
items = sorted(rng.choice(len(ds), size=N, replace=False).tolist())
done = set()
if jl.exists():
    done = {json.loads(l)["idx"] for l in jl.open(encoding="utf-8")}

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
torch.set_num_threads(4)
gtok = AutoTokenizer.from_pretrained(GEN); gtok.padding_side = "left"
if gtok.pad_token is None:
    gtok.pad_token = gtok.eos_token
gm = AutoModelForCausalLM.from_pretrained(GEN).eval()
ctok = AutoTokenizer.from_pretrained(CLF)
cm = AutoModelForSequenceClassification.from_pretrained(CLF).eval()
_names = {i: str(l).lower() for i, l in cm.config.id2label.items()}
print("classifier labels:", _names, flush=True)
tox_idx = [i for i, l in _names.items() if l in ("toxic", "toxicity")]
tox_idx = tox_idx[0] if tox_idx else 0
todo = [i for i in items if i not in done]
t0 = time.time()
with jl.open("a", encoding="utf-8") as f:
    for b in range(0, len(todo), BATCH):
        chunk = todo[b:b + BATCH]
        rows = [ds[i] for i in chunk]
        prompts = [r["prompt"]["text"] for r in rows]
        torch.manual_seed(1000 + items.index(chunk[0]))
        enc = gtok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=96)
        with torch.no_grad():
            out = gm.generate(**enc, do_sample=True, top_p=0.9, temperature=1.0, max_new_tokens=20, pad_token_id=gtok.pad_token_id)
        conts = gtok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        cenc = ctok([c if c.strip() else "." for c in conts], return_tensors="pt", padding=True, truncation=True, max_length=64)
        with torch.no_grad():
            p = torch.sigmoid(cm(**cenc).logits[:, tox_idx]).numpy()
        for i, r, c, pr in zip(chunk, rows, conts, p):
            f.write(json.dumps({"idx": i, "question": r["prompt"]["text"], "prompt_toxicity": r["prompt"]["toxicity"],
                                "challenging": bool(r["challenging"]), "continuation": c, "risk": float(pr)}) + "\n")
        f.flush()
        if (b // BATCH) % 10 == 0:
            print(f"{b + len(chunk)}/{len(todo)} elapsed {time.time() - t0:.0f}s", flush=True)
print("done", flush=True)
