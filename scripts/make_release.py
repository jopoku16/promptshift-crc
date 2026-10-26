"""Build the public release folder (release/promptshift-crc) from revision_v2.

Removes question and prompt text and generated text from the model-output files (offensive content in
RealToxicityPrompts; MMLU and TruthfulQA text is public and re-downloadable), keeps the cached embeddings so the
streams can be rebuilt, copies code, results, figures and the paper source. Does NOT push anything.
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REL = ROOT / "release" / "promptshift-crc"
if REL.exists():
    shutil.rmtree(REL)
(REL / "real_outputs").mkdir(parents=True)

KEEP = {
    "mmlu_Qwen2.5-1.5B-Instruct.jsonl": ["idx", "subject", "answer", "probs", "p_correct", "risk", "error"],
    "truthfulqa_Qwen2.5-1.5B-Instruct.jsonl": ["idx", "category", "n_choices", "p_correct", "risk", "error"],
    "rtp_gpt2.jsonl": ["idx", "prompt_toxicity", "challenging", "risk"],
    "rtp_qwen05b.jsonl": ["idx", "prompt_toxicity", "challenging", "risk"],
}
for name, keys in KEEP.items():
    src = ROOT / "real_outputs" / name
    if not src.exists():
        print("skip (missing)", name)
        continue
    with (REL / "real_outputs" / name).open("w", encoding="utf-8") as out:
        for line in src.open(encoding="utf-8"):
            r = json.loads(line)
            slim = {k: r[k] for k in keys if k in r}
            text = r.get("question", "")
            slim["qlen"] = len(text)
            out.write(json.dumps(slim) + "\n")
for npz in (ROOT / "real_outputs").glob("*_emb_cache.npz"):
    shutil.copy2(npz, REL / "real_outputs" / npz.name)

for folder in ["scripts", "results", "figures_v2"]:
    shutil.copytree(ROOT / folder, REL / folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "real_promptshift_summary.csv"))

paper = REL / "paper"
paper.mkdir()
for f in ["main_v2.tex", "main_v2.pdf", "main_v2.bbl", "numbers_generated.tex", "tables_generated.tex", "tables_appendix_generated.tex",
          "theory_section.tex", "theory_appendix.tex", "sec_intro_related_method.tex", "sec_protocol.tex", "sec_results_synthetic.tex",
          "sec_results_real.tex", "rtp_replication_text.tex", "sec_discussion.tex", "references_v2.bib", "new_refs.bib"]:
    if (ROOT / f).exists():
        shutil.copy2(ROOT / f, paper / f)
shutil.copytree(ROOT / "figures_v2", paper / "figures_v2")
for f in ["README.md", "LICENSE", "CITATION.cff", "requirements.txt", ".gitignore"]:
    if (ROOT / "release_files" / f).exists():
        shutil.copy2(ROOT / "release_files" / f, REL / f)
print("release built at", REL)
