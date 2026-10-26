"""Public benchmark adapter notes for PromptShift-CRC.

This script creates a small manifest describing how prototype streams will be
replaced by public benchmark outputs. It does not download datasets by default,
which keeps the project lightweight and reproducible. When the optional
`datasets` package is available, the same task names can be used to pull
TruthfulQA, RealToxicityPrompts, and SummEval from Hugging Face in a later run.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


BENCHMARKS = [
    {
        "task": "truthfulness_qa",
        "public_dataset": "TruthfulQA",
        "risk_score": "incorrect_or_untruthful_answer",
        "shift_axis": "question category, e.g. health, law, finance, politics",
        "paper_role": "question answering and hallucination-sensitive risk",
    },
    {
        "task": "toxicity_safety",
        "public_dataset": "RealToxicityPrompts",
        "risk_score": "toxicity score or toxicity violation indicator",
        "shift_axis": "prompt toxicity, topic, demographic or style clusters",
        "paper_role": "safety and user-intent shift",
    },
    {
        "task": "summarization_factuality",
        "public_dataset": "SummEval",
        "risk_score": "one minus human consistency score, with relevance/coherence penalties",
        "shift_axis": "source domain and document type",
        "paper_role": "summarization factuality under source-domain shift",
    },
    {
        "task": "summarization_factuality_extensions",
        "public_dataset": "G-Eval/SelfCheckGPT-compatible summarization outputs",
        "risk_score": "judge-based factuality loss or self-consistency hallucination score",
        "shift_axis": "source domain, document type, and evaluator/model family",
        "paper_role": "model-output factuality extensions beyond SummEval",
    },
    {
        "task": "long_context_hallucination",
        "public_dataset": "LongHalQA / QHQK/description_hall_binary_v3",
        "risk_score": "binary hallucination mismatch label with length/detail penalties",
        "shift_axis": "description length, visual detail, and hallucination prevalence",
        "paper_role": "hallucination-sensitive long-context generation",
    },
]


def main() -> None:
    out = RESULTS / "public_benchmark_manifest.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(BENCHMARKS[0].keys()))
        writer.writeheader()
        writer.writerows(BENCHMARKS)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
