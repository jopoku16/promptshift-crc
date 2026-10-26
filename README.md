# PromptShift-CRC: conformal risk control for foundation models under prompt shift

Code, model outputs, result tables and the paper source for

> Opoku, J. and Banahene, D. *Conformal Risk Control for Foundation Models Under Prompt Shift: What Weighting Adds to the Online Update.*

## What this is

A study of what a drift-aware weighting rule (PromptShift-CRC) adds to an online conformal risk control update when the prompts a model receives change over time. The short answer, which the paper supports with theory and experiments, is:

- The long-run violation rate is controlled by the online update for any weights (Theorem 2). Marginal coverage cannot separate weighting schemes.
- PromptShift-CRC stays within about half a percentage point of the Adaptive CRC curve at equal sharpness, avoids the worst Adaptive CRC window, and is matched or beaten by DtACI in most settings. Its value is that no window has to be chosen.
- On MMLU and TruthfulQA the upper tail of the risk does not move between domains, so no weighting can help. On RealToxicityPrompts it does move, and the same conclusions hold.

## Layout

- `scripts/` experiment code. `crc_lib.py` holds all calibration rules, `baselines_online.py` the stronger baselines, `run_*` the drivers, `gen_*` the model-output generators, `analyze_*` and `make_*` the analysis, tables and figures, `check_claims.py` the checks that every sentence of the paper's results is supported by the result files.
- `results/` result tables (CSV) produced by the drivers.
- `real_outputs/` model outputs: risk scores, answer probabilities and cached prompt embeddings. The text of MMLU and TruthfulQA questions is public and is not copied. RealToxicityPrompts prompts and continuations contain offensive language and are not included; `gen_rtp_outputs.py` regenerates them.
- `figures_v2/` the figures. `paper/` the paper source and PDF.

## Reproduce

Everything runs on a CPU. Python 3.11 or later; `pip install -r requirements.txt`.

```bash
# synthetic streams (30 seeds) and stronger baselines
python scripts/run_phase_a_synthetic.py
python scripts/run_phase_a_baselines.py
# real-output streams from the released outputs and embeddings
python scripts/run_real_streams.py mmlu
python scripts/run_real_streams.py truthfulqa
python scripts/run_real_streams.py rtp
python scripts/run_real_streams.py rtp --tag qwen05b        # replication: second generator and classifier
python scripts/run_real_streams.py rtp --alpha 0.05 --only bandblock   # and --alpha 0.2
python scripts/analyze_frontier.py && python scripts/analyze_real.py mmlu && python scripts/analyze_real.py truthfulqa && python scripts/analyze_real.py rtp && python scripts/analyze_real.py rtp:qwen05b && python scripts/analyze_real.py rtp:gpt2:0.05 && python scripts/analyze_real.py rtp:gpt2:0.2
python scripts/make_tables.py && python scripts/make_figures.py all && python scripts/check_claims.py
```

To regenerate the model outputs themselves (several hours on a CPU): `gen_mmlu_outputs.py`, `gen_truthfulqa_outputs.py`, `gen_rtp_outputs.py` (replication: `python scripts/gen_rtp_outputs.py Qwen/Qwen2.5-0.5B unitary/unbiased-toxic-roberta qwen05b`). Models: Qwen2.5-1.5B-Instruct, Qwen2.5-0.5B (Apache-2.0), GPT-2 (MIT), toxic-bert and unbiased-toxic-roberta (Apache-2.0), all-MiniLM-L6-v2 (Apache-2.0). Data: MMLU (MIT), TruthfulQA (Apache-2.0), RealToxicityPrompts (Apache-2.0).

## Caveats

The model-output experiments use small open models, one prompt template and one embedding model. Only RealToxicityPrompts has an upper tail that moves with the prompt. The synthetic generators and their constants come from an earlier version of this work. The paper's Limitations section lists these and the others.

## Licence and citation

MIT licence (`LICENSE`). See `CITATION.cff`. An earlier version of this work is arXiv:2606.15964; this repository supersedes the code of that version.
