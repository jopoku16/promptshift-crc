#!/bin/bash
# waits for items A and C to finish, then runs items E, B1 and B2 one after another (each uses 7 processes)
cd "$(dirname "$0")"
R=../results
until [ -f "$R/real_mmlu_Qwen2.5-1.5B-Instruct_alpha0.3_ps_vs_matched.csv" ] && [ -f "$R/round4_trace_synthetic.pkl" ]; do sleep 60; done
run() { echo "=== $(date +%T) $*"; "$@"; echo "=== done $(date +%T) $*"; }
run python run_round4_larc.py synth
run python run_round4_larc.py real mmlu
run python run_round4_larc.py real rtp
run python run_round4_larc.py real rtp --tag qwen05b
run python run_round4_larc.py real truthfulqa
run python run_round4_sweep.py sweep rtp --tag gpt2
run python run_round4_sweep.py sweep rtp --tag qwen05b
run python run_round4_sweep.py sweep mmlu
run python run_round4_sweep.py recur rtp --tag gpt2
run python run_round4_sweep.py recur rtp --tag qwen05b
run python run_round4_sweep.py recur mmlu
run python run_round4_sweep.py recur mmlu --alpha 0.2
run python run_round4_sweep.py recur mmlu --alpha 0.3
echo "=== ALL DONE $(date +%T)"
