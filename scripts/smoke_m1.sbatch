#!/bin/bash
#SBATCH --partition=gpu-l40s
#SBATCH --account=ark
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=48G
#SBATCH --cpus-per-gpu=4
#SBATCH --time=0:45:00
#SBATCH --job-name=nf_smoke_m1
#SBATCH --output=scripts/%x/%j.out
#SBATCH --error=scripts/%x/%j.err
#SBATCH --exclude=g3110
# Builds/updates the uv env, then runs the M1 smoke test on a small Qwen3 and a Qwen3.5 checkpoint.
set -euo pipefail
cd /gscratch/ark/anjo0/neuron-finder
source scripts/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
uv sync --extra qwen35 2>&1 | tail -3

uv run python tests/smoke_m1.py --model ${MODEL:-Qwen/Qwen3-0.6B}
uv run nf info --model Qwen/Qwen3.5-0.8B
uv run nf gen --model Qwen/Qwen3.5-0.8B --compare -s L12N100+:1.0 "Who are you?"
