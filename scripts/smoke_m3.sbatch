#!/bin/bash
#SBATCH --partition=gpu-l40s
#SBATCH --account=ark
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=48G
#SBATCH --cpus-per-gpu=4
#SBATCH --time=1:30:00
#SBATCH --job-name=nf_smoke_m3
#SBATCH --output=scripts/%x/%j.out
#SBATCH --error=scripts/%x/%j.err
#SBATCH --exclude=g3110
# M2/M3 end to end on a small model: unit checks, index, inspect, and an offline find (fixture + local judge).
set -euo pipefail
cd /gscratch/ark/anjo0/neuron-finder
source scripts/env.sh
export COLUMNS=160
M=${MODEL:-Qwen/Qwen3-0.6B}
uv sync --extra qwen35 2>&1 | tail -2
uv run python tests/smoke_m3.py --model $M
uv run nf index -m $M --tokens ${NTOK:-500000} --batch 32
uv run nf find "talks about the ocean or the sea" -m $M --contrast tests/fixtures/ocean.json --judge local \
  -o scripts/nf_smoke_m3/ocean_${SLURM_JOB_ID}.json
TOP=$(uv run python -c "import json;r=json.load(open('scripts/nf_smoke_m3/ocean_${SLURM_JOB_ID}.report.json'))['scored'][:2];print(' '.join(f\"L{x['layer']}N{x['neuron']}{x['pole']}\" for x in r))")
uv run nf inspect $TOP -m $M --contexts 6
