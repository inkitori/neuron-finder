#!/bin/bash
#SBATCH --partition=gpu-l40s
#SBATCH --account=ark
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-gpu=4
#SBATCH --time=2:00:00
#SBATCH --job-name=nf_validate
#SBATCH --output=scripts/%x/%j.out
#SBATCH --error=scripts/%x/%j.err
#SBATCH --exclude=g3110
# Index + offline find on a mid-size model; also checks the installed anthropic SDK accepts our request shape.
set -euo pipefail
cd /gscratch/ark/anjo0/neuron-finder
source scripts/env.sh
export COLUMNS=160
M=${MODEL:-Qwen/Qwen3-8B}
uv sync --extra qwen35 2>&1 | tail -1
# request-shape check: a bogus key should get a 401 from the API, not a client-side TypeError
ANTHROPIC_API_KEY=invalid uv run python -c "
from neuron_finder import llm
try:
    llm.contrast_set('test', n=8)
except RuntimeError as e:
    print('SDK request shape OK ->', e)
except TypeError as e:
    print('SDK request shape BROKEN ->', e); raise
"
uv run nf index -m $M --tokens ${NTOK:-2000000} --batch 16
uv run nf find "talks about the ocean or the sea" -m $M --contrast tests/fixtures/ocean.json --judge local \
  --batch 32 -o scripts/nf_validate/ocean_${SLURM_JOB_ID}.json
