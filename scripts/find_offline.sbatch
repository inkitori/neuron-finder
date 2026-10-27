#!/bin/bash
#SBATCH --partition=gpu-l40s
#SBATCH --account=ark
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-gpu=4
#SBATCH --time=1:30:00
#SBATCH --job-name=nf_find
#SBATCH --output=scripts/%x/%j.out
#SBATCH --error=scripts/%x/%j.err
#SBATCH --exclude=g3110
# Offline `nf find` from a hand-written contrast set (no Claude key needed), then inspect the top hits.
#   sbatch --export=ALL,FIXTURE=tests/fixtures/golden_gate.json,PROP="talks about the Golden Gate Bridge" scripts/find_offline.sbatch
set -euo pipefail
cd /gscratch/ark/anjo0/neuron-finder
source scripts/env.sh
export COLUMNS=160
M=${MODEL:-Qwen/Qwen3-8B}
NAME=$(basename ${FIXTURE%.json})
OUT=scripts/nf_find/${NAME}_${SLURM_JOB_ID}.json
uv sync --extra qwen35 2>&1 | tail -1
uv run python -c "from neuron_finder.model import load; from neuron_finder.index import index_path; import sys; sys.exit(0 if index_path('$M').exists() else 1)" \
  || uv run nf index -m $M --batch 16
uv run nf find "$PROP" -m $M --contrast $FIXTURE --judge local --batch 32 --ks ${KS:-4,8,16,32} -o $OUT
TOP=$(uv run python -c "import json;r=json.load(open('${OUT%.json}.report.json'));c=[x for x in r['candidates'] if x['kind']=='single' and x['info'].get('best')];c.sort(key=lambda x:-x['info']['best']['lift']);print(' '.join(x['name'] for x in c[:3]))")
uv run nf inspect $TOP -m $M --contexts 8 --steer 0.5,0.8,1.2
