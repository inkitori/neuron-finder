# Source from inside a Slurm job (sbatch or salloc shell) before running nf.
# /gscratch/scrubbed is nearly full and /gscratch/ark is at its file-count quota, so the venv is built on the
# job's node-local SSD ($TMPDIR) from the existing uv cache; `uv sync` takes about a minute per job.
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-${TMPDIR:-/tmp}/nf-venv}
export UV_LINK_MODE=copy
export NF_CACHE_DIR=${NF_CACHE_DIR:-/gscratch/ark/anjo0/neuron-finder/.cache}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}   # models are already in $HF_HOME; unset to download new ones
# Triton kernel cache (compiled for Qwen3.5's linear attention): ~/.bashrc points it at scrubbed, which is full
export TRITON_CACHE_DIR=${TMPDIR:-/tmp}/nf-triton
