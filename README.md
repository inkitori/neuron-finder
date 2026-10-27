# neuron-finder

Find and steer raw MLP neurons in local Qwen models, and chat with the steered model. See `DESIGN.md` for how it
works and the roadmap. Status: **M1** (manual steering, chat, probing). `nf find` (property → neurons via the
Claude API) is M3.

## Setup (Hyak)

Run model code inside a Slurm job:

```bash
salloc -p gpu-l40s -A ark --gres=gpu:1 --mem-per-gpu=48G --cpus-per-gpu=4 --time=2:00:00 --exclude=g3110
cd /gscratch/ark/anjo0/neuron-finder
source scripts/env.sh          # venv + caches on /gscratch/scrubbed
uv sync --extra qwen35         # qwen35 = fast linear-attention kernels for Qwen3.5
```

## Usage

```bash
uv run nf info -m Qwen/Qwen3-0.6B                       # layers, d_mlp, residual norms
uv run nf gen  -m Qwen/Qwen3-0.6B -s L14N123+:0.8 --compare "Tell me about your weekend."
uv run nf chat -m Qwen/Qwen3-8B [-r recipe.json]
```

Neuron IDs: `L14N2742+` = layer 14, neuron 2742, positive pole (`-` pushes it negative). A spec is
`ID[:DOSE[:MODE|UNIT]]`:

| mode    | effect                                  |
|---------|-----------------------------------------|
| `add`   | activation += dose (default)            |
| `mul`   | activation ×= dose (`0` = ablate)       |
| `clamp` | activation = dose                       |

Units: `rel` (default) scales the push to a fraction of the layer's median residual norm, so dose means the same
thing for every neuron (≈0.5–0.8 is coherent, ≥1.5–2 tends to loop). `raw` is raw activation units.

In chat, `/help` lists commands: `/steer`, `/ablate`, `/clamp`, `/unsteer`, `/list`, `/scale`, `/compare`
(▼ / 0 / ▲ side by side), `/fires` (per-token activations on your text), `/logprobs`, `/think`, `/save`,
`/load`, `/reset`.

Tests: `uv run python tests/smoke_m1.py --model Qwen/Qwen3-0.6B` (or `sbatch scripts/smoke_m1.sbatch`).
