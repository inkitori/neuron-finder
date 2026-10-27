# neuron-finder

Find and steer raw MLP neurons in local Qwen models from a natural-language property, then chat with the steered
model. See `DESIGN.md` for how it works.

```bash
uv run nf find "talks about the ocean" -m Qwen/Qwen3-8B -o ocean.json   # property -> verified recipe
uv run nf chat -m Qwen/Qwen3-8B -r ocean.json                          # chat with it applied
```

## Setup (Hyak)

Run model code inside a Slurm job:

```bash
salloc -p gpu-l40s -A ark --gres=gpu:1 --mem-per-gpu=64G --cpus-per-gpu=4 --time=2:00:00 --exclude=g3110
cd /gscratch/ark/anjo0/neuron-finder
source scripts/env.sh          # venv on node-local disk, index/cache on ark, offline HF
uv sync --extra qwen35         # qwen35 = fast linear-attention kernels for Qwen3.5
export ANTHROPIC_API_KEY=...   # for `nf find` (or `ant auth login`)
```

## Commands

| command | what it does |
|---|---|
| `nf info -m MODEL` | layers, d_mlp, per-layer residual norms |
| `nf index -m MODEL` | one pass over 2M FineWeb-Edu tokens: per-neuron stats, top contexts for both poles, logit lens. One `.npz` per model in `.cache/` (Qwen3-0.6B: 30 s / 33 MB on an L40S) |
| `nf inspect -m MODEL L4N2729+` | what a neuron fires on, what it writes, steered samples |
| `nf find "PROPERTY" -m MODEL -o r.json` | Claude writes contrast texts → score every neuron pole → steer the best (plus composites and random controls) → blind Claude judge → stack levers → recipe + `r.report.json` |
| `nf gen -m MODEL -s L14N123+:0.8 --compare "prompt"` | one-shot steered generation |
| `nf chat -m MODEL [-r recipe.json]` | interactive chat; `/help` inside |

`nf find` without credentials: `--contrast FILE --judge local` runs fully offline (the loaded model judges itself;
fine for testing, poorly calibrated for real use). `tests/fixtures/ocean.json` shows the contrast-set format.

## Neuron IDs and doses

`L14N2742+` = layer 14, neuron 2742, positive pole (`-` pushes it negative). A spec is `ID[:DOSE[:MODE|UNIT]]`:

| mode    | effect                                  |
|---------|-----------------------------------------|
| `add`   | activation += dose (default)            |
| `mul`   | activation ×= dose (`0` = ablate)       |
| `clamp` | activation = dose                       |

Units: `rel` (default) scales the push to a fraction of the layer's median residual norm, so dose means the same
thing for every neuron (≈0.4–0.8 is coherent, ≥1.5 tends to loop). `raw` is raw activation units. `absmax` is a
multiple of the pole's largest corpus activation (needs `nf index`).

Chat commands: `/steer`, `/ablate`, `/clamp`, `/unsteer`, `/list`, `/scale`, `/compare` (▼ / 0 / ▲ side by side),
`/fires` (per-token activations on your text), `/logprobs`, `/inspect`, `/find`, `/think`, `/save`, `/load`,
`/reset`.

## Tests

`sbatch scripts/smoke_m1.sbatch` (steering math, generation paths) and `sbatch scripts/smoke_m3.sbatch` (readouts,
per-row steering, composite identity, then index → offline find → inspect).
