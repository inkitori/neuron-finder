# neuron-finder — design

Find and steer raw MLP neurons (or small groups of them) in local Qwen models from a natural-language
property ("sarcastic", "wants to talk about the ocean"), then chat with the steered model from a CLI.

## Lineage

- **Goodfire, "Neural Networks Are Already Clean"** (Cammarata, https://already-clean.goodfire.com/) — the thesis.
  Raw MLP neurons in modern LMs are mostly monosemantic (rough Qwen3-32B audit: ~90% of neuron *poles* have one
  job), so we work with the model's own neurons, not an SAE. Takeaways we build in:
  - the unit is a **neuron pole**: positive and negative activations are separate concepts;
  - SwiGLU neuron = `silu(gate·x) * (up·x)` (the input to `down_proj`); sign regimes (gate × up) matter;
  - context/stance neurons fire at **summary positions** (`.`, `,`, `\n`, end of turn) and often on the token
    *before* the thing they predict;
  - **steering** = turn a neuron up and read what the model writes.
- **lyramakesmusic/neuron-explorer** (https://github.com/lyramakesmusic/neuron-explorer) — the tooling, for
  Qwen3.5-4B. Sweep-first: steer *every* neuron of a layer on fixed prompts, rank by distinctiveness versus
  random-direction controls, browse in a web UI with a Claude agent attached. We copy:
  - steering neuron *n* = adding `dose · W_down[:, n]` to the MLP output (equivalently, adding to activation *n*);
  - **relative dose units**: push size as a fraction of the median residual norm leaving the layer; coherent
    window ≈ 0.5–0.8, ≥1.5–2 tends to collapse into loops;
  - random-direction controls as the null;
  - the "finding a lever for a behaviour" methodology (`explorer/agent_methodology.md`): contrast target /
    same-topic-no-stance / same-stance-other-object texts, keep neurons that fire only on the target, steer
    candidates on unrelated prompts, reject word-writers, steer drivers of detectors that steer badly, stack
    2–3 levers at lower doses, report a recipe.
- **Prior local work**: `/gscratch/scrubbed/anjo0/already_clean` (Golden Gate neurons in Qwen3.5-27B). Lessons:
  position-0 / document-boundary tokens carry sink-like activations and must be masked; contrast z-scores favour
  early layers; single mid-layer neurons often barely move generations, which is why composites matter.

**Difference from neuron-explorer:** query-first, not sweep-first. We search from the user's property and only
verify a few dozen candidates causally, so it is cheap enough to run on demand for any Qwen.

## Pipeline

```
            ┌──────────────── one-time per model (cached) ────────────────┐
 load Qwen →│ arch adapter → stats pass (≈1–5M tokens) → neuron index      │
            │   abs_max per pole, quantiles, top-k contexts, logit lens    │
            └──────────────────────────────────────────────────────────────┘
 "sarcastic" ─→ 1. contrast sets ─→ 2. candidate scoring ─→ 3. causal verify ─→ 4. compose ─→ recipe.json
                (Claude writes       (detector stats + dense  (steer on neutral    (stack + tune
                 target / near-miss   direction decomposed     prompts vs random   doses under a
                 / distant)           onto neurons)            controls, judged)   coherence budget)
```

### 0. Model adapter and hooks
- Plain `transformers` + forward hooks (no vLLM fork), so any dense Qwen (2.5 / 3 / 3.5) works. The adapter
  locates the decoder-layer `ModuleList` whose layers have `mlp.down_proj`; Qwen3.5 lives under
  `model.model.language_model`, Qwen3 under `model.model`.
- **Read**: forward pre-hook on `down_proj` → all neuron activations `[B, T, d_mlp]`.
- **Write**: same pre-hook edits activation *n*. Modes: `add` (dose), `mul` (0 = ablate), `clamp` (set value).
  Per-batch-row scaling so ▼/0/▲ comparisons and verification sweeps run as one batch.
- MoE (Qwen3-30B-A3B etc.) is out of scope for now: a neuron would be (layer, expert, idx) and routing makes
  steering conditional.

### 1. Per-model index (M2)
Stream ≈2M FineWeb-Edu tokens; per neuron pole keep raw abs_max, quantiles, sign-regime counts, top-32 contexts
(as corpus offsets). Static: logit lens of ±`W_down[:, n]`. Mask position 0 and document boundaries.

### 2. Query → candidates (M3)
- **Contrast sets (Claude API)**: ~10 triples of target / near-miss (same topic, no property) / distant control,
  plus neutral steering prompts and a rubric for the judge. Structured output → JSON.
- **Detector score**: per pole AUROC / effect size, target vs near-miss and target vs control, at each text's
  max token and at summary positions. Keep poles that beat both.
- **Composite score**: `v_L = mean(resid | P) − mean(resid | ¬P)`, decomposed as a sparse combination of the
  layer's `W_down` columns (OMP / lasso) → neurons whose writes add up to P, plus explained variance (tells us
  whether P is concentrated or genuinely distributed).
- Optional: logit-lens match for late "say X" neurons.

### 3. Causal verification (M3)
Steer top ~30–60 candidates on neutral prompts at rel dose ≈0.5 / 0.8 / 1.5. Score:
- **P-presence**: Claude judge (with the rubric from step 1), batched;
- **coherence**: repetition rate, perplexity under the unsteered model;
- **null**: random-direction controls of equal norm;
- logprob fork tests ("Right now I'd love a ___").

### 4. Compose (M4)
Greedy stacking of verified levers with doses scaled down; coordinate ascent on P-score under a
coherence/KL budget. v2: driver search (ablate upstream at a detector's peak, steer the strongest drivers).

### Recipe format
```json
{"model": "Qwen/Qwen3-8B", "property": "sarcastic",
 "interventions": [
   {"layer": 15, "neuron": 3920, "pole": "+", "mode": "add", "dose": 0.6, "unit": "rel"},
   {"layer": 12, "neuron": 2920, "pole": "-", "mode": "add", "dose": 0.4, "unit": "rel"}],
 "scores": {"p_score": 0.81, "coherence": 0.93, "control_p_score": 0.07}}
```

### CLI
```
nf info   --model Qwen/Qwen3-0.6B                    # architecture summary
nf gen    --model ... -s L14N123+:0.8 "prompt"        # one-shot steered generation (scripting/tests)
nf chat   --model ... [--recipe r.json]               # interactive chat
nf index  --model ...                                 # (M2)
nf inspect --model ... L22N2742+                      # (M2)
nf find   --model ... "wants to talk about the ocean" -o ocean.json   # (M3)
```
Chat commands: `/steer`, `/ablate`, `/clamp`, `/unsteer`, `/list`, `/scale`, `/compare` (▼ / 0 / ▲ side by
side), `/fires`, `/logprobs`, `/think`, `/save`, `/load`, `/reset`, `/find` (M3).

## Claude API usage (M3+)
- Python `anthropic` SDK, model `claude-opus-5-5` by default (override with `NF_CLAUDE_MODEL`), structured
  outputs for contrast sets and judge verdicts.
- Credentials: `ANTHROPIC_API_KEY` (or an `ant auth login` profile).
- Compute nodes reach the internet through a shared proxy: keep concurrent requests low (≤4) and batch judge
  items into a few requests rather than one call per rollout.

## Milestones
1. **M1**: adapter, hooks, interventions + recipes, `info` / `gen` / `chat` with manual steering. Verified on
   Qwen3-0.6B.
2. **M2**: index pass, `inspect`, abs_max dose unit.
3. **M3**: `find` (Claude contrast sets → scoring → verification) and recipe output.
4. **M4**: sparse decomposition, composition, dose tuning; scale to Qwen3-8B / Qwen3.5-9B.
5. Later: MoE, driver search, optional vLLM backend for exhaustive sweeps.

## Environment (Hyak)
- `uv` project; the venv lives on scratch (`UV_PROJECT_ENVIRONMENT=/gscratch/scrubbed/$USER/venvs/neuron-finder`)
  because `/gscratch/ark` is near its inode quota. Scratch is purged after ~21 days idle; `uv sync` rebuilds it.
- Caches (residual-norm calibration, indices) go to `$NF_CACHE_DIR`, default `$XDG_CACHE_HOME/neuron-finder`.
- Everything model-related runs inside a Slurm job (`scripts/` has sbatch helpers).
