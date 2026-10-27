"""M1 smoke test: adapter, hook math, batched row scaling, recipes, and a few steered generations.

    uv run python tests/smoke_m1.py --model Qwen/Qwen3-0.6B
"""
import argparse
import sys
import tempfile

import torch

sys.path.insert(0, ".")
from neuron_finder import generate as G  # noqa: E402
from neuron_finder.model import load  # noqa: E402
from neuron_finder.steer import Intervention, Recipe, Steerer  # noqa: E402


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        check.failed += 1


check.failed = 0


@torch.no_grad()
def mlp_out(lm, layer, ids):
    got = {}
    h = lm.layers[layer].mlp.register_forward_hook(lambda m, a, o: got.__setitem__("o", o[0].float().clone()))
    try:
        lm.model(input_ids=torch.tensor([ids], device=lm.device))
    finally:
        h.remove()
    return got["o"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--dtype", default="float32", help="float32 makes the exactness checks tight")
    a = ap.parse_args()

    lm = load(a.model, dtype=a.dtype)
    print(f"{a.model}: {type(lm.model).__name__}, layers at {lm.layers_path}, {lm.n_layers} x {lm.d_mlp}, "
          f"d_model {lm.d_model}")
    check("calibration", len(lm.resid_norm) == lm.n_layers and all(x > 0 for x in lm.resid_norm),
          f"resid norms {lm.resid_norm[0]:.1f} .. {lm.resid_norm[-1]:.1f}")

    st = Steerer(lm)
    L, N = lm.n_layers // 2, 123
    ids = lm.chat_ids([{"role": "user", "content": "Describe the color orange."}])

    # 1. add-mode steering == adding value * W_down[:, n] to the MLP output, at every position
    base = mlp_out(lm, L, ids)
    st.set([Intervention(L, N, "+", "add", 0.8, "rel")])
    steered = mlp_out(lm, L, ids)
    want = st.value(st.interventions[0]) * lm.w_down_col(L, N).float()
    err = (steered - base - want).abs().max().item() / want.abs().max().item()
    check("add == W_down push", err < 1e-2, f"(rel err {err:.2e})")
    push = (st.value(st.interventions[0]) * lm.w_down_col(L, N).float()).norm().item()
    check("rel dose 0.8 = 0.8 x median resid norm", abs(push / lm.resid_norm[L] - 0.8) < 1e-3,
          f"(push {push:.2f}, resid {lm.resid_norm[L]:.2f})")

    # 2. negative pole flips the sign
    st.set([Intervention(L, N, "-", "add", 0.8, "rel")])
    neg = mlp_out(lm, L, ids)
    err = (neg - base + want).abs().max().item() / want.abs().max().item()
    check("'-' pole pushes the opposite way", err < 1e-2, f"(rel err {err:.2e})")

    # 3. ablation zeroes the activation
    st.set([Intervention(L, N, mode="mul", dose=0.0)])
    acts = G.activations(lm, ids, [(L, N)])
    with st.disabled():
        nat = G.activations(lm, ids, [(L, N)])
    check("mul 0 ablates", acts.abs().max().item() == 0.0 and nat.abs().max().item() > 0,
          f"(natural max |a| {nat.abs().max().item():.3f})")

    # 4. clamp pins the value
    st.set([Intervention(L, N, "+", "clamp", 3.0, "raw")])
    acts = G.activations(lm, ids, [(L, N)])
    check("clamp pins value", torch.allclose(acts, torch.full_like(acts, 3.0)))

    # 5. batched row scaling matches separate runs (greedy)
    st.set([Intervention(L, N, "+", "add", 1.0, "rel")])
    rows = G.generate(lm, st, [ids] * 3, max_new=40, row_scale=[-1, 0, 1])
    solo_up = G.generate(lm, st, [ids], max_new=40)[0]
    with st.disabled():
        solo_base = G.generate(lm, st, [ids], max_new=40)[0]
    check("row scale 0 == unsteered", rows[1] == solo_base)
    check("row scale 1 == steered", rows[2] == solo_up)
    check("steering changes output", rows[2] != rows[1] or rows[0] != rows[1])

    # 6. streaming matches batch generation
    streamed = "".join(G.stream(lm, st, ids, max_new=40))
    check("stream == generate", streamed == solo_up)

    # 7. recipe round-trip and spec parsing
    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        Recipe(model=a.model, interventions=st.interventions, property="test").save(f.name)
        r = Recipe.load(f.name)
    check("recipe round-trip", r.interventions == st.interventions and r.property == "test")
    iv = Intervention.parse("L3N7-:4:clamp:raw")
    check("spec parse", (iv.layer, iv.neuron, iv.pole, iv.mode, iv.dose, iv.unit) == (3, 7, "-", "clamp", 4.0, "raw"))

    # 8. dose-response on an open prompt, for eyeballing
    st.clear()
    prompt = lm.chat_ids([{"role": "user", "content": "Tell me about your weekend."}])
    for d in (0.0, 0.5, 1.0, 2.0):
        st.set([Intervention(L, N, "+", "add", d, "rel")] if d else [])
        print(f"\n--- L{L}N{N}+ dose {d} ---\n" + G.generate(lm, st, [prompt], max_new=60)[0])

    print(f"\n{check.failed} failed")
    sys.exit(1 if check.failed else 0)


if __name__ == "__main__":
    main()
