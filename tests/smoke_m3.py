"""M2/M3 checks: per-row steering, AUROC, readouts, composite decomposition identity.

    uv run python tests/smoke_m3.py --model Qwen/Qwen3-0.6B
"""
import argparse
import sys

import torch

sys.path.insert(0, ".")
from neuron_finder import generate as G  # noqa: E402
from neuron_finder.find import _wrap, auroc, readouts  # noqa: E402
from neuron_finder.model import load  # noqa: E402
from neuron_finder.steer import Intervention, Steerer  # noqa: E402

failed = 0


def check(name, ok, detail=""):
    global failed
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    failed += not ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    a = ap.parse_args()
    lm = load(a.model, dtype="float32")
    st = Steerer(lm)
    L = lm.n_layers // 2
    ids = lm.chat_ids([{"role": "user", "content": "Describe the color orange."}])

    # per-row steering == persistent steering, row by row
    rows = [[], [Intervention(L, 5, "+", "add", 1.0, "rel")],
            [Intervention(L, 5, "-", "add", 1.0, "rel"), Intervention(L + 2, 77, "+", "add", 0.6, "rel")]]
    got = G.generate_per_row(lm, st, [ids] * 3, rows, max_new=30)
    ok = True
    for r, g in zip(rows, got):
        st.set(r)
        want = G.generate(lm, st, [ids], max_new=30)[0]
        ok &= lm.tok.decode(g, skip_special_tokens=True) == want
    st.clear()
    check("per_row == persistent steering", ok)
    check("per_row rows differ", len({tuple(g) for g in got}) == 3)

    # AUROC sanity
    x = torch.tensor([[3.0, 0.0], [4.0, 1.0]])
    y = torch.tensor([[1.0, 5.0], [2.0, 6.0]])
    au = auroc(x, y)
    check("auroc", torch.allclose(au, torch.tensor([1.0, 0.0])), f"{au.tolist()}")

    # readouts agree with a direct activation read on the content span
    texts = ["The ocean was calm and blue.", "Add the flour, then stir slowly."]
    R, resid = readouts(lm, texts, batch=2)  # batched with left padding
    wid, content, summ = _wrap(lm, texts[0])
    acts = G.activations(lm, wid, [(L, n) for n in range(lm.d_mlp)])  # [T, D]
    direct_max = acts[content].max(0).values
    direct_summ = acts[summ].mean(0)
    e1 = (R["max"][0, L] - direct_max).abs().max().item()
    e2 = (R["summ"][0, L] - direct_summ).abs().max().item()
    check("readout max matches direct read (padded batch)", e1 < 1e-3, f"(err {e1:.1e})")
    check("readout summary matches direct read", e2 < 1e-3, f"(err {e2:.1e}, {len(summ)} summary positions)")
    check("summary positions include end of turn", summ[-1] == content[-1])

    # composite identity: the mean MLP-output difference is exactly sum_n dA_n * W_down[:, n]
    dA = R["mean"][0, L] - R["mean"][1, L]
    W = lm.down_proj(L).weight.float().cpu()
    outs = []
    for t in texts:
        w, c, _ = _wrap(lm, t)
        got = {}
        h = lm.layers[L].mlp.register_forward_hook(lambda m, i, o: got.__setitem__("o", o[0].float().cpu()))
        lm.model(input_ids=torch.tensor([w], device=lm.device))
        h.remove()
        outs.append(got["o"][c].mean(0))
    e3 = ((outs[0] - outs[1]) - W @ dA).abs().max().item() / (outs[0] - outs[1]).abs().max().item()
    check("Δmlp_out == W_down · Δa", e3 < 1e-3, f"(rel err {e3:.1e})")

    print(f"\n{failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
