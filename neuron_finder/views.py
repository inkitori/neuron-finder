"""Rich renderings shared by the CLI and the chat REPL."""
from rich.console import Console
from rich.table import Table
from rich.text import Text

from . import generate as G
from .model import LoadedModel
from .steer import Intervention, Steerer

INSPECT_PROMPTS = ["Tell me about your weekend.", "Write a short poem about the ocean."]


def show_neuron(console: Console, lm: LoadedModel, steerer: Steerer, layer: int, neuron: int, pole: str,
                n_ctx: int = 8, steer_doses: tuple = (), max_new: int = 60):
    idx = lm.index
    console.rule(f"L{layer}N{neuron}{pole}")
    cn = float(lm.col_norms(layer)[neuron])
    if idx is None:
        console.print("[yellow]no index for this model; run `nf index` for corpus stats and contexts[/yellow]")
    else:
        console.print(f"corpus: mean {idx.mean[layer, neuron]:+.3f}  std {idx.std[layer, neuron]:.3f}  "
                      f"max {idx.max[layer, neuron]:+.3f}  min {idx.min[layer, neuron]:+.3f}  "
                      f"frac>0 {idx.frac_pos[layer, neuron]:.2f}   |W_down col| {cn:.3f}")
        for p in ([pole, "-" if pole == "+" else "+"]):
            console.print(f"\n[bold]{'positive' if p == '+' else 'negative'} pole[/bold]"
                          f"{'' if p == pole else ' (other side)'}  "
                          f"writes → {' '.join(repr(t) for t in idx.lens(layer, neuron, p))}")
            for c in idx.contexts(layer, neuron, p, n=n_ctx if p == pole else max(3, n_ctx // 3)):
                line = Text(f"{c.value:+7.2f}  ")
                line.append(c.left.replace("\n", "⏎")[-90:], style="dim")
                line.append(c.token.replace("\n", "⏎"), style="bold green" if p == "+" else "bold red")
                line.append(c.right.replace("\n", "⏎"), style="dim")
                console.print(line, overflow="ellipsis", no_wrap=True)
    for prompt in INSPECT_PROMPTS if steer_doses else []:
        ids = lm.chat_ids([{"role": "user", "content": prompt}])
        rows = [[]] + [[Intervention(layer, neuron, pole, "add", d, "rel")] for d in steer_doses]
        outs = G.generate_per_row(lm, steerer, [ids] * len(rows), rows, max_new)
        t = Table(title=prompt, expand=True, show_lines=True)
        t.add_column("dose", width=6)
        t.add_column("output", overflow="fold")
        for d, o in zip((0.0,) + tuple(steer_doses), outs):
            t.add_row(f"{d:g}", lm.tok.decode(o, skip_special_tokens=True))
        console.print(t)


def show_find_report(console: Console, report: dict, top: int = 12):
    scored = report.get("scored", [])
    t = Table("neuron", "readout", "effect", "AUROC vs near-miss", "AUROC vs control", title="detector candidates")
    for r in scored[:top]:
        t.add_row(f"L{r['layer']}N{r['neuron']}{r['pole']}", r["readout"], f"{r['effect']:.2f}",
                  f"{r['auroc_tn']:.2f}", f"{r['auroc_tc']:.2f}")
    console.print(t)
    conc = report.get("concentration", {}).get("positive_share_by_k", {})
    if conc:
        console.print("composite concentration (share of positive MLP contribution): " +
                      ", ".join(f"top{k} {v:.0%}" for k, v in conc.items()))
    base = report["baseline"]["property"]
    ctrl = report["control_property"]
    console.print(f"baseline property {base:.1f}/10; random-neuron controls " +
                  ", ".join(f"dose {d}: {v:.1f}" for d, v in ctrl.items()))
    t = Table("candidate", "kind", "best dose", "property", "coherence", "lift", title="verification (steering)")
    rows = [c for c in report["candidates"] if c["info"].get("best")]
    rows.sort(key=lambda c: -c["info"]["best"]["lift"])
    for c in rows[:top + 6]:
        b = c["info"]["best"]
        style = "green" if c["kind"] != "control" and b["lift"] >= report["config"]["min_lift"] else (
            "dim" if c["kind"] == "control" else "")
        t.add_row(c["name"], c["kind"], f"{b['dose']:g}", f"{b['property']:.1f}", f"{b['coherence']:.1f}",
                  f"{b['lift']:+.1f}", style=style)
    console.print(t)
    rec = report.get("recipe")
    if rec:
        s = rec["scores"]
        console.print(f"[bold green]recipe[/bold green]: {len(rec['interventions'])} interventions from "
                      f"{', '.join(s.get('members', []))}; property {s['property']:.1f} vs baseline "
                      f"{s['baseline_property']:.1f}, coherence {s['coherence']:.1f}")
    else:
        console.print("[yellow]no candidate beat the controls at acceptable coherence; see the report for "
                      "near misses (try a more specific property, or a bigger model)[/yellow]")
