"""`nf` command line."""
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Find and steer MLP neurons in local Qwen models.")
console = Console(highlight=False)

ModelOpt = Annotated[str, typer.Option("--model", "-m", help="HF model id or local path")]
DeviceOpt = Annotated[Optional[str], typer.Option(help="cuda / cpu (default: cuda if available)")]
SteerOpt = Annotated[list[str], typer.Option("--steer", "-s", help="ID[:DOSE[:MODE|UNIT]], repeatable, e.g. L14N2742+:0.8")]
RecipeOpt = Annotated[Optional[str], typer.Option("--recipe", "-r", help="recipe JSON to apply")]


def _setup(model: str, device: str | None, steer: list[str], recipe: str | None):
    from .model import load
    from .steer import Intervention, Recipe, Steerer

    with console.status(f"loading {model} ..."):
        lm = load(model, device=device)
    steerer = Steerer(lm)
    ivs = []
    if recipe:
        r = Recipe.load(recipe)
        if r.model != model:
            console.print(f"[yellow]recipe was made for {r.model}, loading into {model}[/yellow]")
        ivs += r.interventions
    ivs += [Intervention.parse(s) for s in steer]
    steerer.set(ivs)
    return lm, steerer


@app.command()
def info(model: ModelOpt = "Qwen/Qwen3-0.6B", device: DeviceOpt = None,
         recalibrate: Annotated[bool, typer.Option(help="recompute the residual-norm calibration")] = False):
    """Architecture summary and per-layer residual norms (the scale of 'rel' doses)."""
    from .model import load

    with console.status(f"loading {model} ..."):
        lm = load(model, device=device)
    if recalibrate:
        lm.calibrate(force=True)
    console.print(f"[bold]{model}[/bold]  ({type(lm.model).__name__})")
    console.print(f"decoder layers: {lm.layers_path}  ×{lm.n_layers}")
    console.print(f"d_model {lm.d_model}   d_mlp {lm.d_mlp}   neurons {lm.n_layers * lm.d_mlp:,}   device {lm.device}")
    t = Table("layer", "median resid norm", "median |W_down col|", "raw act for rel dose 1.0")
    for l in range(lm.n_layers):
        cn = float(lm.col_norms(l).median())
        t.add_row(str(l), f"{lm.resid_norm[l]:.2f}", f"{cn:.4f}", f"{lm.resid_norm[l] / cn:.1f}")
    console.print(t)


@app.command()
def gen(prompt: Annotated[list[str], typer.Argument(help="user message(s); each is generated separately")],
        model: ModelOpt = "Qwen/Qwen3-0.6B", device: DeviceOpt = None, steer: SteerOpt = [], recipe: RecipeOpt = None,
        max_new: int = 200, temperature: float = 0.0, thinking: bool = False,
        compare: Annotated[bool, typer.Option(help="also show ▼ (opposite pole) and unsteered outputs")] = False):
    """One-shot steered generation (greedy by default)."""
    from . import generate as G

    lm, steerer = _setup(model, device, steer, recipe)
    console.print("[dim]" + (", ".join(iv.describe() for iv in steerer.interventions) or "unsteered") + "[/dim]")
    for p in prompt:
        ids = lm.chat_ids([{"role": "user", "content": p}], thinking=thinking)
        if compare and steerer.interventions:
            outs = G.generate(lm, steerer, [ids] * 3, max_new, temperature, row_scale=[-1, 0, 1])
            labels = ["▼", "0", "▲"]
        else:
            outs, labels = G.generate(lm, steerer, [ids], max_new, temperature), ["▲" if steerer.interventions else "0"]
        console.rule(p)
        for lab, o in zip(labels, outs):
            console.print(f"[bold]{lab}[/bold] ", end="")
            console.print(o + "\n", markup=False)


@app.command()
def index(model: ModelOpt = "Qwen/Qwen3-0.6B", device: DeviceOpt = None,
          tokens: Annotated[int, typer.Option(help="corpus tokens to scan")] = 2_000_000,
          seq_len: int = 512, batch: int = 16,
          parquet: Annotated[Optional[str], typer.Option(help="FineWeb-style parquet with a 'text' column")] = None):
    """Build the neuron index: corpus stats, top contexts for both poles, logit lens (one .npz per model)."""
    from .index import FINEWEB, build
    from .model import load

    with console.status(f"loading {model} ..."):
        lm = load(model, device=device)
    build(lm, n_tokens=tokens, seq_len=seq_len, batch=batch, parquet=parquet or FINEWEB,
          log=lambda m: console.print(m, markup=False))


@app.command()
def inspect(neuron: Annotated[list[str], typer.Argument(help="neuron ids, e.g. L14N2742+")],
            model: ModelOpt = "Qwen/Qwen3-0.6B", device: DeviceOpt = None, contexts: int = 8,
            steer: Annotated[str, typer.Option(help="comma-separated rel doses to sample, '' to skip")] = "0.5,1.0"):
    """Show what a neuron fires on (corpus contexts), what it writes (logit lens), and steered samples."""
    from .steer import parse_id
    from .views import show_neuron

    lm, steerer = _setup(model, device, [], None)
    doses = tuple(float(x) for x in steer.split(",") if x)
    for nid in neuron:
        l, n, pole = parse_id(nid)
        show_neuron(console, lm, steerer, l, n, pole, n_ctx=contexts, steer_doses=doses)


@app.command()
def find(prop: Annotated[str, typer.Argument(metavar="PROPERTY", help="e.g. 'sarcastic' or 'wants to talk about the ocean'")],
         model: ModelOpt = "Qwen/Qwen3-0.6B", device: DeviceOpt = None,
         out: Annotated[Optional[str], typer.Option("--out", "-o", help="recipe path (report goes next to it)")] = None,
         contrast: Annotated[Optional[str], typer.Option(help="contrast-set JSON instead of asking Claude")] = None,
         judge: Annotated[str, typer.Option(help="claude, or local (the loaded model; offline testing only)")] = "claude",
         singles: int = 16, controls: int = 6, prompts: int = 5, max_new: int = 80, batch: int = 48, seed: int = 0,
         ks: Annotated[str, typer.Option(help="composite sizes to try, comma-separated")] = "4,8,16,32"):
    """Find neurons / composites that steer the model toward PROPERTY and save a verified recipe."""
    import json
    import re
    from pathlib import Path

    from .find import FindConfig, find as run_find, save_report
    from .views import show_find_report

    if judge == "claude" or contrast is None:
        from . import llm
        if not llm.available():
            console.print("[red]no Claude API credentials: set ANTHROPIC_API_KEY or run `ant auth login` "
                          "(or pass --contrast FILE --judge local for an offline run)[/red]")
            raise typer.Exit(1)
    lm, steerer = _setup(model, device, [], None)
    cfg = FindConfig(n_singles=singles, n_controls=controls, n_prompts=prompts, max_new=max_new, batch=batch,
                     judge=judge, seed=seed, composite_ks=tuple(int(k) for k in ks.split(",")))
    cs = json.loads(Path(contrast).read_text()) if contrast else None
    recipe, report = run_find(lm, steerer, prop, cfg, contrast=cs, log=lambda m: console.print(m, markup=False))
    out = out or re.sub(r"[^a-z0-9]+", "_", prop.lower()).strip("_")[:40] + ".json"
    rep_path = Path(out).with_suffix(".report.json")
    save_report(report, rep_path)
    show_find_report(console, report)
    if recipe:
        recipe.save(out)
        console.print(f"saved recipe to {out}  (try: nf chat -m {model} -r {out})")
    console.print(f"full report: {rep_path}")


@app.command()
def chat(model: ModelOpt = "Qwen/Qwen3-0.6B", device: DeviceOpt = None, steer: SteerOpt = [], recipe: RecipeOpt = None,
         max_new: int = 300, temperature: float = 0.0, thinking: bool = False,
         system: Annotated[Optional[str], typer.Option(help="system prompt")] = None):
    """Interactive chat with live steering (/help inside)."""
    from .chat import run_chat

    lm, steerer = _setup(model, device, steer, recipe)
    run_chat(lm, steerer, max_new=max_new, temperature=temperature, thinking=thinking, system=system)


if __name__ == "__main__":
    app()
