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
def chat(model: ModelOpt = "Qwen/Qwen3-0.6B", device: DeviceOpt = None, steer: SteerOpt = [], recipe: RecipeOpt = None,
         max_new: int = 300, temperature: float = 0.0, thinking: bool = False,
         system: Annotated[Optional[str], typer.Option(help="system prompt")] = None):
    """Interactive chat with live steering (/help inside)."""
    from .chat import run_chat

    lm, steerer = _setup(model, device, steer, recipe)
    run_chat(lm, steerer, max_new=max_new, temperature=temperature, thinking=thinking, system=system)


if __name__ == "__main__":
    app()
