"""Interactive steered chat."""
import shlex
import threading

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.table import Table
from rich.text import Text

from . import generate as G
from .model import LoadedModel, cache_dir
from .steer import Intervention, Recipe, Steerer, parse_id

HELP = """\
[bold]Steering[/bold]
  /steer ID [DOSE] [MODE] [UNIT]   e.g. /steer L14N2742+ 0.8   (default add, 0.8 rel; '-' pole pushes negative)
  /ablate ID                       zero a neuron (mul 0)
  /clamp ID VALUE [UNIT]           pin a neuron to VALUE (default raw activation units)
  /unsteer [ID|all]                remove one or all interventions
  /list                            show active interventions
  /scale K                         multiply all add-doses by K
[bold]Probing[/bold]
  /compare [MESSAGE]               ▼ / 0 / ▲ side by side for MESSAGE (default: your last message)
  /fires [@ID,ID,...] TEXT         per-token activations of the steered (or listed) neurons on TEXT, unsteered
  /logprobs MESSAGE [|| PREFIX]    top next tokens, unsteered vs steered, after MESSAGE (+ assistant PREFIX)
[bold]Session[/bold]
  /think on|off   /temp T   /max N   /system TEXT|off   /reset
  /save PATH      /load PATH       (recipe JSON)
  /find PROPERTY                   (not yet: milestone M3)
  /help   /quit
Ctrl-C stops a generation in progress."""


class ChatSession:
    def __init__(self, lm: LoadedModel, steerer: Steerer, *, max_new: int = 300, temperature: float = 0.0,
                 thinking: bool = False, system: str | None = None):
        self.lm, self.steerer = lm, steerer
        self.max_new, self.temperature, self.thinking, self.system = max_new, temperature, thinking, system
        self.messages: list[dict] = []
        self.property = ""
        self.console = Console(highlight=False)

    # -- helpers -----------------------------------------------------------------------------------------------
    def _context(self, extra: list[dict] = ()) -> list[dict]:
        sys = [{"role": "system", "content": self.system}] if self.system else []
        return sys + self.messages + list(extra)

    def _ids(self, extra: list[dict] = ()) -> list[int]:
        return self.lm.chat_ids(self._context(extra), thinking=self.thinking)

    def _status(self) -> str:
        ivs = self.steerer.interventions
        steer = ", ".join(iv.describe() for iv in ivs) if ivs else "unsteered"
        return f"[dim]\\[{steer}][/dim]"

    # -- main loop ---------------------------------------------------------------------------------------------
    def run(self):
        c = self.console
        lm = self.lm
        c.print(f"[bold]{lm.model_id}[/bold]  {lm.n_layers} layers × {lm.d_mlp} neurons   /help for commands")
        c.print(self._status())
        session = PromptSession(history=FileHistory(str(cache_dir() / "chat_history")))
        while True:
            try:
                line = session.prompt("you> ").strip()
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
            if not line:
                continue
            if line.startswith("/"):
                try:
                    if self.command(line) == "quit":
                        break
                except (ValueError, IndexError, FileNotFoundError) as e:
                    c.print(f"[red]{e}[/red]")
                continue
            self.reply(line)

    def reply(self, text: str):
        self.messages.append({"role": "user", "content": text})
        ids = self._ids()
        stop = threading.Event()
        chunks = []
        self.console.print("[bold cyan]model>[/bold cyan] ", end="")
        try:
            for chunk in G.stream(self.lm, self.steerer, ids, self.max_new, self.temperature, stop=stop):
                chunks.append(chunk)
                self.console.print(chunk, end="", markup=False, soft_wrap=True)
        except KeyboardInterrupt:
            stop.set()
            self.console.print(" [dim]\\[stopped][/dim]", end="")
        self.console.print()
        self.messages.append({"role": "assistant", "content": "".join(chunks)})

    # -- commands ----------------------------------------------------------------------------------------------
    def command(self, line: str):
        c = self.console
        cmd, _, rest = line.partition(" ")
        rest = rest.strip()
        args = shlex.split(rest) if rest else []
        st = self.steerer

        if cmd in ("/quit", "/exit", "/q"):
            return "quit"
        elif cmd == "/help":
            c.print(HELP)
        elif cmd == "/steer":
            if not args:
                raise ValueError("usage: /steer ID [DOSE] [MODE] [UNIT]")
            st.add(Intervention.parse(":".join(args)))
            c.print(self._status())
        elif cmd == "/ablate":
            for a in args:
                l, n, _ = parse_id(a)
                st.add(Intervention(l, n, mode="mul", dose=0.0))
            c.print(self._status())
        elif cmd == "/clamp":
            if len(args) < 2:
                raise ValueError("usage: /clamp ID VALUE [UNIT]")
            l, n, pole = parse_id(args[0])
            st.add(Intervention(l, n, pole, mode="clamp", dose=float(args[1]), unit=args[2] if len(args) > 2 else "raw"))
            c.print(self._status())
        elif cmd == "/unsteer":
            if not args or args[0] == "all":
                st.clear()
            else:
                for a in args:
                    l, n, _ = parse_id(a)
                    if not st.remove(l, n):
                        c.print(f"[yellow]{a} was not steered[/yellow]")
            c.print(self._status())
        elif cmd == "/list":
            self.show_interventions()
        elif cmd == "/scale":
            st.scale_doses(float(args[0]))
            c.print(self._status())
        elif cmd == "/compare":
            self.compare(rest)
        elif cmd == "/fires":
            self.fires(rest)
        elif cmd == "/logprobs":
            self.logprobs(rest)
        elif cmd == "/think":
            self.thinking = (args[0].lower() in ("on", "1", "true")) if args else not self.thinking
            c.print(f"[dim]thinking {'on' if self.thinking else 'off'}[/dim]")
        elif cmd == "/temp":
            self.temperature = float(args[0])
        elif cmd == "/max":
            self.max_new = int(args[0])
        elif cmd == "/system":
            self.system = None if rest in ("", "off") else rest
            c.print(f"[dim]system prompt {'cleared' if self.system is None else 'set'}[/dim]")
        elif cmd == "/reset":
            self.messages = []
            c.print("[dim]conversation cleared (steering kept)[/dim]")
        elif cmd == "/save":
            Recipe(model=self.lm.model_id, interventions=list(st.interventions), property=self.property).save(args[0])
            c.print(f"[dim]saved {len(st.interventions)} interventions to {args[0]}[/dim]")
        elif cmd == "/load":
            r = Recipe.load(args[0])
            if r.model != self.lm.model_id:
                c.print(f"[yellow]recipe was made for {r.model}, this is {self.lm.model_id}[/yellow]")
            st.set(r.interventions)
            self.property = r.property
            c.print(self._status())
        elif cmd == "/find":
            c.print("[yellow]/find arrives in milestone M3 (Claude-generated contrast sets + verification).[/yellow]")
        else:
            c.print(f"[red]unknown command {cmd}[/red] (/help)")

    def show_interventions(self):
        ivs = self.steerer.interventions
        if not ivs:
            self.console.print("[dim]no interventions[/dim]")
            return
        t = Table("id", "mode", "dose", "unit", "raw value", "|W_down col|", "resid norm")
        for iv in ivs:
            t.add_row(iv.id, iv.mode, f"{iv.dose:g}", iv.unit if iv.mode != "mul" else "-",
                      f"{self.steerer.value(iv):+.3f}", f"{float(self.lm.col_norms(iv.layer)[iv.neuron]):.3f}",
                      f"{self.lm.resid_norm[iv.layer]:.1f}")
        self.console.print(t)

    def compare(self, text: str):
        if not self.steerer.interventions:
            raise ValueError("nothing is steered; /steer something first")
        if text:
            extra, ctx = [{"role": "user", "content": text}], self.messages
        else:
            last = max((i for i, m in enumerate(self.messages) if m["role"] == "user"), default=None)
            if last is None:
                raise ValueError("usage: /compare MESSAGE (no previous message to re-run)")
            extra, ctx = [self.messages[last]], self.messages[:last]
        saved, self.messages = self.messages, ctx
        try:
            ids = self._ids(extra)
        finally:
            self.messages = saved
        with self.console.status("generating ▼ / 0 / ▲ ..."):
            outs = G.generate(self.lm, self.steerer, [ids] * 3, self.max_new, self.temperature, row_scale=[-1, 0, 1])
        t = Table(show_lines=False, expand=True)
        for name in ("▼ opposite pole", "0 unsteered", "▲ steered"):
            t.add_column(name, ratio=1, overflow="fold")
        t.add_row(*[Text(o) for o in outs])
        self.console.print(t)
        if any(iv.mode != "add" for iv in self.steerer.interventions):
            self.console.print("[dim]mul/clamp interventions are applied unchanged in both ▼ and ▲ columns[/dim]")

    def fires(self, rest: str):
        ids_spec = None
        if rest.startswith("@"):
            ids_spec, _, rest = rest[1:].partition(" ")
        if not rest:
            raise ValueError("usage: /fires [@ID,ID,...] TEXT")
        neurons = [parse_id(x)[:2] for x in ids_spec.split(",")] if ids_spec else \
            [iv.key for iv in self.steerer.interventions]
        if not neurons:
            raise ValueError("no neurons: steer some, or pass @L14N2742,L3N7")
        toks = self.lm.tok(rest)["input_ids"]
        with self.steerer.disabled():
            acts = G.activations(self.lm, toks, neurons)
        pieces = [self.lm.tok.decode([t]) for t in toks]
        for j, (l, n) in enumerate(neurons):
            a = acts[:, j]
            body = a[1:] if len(a) > 1 else a  # position 0 often carries sink-like values
            peak = float(body.abs().max()) or 1.0
            line = Text(f"L{l}N{n}  max {float(body.max()):+.2f}  min {float(body.min()):+.2f}\n", style="bold")
            for p, v in zip(pieces, a.tolist()):
                x = min(abs(v) / peak, 1.0)
                if x < 0.15:
                    line.append(p)
                else:
                    shade = int(40 + 160 * x)
                    line.append(p, style=f"on rgb({shade},0,0)" if v < 0 else f"on rgb(0,{shade},0)")
            self.console.print(line)
        self.console.print("[dim]green = positive pole, red = negative pole, brightness = |act| / max|act|[/dim]")

    def logprobs(self, rest: str):
        msg, _, prefix = rest.partition("||")
        msg, prefix = msg.strip(), prefix.strip()
        if not msg:
            raise ValueError("usage: /logprobs MESSAGE [|| ASSISTANT PREFIX]")
        ids = self._ids([{"role": "user", "content": msg}])
        if prefix:
            ids = ids + self.lm.tok(prefix, add_special_tokens=False)["input_ids"]
        base, steered = G.next_token_logprobs(self.lm, self.steerer, ids, row_scale=[0, 1])
        t = Table("unsteered", "logprob", "steered", "logprob")
        for (a, la), (b, lb) in zip(base, steered):
            t.add_row(repr(a), f"{la:.2f}", repr(b), f"{lb:.2f}")
        self.console.print(t)


def run_chat(lm: LoadedModel, steerer: Steerer, **kw):
    ChatSession(lm, steerer, **kw).run()
