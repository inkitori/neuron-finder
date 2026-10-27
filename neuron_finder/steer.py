"""Neuron interventions, the hooks that apply them, and recipe files.

Neuron IDs look like `L14N2742+`: layer 14, neuron 2742, positive pole. Pushing the `-` pole adds a negative
value. Modes:
  add    activation += sign * dose * unit                 (steering; same as adding to the MLP output along W_down[:, n])
  mul    activation *= dose                               (dose 0 = ablate, 2 = double)
  clamp  activation  = sign * dose * unit                 (pin to a fixed value)
Units for add/clamp:
  rel    dose is a fraction of the median residual norm leaving the layer, divided by |W_down[:, n]|, so every
         neuron gets the same push size in the residual stream. Coherent window is roughly 0.5-0.8.
  raw    dose is in raw activation units.
  absmax dose is a multiple of the pole's largest activation over the corpus (needs `nf index`).
"""
import contextlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .model import LoadedModel

ID_RE = re.compile(r"^L(\d+)[\s_]*N(\d+)([+-]?)$", re.IGNORECASE)
MODES = ("add", "mul", "clamp")
UNITS = ("rel", "raw", "absmax")


def parse_id(s: str) -> tuple[int, int, str]:
    m = ID_RE.match(s.strip())
    if not m:
        raise ValueError(f"bad neuron id {s!r}; expected e.g. L14N2742+ or L14N2742-")
    return int(m[1]), int(m[2]), m[3] or "+"


@dataclass
class Intervention:
    layer: int
    neuron: int
    pole: str = "+"
    mode: str = "add"
    dose: float = 0.8
    unit: str = "rel"

    def __post_init__(self):
        if self.pole not in "+-" or len(self.pole) != 1:
            raise ValueError(f"pole must be + or -, got {self.pole!r}")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.unit not in UNITS:
            raise ValueError(f"unit must be one of {UNITS}, got {self.unit!r}")

    @property
    def key(self) -> tuple[int, int]:
        return self.layer, self.neuron

    @property
    def id(self) -> str:
        return f"L{self.layer}N{self.neuron}{self.pole}"

    def describe(self) -> str:
        if self.mode == "mul":
            return f"{self.id[:-1]} ×{self.dose:g}" + (" (ablated)" if self.dose == 0 else "")
        verb = "+=" if self.mode == "add" else "="
        return f"{self.id} {verb} {self.dose:g} {self.unit}"

    @classmethod
    def parse(cls, spec: str) -> "Intervention":
        """`ID[:DOSE[:MODE|UNIT...]]`, e.g. `L14N2742+:0.8`, `L14N2742:0:mul`, `L3N7-:4:clamp:raw`."""
        parts = spec.strip().split(":")
        layer, neuron, pole = parse_id(parts[0])
        iv = dict(layer=layer, neuron=neuron, pole=pole)
        if len(parts) > 1 and parts[1]:
            iv["dose"] = float(parts[1])
        for p in parts[2:]:
            p = p.lower()
            if p in MODES:
                iv["mode"] = p
            elif p in UNITS:
                iv["unit"] = p
            else:
                raise ValueError(f"unknown option {p!r} in {spec!r} (modes: {MODES}, units: {UNITS})")
        return cls(**iv)


class Steerer:
    """Holds the active interventions and keeps one pre-hook per affected layer on `mlp.down_proj`.

    `row_scale` (set via `scaled`) multiplies interventions per batch row, so variants run in one batch:
    add doses are multiplied by the row's scale (−1 flips to the opposite pole); mul/clamp apply wherever the
    scale is non-zero and are skipped on rows with scale 0 (the unsteered baseline).
    """

    def __init__(self, lm: LoadedModel):
        self.lm = lm
        self.interventions: list[Intervention] = []
        self._hooks = []
        self._row_scale: torch.Tensor | None = None

    # -- editing -----------------------------------------------------------------------------------------------
    def add(self, iv: Intervention):
        if not 0 <= iv.layer < self.lm.n_layers:
            raise ValueError(f"layer {iv.layer} out of range (model has {self.lm.n_layers} layers)")
        if not 0 <= iv.neuron < self.lm.d_mlp:
            raise ValueError(f"neuron {iv.neuron} out of range (d_mlp = {self.lm.d_mlp})")
        self.interventions = [x for x in self.interventions if x.key != iv.key] + [iv]
        self._install()

    def remove(self, layer: int, neuron: int) -> bool:
        n = len(self.interventions)
        self.interventions = [x for x in self.interventions if x.key != (layer, neuron)]
        self._install()
        return len(self.interventions) < n

    def set(self, ivs: list[Intervention]):
        self.interventions = []
        for iv in ivs:
            self.add(iv)
        self._install()

    def clear(self):
        self.interventions = []
        self._install()

    def scale_doses(self, k: float):
        for iv in self.interventions:
            if iv.mode == "add":
                iv.dose *= k
        self._install()

    # -- applying ----------------------------------------------------------------------------------------------
    def value(self, iv: Intervention) -> float:
        """The raw number the hook uses: the added/clamped activation, or the multiplier."""
        if iv.mode == "mul":
            return iv.dose
        v = iv.dose * (1.0 if iv.pole == "+" else -1.0)
        if iv.unit == "rel":
            v *= self.lm.resid_norm[iv.layer] / float(self.lm.col_norms(iv.layer)[iv.neuron])
        elif iv.unit == "absmax":
            if self.lm.index is None:
                raise ValueError("unit 'absmax' needs the neuron index; run `nf index` first")
            v *= self.lm.index.abs_max(iv.layer, iv.neuron, iv.pole)
        return v

    @contextlib.contextmanager
    def scaled(self, row_scale: list[float] | None):
        prev = self._row_scale
        self._row_scale = None if row_scale is None else torch.tensor(row_scale, device=self.lm.device)
        try:
            yield
        finally:
            self._row_scale = prev

    @contextlib.contextmanager
    def disabled(self):
        saved = self.interventions
        self.interventions = []
        self._install()
        try:
            yield
        finally:
            self.interventions = saved
            self._install()

    @contextlib.contextmanager
    def per_row(self, rows: list[list[Intervention]]):
        """Give every batch row its own set of add-mode interventions (for verification sweeps). The persistent
        interventions are suspended meanwhile."""
        if any(iv.mode != "add" for ivs in rows for iv in ivs):
            raise ValueError("per_row supports add-mode interventions only")
        dev, B = self.lm.device, len(rows)
        deltas: dict[int, torch.Tensor] = {}
        for b, ivs in enumerate(rows):
            for iv in ivs:
                if iv.layer not in deltas:
                    deltas[iv.layer] = torch.zeros(B, self.lm.d_mlp, device=dev)
                deltas[iv.layer][b, iv.neuron] += self.value(iv)

        def mk(delta):
            def hook(mod, args):
                x = args[0]
                return ((x.float() + delta[:, None, :]).to(x.dtype),) + tuple(args[1:])
            return hook

        with self.disabled():
            hs = [self.lm.down_proj(l).register_forward_pre_hook(mk(d)) for l, d in deltas.items()]
            try:
                yield
            finally:
                for h in hs:
                    h.remove()

    def _install(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        by_layer: dict[int, list[Intervention]] = {}
        for iv in self.interventions:
            by_layer.setdefault(iv.layer, []).append(iv)
        for layer, ivs in by_layer.items():
            self._hooks.append(self.lm.down_proj(layer).register_forward_pre_hook(self._make_hook(ivs)))

    def _make_hook(self, ivs: list[Intervention]):
        dev = self.lm.device
        groups = {}
        for mode in MODES:
            sel = [iv for iv in ivs if iv.mode == mode]
            if sel:
                groups[mode] = (torch.tensor([iv.neuron for iv in sel], device=dev),
                                torch.tensor([self.value(iv) for iv in sel], device=dev, dtype=torch.float32))

        def hook(mod, args):
            x = args[0]
            s = self._row_scale
            if s is not None:
                s = s.view(-1, *([1] * (x.dim() - 1))).float()  # [B, 1, 1]
            x = x.clone()
            if "add" in groups:
                idx, val = groups["add"]
                delta = val if s is None else val * s
                x[..., idx] = (x[..., idx].float() + delta).to(x.dtype)
            if "mul" in groups:
                idx, val = groups["mul"]
                f = val if s is None else torch.where(s != 0, val, torch.ones_like(val))
                x[..., idx] = (x[..., idx].float() * f).to(x.dtype)
            if "clamp" in groups:
                idx, val = groups["clamp"]
                cur = x[..., idx].float()
                new = val.expand_as(cur) if s is None else torch.where(s != 0, val, cur)
                x[..., idx] = new.to(x.dtype)
            return (x,) + tuple(args[1:])

        return hook


# -- recipes ---------------------------------------------------------------------------------------------------
@dataclass
class Recipe:
    model: str
    interventions: list[Intervention]
    property: str = ""
    scores: dict = field(default_factory=dict)
    notes: str = ""

    def save(self, path: str | Path):
        d = asdict(self)
        Path(path).write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "Recipe":
        d = json.loads(Path(path).read_text())
        d["interventions"] = [Intervention(**iv) for iv in d.get("interventions", [])]
        return cls(**d)
