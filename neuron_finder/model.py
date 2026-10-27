"""Load a Qwen model and expose its MLP neurons.

A neuron is one channel of the SwiGLU activation `silu(gate_proj(x)) * up_proj(x)`, i.e. one input channel of
`mlp.down_proj`. Its write direction into the residual stream is column `down_proj.weight[:, n]`.
"""
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Varied snippets for calibrating the residual-stream norm per layer (the unit of "rel" doses).
CALIB_TEXTS = [
    "The mitochondria is the powerhouse of the cell, converting nutrients into ATP through respiration.",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
    "She opened the letter slowly, afraid of what it might say, and read it twice before sitting down.",
    "Preheat the oven to 180°C. Whisk the eggs with sugar until pale, then fold in the flour and butter.",
    "The central bank raised interest rates by a quarter point on Wednesday, citing persistent inflation.",
    "In 1492, Columbus crossed the Atlantic with three ships, reaching the Bahamas in October.",
    "Let x be a real number such that x^2 - 5x + 6 = 0. Then x = 2 or x = 3.",
    "lol did you see the game last night?? that last-second shot was insane",
    "Photosynthesis takes place in the chloroplasts, where light energy is captured by chlorophyll.",
    "The hikers reached the summit at dawn, the valley below still wrapped in fog and silence.",
    "SELECT name, COUNT(*) FROM orders GROUP BY name HAVING COUNT(*) > 10 ORDER BY 2 DESC;",
    "我们明天早上九点在火车站见面，然后一起去博物馆参观。",
]
CALIB_CHATS = [
    "Who are you?",
    "Tell me about your weekend.",
    "Write a short poem about the ocean.",
    "How do I make a good cup of coffee?",
]


def cache_dir() -> Path:
    root = os.environ.get("NF_CACHE_DIR") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "neuron-finder")
    p = Path(root)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return p


def _slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "--", model_id.strip("/"))


def _find_layers(model: nn.Module) -> tuple[str, nn.ModuleList]:
    """The longest ModuleList whose elements have `mlp.down_proj` (the text decoder stack)."""
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) and hasattr(getattr(mod[0], "mlp", None), "down_proj"):
            if best is None or len(mod) > len(best[1]):
                best = (name, mod)
    if best is None:
        raise ValueError("no decoder layers with a dense mlp.down_proj found (MoE models are not supported yet)")
    return best


def _load_hf(model_id: str, dtype: torch.dtype, device: str) -> nn.Module:
    kw = dict(dtype=dtype, device_map=device)
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **kw)
    except ValueError:
        # multimodal checkpoints (e.g. Qwen3.5) only register an image-text-to-text head
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(model_id, **kw)


@dataclass
class LoadedModel:
    model_id: str
    tok: object
    model: nn.Module
    layers: nn.ModuleList
    layers_path: str
    resid_norm: list[float] = field(default_factory=list)  # median residual norm leaving each layer
    _col_norms: dict = field(default_factory=dict, repr=False)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def d_mlp(self) -> int:
        return self.layers[0].mlp.down_proj.in_features

    @property
    def d_model(self) -> int:
        return self.layers[0].mlp.down_proj.out_features

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def eos_ids(self) -> list[int]:
        ids = {self.tok.eos_token_id}
        im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end, int) and im_end != self.tok.unk_token_id:
            ids.add(im_end)
        return sorted(i for i in ids if i is not None)

    def down_proj(self, layer: int) -> nn.Linear:
        return self.layers[layer].mlp.down_proj

    def w_down_col(self, layer: int, neuron: int) -> torch.Tensor:
        return self.down_proj(layer).weight[:, neuron]

    def col_norms(self, layer: int) -> torch.Tensor:
        """L2 norm of every neuron's write direction in `layer`, shape [d_mlp] (float32, cpu)."""
        if layer not in self._col_norms:
            w = self.down_proj(layer).weight
            self._col_norms[layer] = w.float().norm(dim=0).cpu()
        return self._col_norms[layer]

    def chat_ids(self, messages: list[dict], thinking: bool = False, add_generation_prompt: bool = True) -> list[int]:
        out = self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt,
                                           enable_thinking=thinking)
        return list(out["input_ids"] if hasattr(out, "keys") else out)

    @torch.no_grad()
    def calibrate(self, force: bool = False) -> list[float]:
        """Median residual-stream norm leaving each layer, cached on disk per model."""
        path = cache_dir() / _slug(self.model_id) / "resid_norm.json"
        if path.exists() and not force:
            self.resid_norm = json.loads(path.read_text())["median"]
            return self.resid_norm
        per_layer = [[] for _ in range(self.n_layers)]

        def mk(l):
            def hook(mod, args, out):
                h = out[0] if isinstance(out, tuple) else out
                per_layer[l].append(h[0, 1:].float().norm(dim=-1).cpu())  # skip position 0 (attention sink)
            return hook

        hs = [layer.register_forward_hook(mk(l)) for l, layer in enumerate(self.layers)]
        try:
            seqs = [self.tok(t)["input_ids"] for t in CALIB_TEXTS]
            seqs += [self.chat_ids([{"role": "user", "content": c}]) for c in CALIB_CHATS]
            for ids in seqs:
                self.model(input_ids=torch.tensor([ids], device=self.device))
        finally:
            for h in hs:
                h.remove()
        self.resid_norm = [float(torch.cat(v).median()) for v in per_layer]
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"model": self.model_id, "median": self.resid_norm}))
        except OSError as e:  # a full filesystem shouldn't stop a chat session
            print(f"warning: could not cache calibration at {path}: {e}")
        return self.resid_norm


def load(model_id: str, device: str | None = None, dtype: str = "bfloat16") -> LoadedModel:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = _load_hf(model_id, getattr(torch, dtype), device).eval()
    path, layers = _find_layers(model)
    lm = LoadedModel(model_id=model_id, tok=tok, model=model, layers=layers, layers_path=path)
    lm.calibrate()
    return lm
