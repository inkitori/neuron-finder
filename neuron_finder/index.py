"""Per-model neuron index: activation statistics, top contexts for both poles, and the logit lens.

One pass over a slice of FineWeb-Edu. Everything is written to a single .npz (the ark fileset is at its file-count
quota, so no per-neuron files). Position 0 of every row is masked: it carries attention-sink-like values that
would otherwise top every neuron's list.
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .model import LoadedModel, _slug, cache_dir

FINEWEB = "/mmfs1/data/fineweb_edu/sample/10BT/000_00000.parquet"


def index_path(model_id: str) -> Path:
    return cache_dir() / _slug(model_id) / "index.npz"


def corpus_rows(tok, parquet: str, n_tokens: int, seq_len: int, max_chunks_per_doc: int = 2) -> np.ndarray:
    """Tokenized FineWeb-Edu rows, [n_rows, seq_len] int32, padded with -1. Rows never cross documents."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet)
    rows, total = [], 0
    for batch in pf.iter_batches(batch_size=256, columns=["text"]):
        for text in batch.column("text").to_pylist():
            ids = tok(text, add_special_tokens=False)["input_ids"]
            for c in range(min(max_chunks_per_doc, (len(ids) + seq_len - 1) // seq_len)):
                chunk = ids[c * seq_len:(c + 1) * seq_len]
                if len(chunk) < 32:
                    break
                rows.append(chunk + [-1] * (seq_len - len(chunk)))
                total += len(chunk)
            if total >= n_tokens:
                return np.array(rows, dtype=np.int32)
    return np.array(rows, dtype=np.int32)


def final_norm_gain(lm: LoadedModel) -> torch.Tensor:
    """Effective elementwise gain of the final RMSNorm (works for both `w*x` and `(1+w)*x` variants)."""
    parent = lm.model.get_submodule(lm.layers_path.rsplit(".", 1)[0]) if "." in lm.layers_path else lm.model
    norm = getattr(parent, "norm", None)
    if norm is None:
        return torch.ones(lm.d_model, device=lm.device)
    p = next(norm.parameters())
    with torch.no_grad():
        return norm(torch.ones(1, lm.d_model, device=p.device, dtype=p.dtype))[0].float()


@torch.no_grad()
def logit_lens(lm: LoadedModel, k: int = 10, chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Top-k tokens each neuron's +/− write direction promotes, via the unembedding. [L, d_mlp, k] int32 each."""
    W_U = lm.model.get_output_embeddings().weight  # [vocab, d_model]
    g = final_norm_gain(lm).to(W_U.dtype)
    pos = np.zeros((lm.n_layers, lm.d_mlp, k), np.int32)
    neg = np.zeros_like(pos)
    for l in range(lm.n_layers):
        W = lm.down_proj(l).weight.to(W_U.device)  # [d_model, d_mlp]
        for s in range(0, lm.d_mlp, chunk):
            logits = (W_U @ (W[:, s:s + chunk] * g[:, None])).float()  # [vocab, c]
            pos[l, s:s + chunk] = logits.topk(k, dim=0).indices.T.cpu().numpy()
            neg[l, s:s + chunk] = (-logits).topk(k, dim=0).indices.T.cpu().numpy()
    return pos, neg


@torch.no_grad()
def build(lm: LoadedModel, n_tokens: int = 2_000_000, seq_len: int = 512, batch: int = 16, k: int = 16,
          parquet: str = FINEWEB, log=print) -> Path:
    t0 = time.time()
    rows = corpus_rows(lm.tok, parquet, n_tokens, seq_len)
    log(f"corpus: {len(rows)} rows, {(rows >= 0).sum():,} tokens ({time.time() - t0:.0f}s)")
    L, D, dev = lm.n_layers, lm.d_mlp, lm.device
    acc = {
        "sum": torch.zeros(L, D, device=dev, dtype=torch.float64),
        "sumsq": torch.zeros(L, D, device=dev, dtype=torch.float64),
        "max": torch.full((L, D), -torch.inf, device=dev),
        "min": torch.full((L, D), torch.inf, device=dev),
        "npos": torch.zeros(L, D, device=dev, dtype=torch.float64),
        "tpv": torch.full((L, k, D), -torch.inf, device=dev),  # top positive values
        "tpi": torch.zeros(L, k, D, device=dev, dtype=torch.int64),  # flat token index into rows
        "tnv": torch.full((L, k, D), -torch.inf, device=dev),  # top negative values (stored negated)
        "tni": torch.zeros(L, k, D, device=dev, dtype=torch.int64),
    }
    count = 0
    state = {}

    def merge(vals, idx, layer, new_v, new_i):
        cat_v = torch.cat([vals[layer], new_v])
        cat_i = torch.cat([idx[layer], new_i])
        top = cat_v.topk(k, dim=0)
        vals[layer] = top.values
        idx[layer] = cat_i.gather(0, top.indices)

    def mk(l):
        def hook(mod, args):
            a = args[0][state["mask"]].float()  # [M, D] valid positions only
            acc["sum"][l] += a.sum(0, dtype=torch.float64)
            acc["sumsq"][l] += (a.double() ** 2).sum(0)
            acc["max"][l] = torch.maximum(acc["max"][l], a.max(0).values)
            acc["min"][l] = torch.minimum(acc["min"][l], a.min(0).values)
            acc["npos"][l] += (a > 0).sum(0)
            kk = min(k, a.shape[0])
            flat = state["flat"]
            v, i = a.topk(kk, dim=0)
            merge(acc["tpv"], acc["tpi"], l, v, flat[i])
            v, i = (-a).topk(kk, dim=0)
            merge(acc["tnv"], acc["tni"], l, v, flat[i])
        return hook

    hs = [lm.down_proj(l).register_forward_pre_hook(mk(l)) for l in range(L)]
    try:
        for b in range(0, len(rows), batch):
            r = torch.from_numpy(rows[b:b + batch]).to(dev)
            attn = (r >= 0).long()
            mask = attn.bool().clone()
            mask[:, 0] = False
            state["mask"] = mask
            pos = torch.arange(b * seq_len, (b + len(r)) * seq_len, device=dev).view(len(r), seq_len)
            state["flat"] = pos[mask]
            count += int(mask.sum())
            lm.model(input_ids=r.clamp(min=0), attention_mask=attn)
            if (b // batch) % 50 == 0:
                log(f"  {b + len(r)}/{len(rows)} rows ({time.time() - t0:.0f}s)")
    finally:
        for h in hs:
            h.remove()

    mean = acc["sum"] / count
    std = (acc["sumsq"] / count - mean ** 2).clamp(min=0).sqrt()
    log(f"logit lens ... ({time.time() - t0:.0f}s)")
    lens_pos, lens_neg = logit_lens(lm)
    out = index_path(lm.model_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        meta=np.array(json.dumps({"model": lm.model_id, "n_tokens": count, "seq_len": seq_len, "k": k,
                                  "parquet": parquet})),
        rows=rows,
        mean=mean.float().cpu().numpy(), std=std.float().cpu().numpy(),
        max=acc["max"].cpu().numpy(), min=acc["min"].cpu().numpy(),
        frac_pos=(acc["npos"] / count).float().cpu().numpy(),
        top_pos_val=acc["tpv"].permute(0, 2, 1).cpu().numpy(),
        top_pos_idx=acc["tpi"].permute(0, 2, 1).cpu().numpy().astype(np.int32),
        top_neg_val=(-acc["tnv"]).permute(0, 2, 1).cpu().numpy(),
        top_neg_idx=acc["tni"].permute(0, 2, 1).cpu().numpy().astype(np.int32),
        lens_pos=lens_pos, lens_neg=lens_neg,
    )
    tmp.replace(out)
    log(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f}s)")
    return out


@dataclass
class Context:
    value: float
    left: str
    token: str
    right: str


class Index:
    def __init__(self, path: Path, tok):
        z = np.load(path)
        self.meta = json.loads(str(z["meta"]))
        self.tok = tok
        for name in ("rows", "mean", "std", "max", "min", "frac_pos", "top_pos_val", "top_pos_idx",
                     "top_neg_val", "top_neg_idx", "lens_pos", "lens_neg"):
            setattr(self, name, z[name])
        self.seq_len = self.rows.shape[1]

    @classmethod
    def load(cls, lm: LoadedModel) -> "Index | None":
        p = index_path(lm.model_id)
        return cls(p, lm.tok) if p.exists() else None

    def abs_max(self, layer: int, neuron: int, pole: str) -> float:
        """Largest corpus activation on this pole (positive number)."""
        return float(self.max[layer, neuron]) if pole == "+" else float(-self.min[layer, neuron])

    def contexts(self, layer: int, neuron: int, pole: str, n: int = 8, before: int = 24, after: int = 6) -> list[Context]:
        vals = self.top_pos_val if pole == "+" else self.top_neg_val
        idxs = self.top_pos_idx if pole == "+" else self.top_neg_idx
        out = []
        for v, flat in zip(vals[layer, neuron][:n], idxs[layer, neuron][:n]):
            if not np.isfinite(v):
                continue
            r, p = divmod(int(flat), self.seq_len)
            row = self.rows[r]
            left = [t for t in row[max(0, p - before):p] if t >= 0]
            right = [t for t in row[p + 1:p + 1 + after] if t >= 0]
            out.append(Context(float(v), self.tok.decode(left), self.tok.decode([int(row[p])]), self.tok.decode(right)))
        return out

    def lens(self, layer: int, neuron: int, pole: str) -> list[str]:
        ids = (self.lens_pos if pole == "+" else self.lens_neg)[layer, neuron]
        return [self.tok.decode([int(i)]) for i in ids]
