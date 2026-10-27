"""Generation, activation readout, and next-token logprobs, all aware of the active steering."""
import threading
from collections.abc import Iterator

import torch
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

from .model import LoadedModel
from .steer import Steerer


def _gen_kwargs(lm: LoadedModel, max_new: int, temperature: float) -> dict:
    kw = dict(max_new_tokens=max_new, eos_token_id=lm.eos_ids, pad_token_id=lm.tok.pad_token_id)
    if temperature > 0:
        kw.update(do_sample=True, temperature=temperature, top_p=0.95)
    else:
        kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    return kw


def _batch(lm: LoadedModel, seqs: list[list[int]]) -> dict:
    return lm.tok.pad({"input_ids": seqs}, padding=True, return_tensors="pt").to(lm.device)


@torch.no_grad()
def generate(lm: LoadedModel, steerer: Steerer, seqs: list[list[int]], max_new: int = 256,
             temperature: float = 0.0, row_scale: list[float] | None = None) -> list[str]:
    """Generate a continuation for each prompt (token ids). `row_scale` gives each row its own steering scale."""
    batch = _batch(lm, seqs)
    with steerer.scaled(row_scale):
        out = lm.model.generate(**batch, **_gen_kwargs(lm, max_new, temperature))
    n = batch["input_ids"].shape[1]
    return [lm.tok.decode(o[n:], skip_special_tokens=True) for o in out]


class _StopOnEvent(StoppingCriteria):
    def __init__(self, event: threading.Event):
        self.event = event

    def __call__(self, input_ids, scores, **kw):
        return torch.full((input_ids.shape[0],), self.event.is_set(), dtype=torch.bool, device=input_ids.device)


def stream(lm: LoadedModel, steerer: Steerer, ids: list[int], max_new: int = 256, temperature: float = 0.0,
           stop: threading.Event | None = None) -> Iterator[str]:
    """Yield decoded text chunks as they are generated (single prompt, current steering). Setting `stop`
    ends generation after the current token."""
    streamer = TextIteratorStreamer(lm.tok, skip_prompt=True, skip_special_tokens=True)
    batch = _batch(lm, [ids])
    criteria = StoppingCriteriaList([_StopOnEvent(stop)]) if stop is not None else None
    err = []

    def run():
        try:
            with torch.no_grad():
                lm.model.generate(**batch, **_gen_kwargs(lm, max_new, temperature), streamer=streamer,
                                  stopping_criteria=criteria)
        except Exception as e:  # surface errors from the worker thread instead of hanging the streamer
            err.append(e)
            streamer.end()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    yield from streamer
    t.join()
    if err:
        raise err[0]


@torch.no_grad()
def activations(lm: LoadedModel, ids: list[int], neurons: list[tuple[int, int]]) -> torch.Tensor:
    """Activations of the given (layer, neuron) pairs at every position, under whatever steering is active
    (wrap in `steerer.disabled()` for natural activations). Returns float32 [T, len(neurons)] on cpu."""
    layers = sorted({l for l, _ in neurons})
    got = {}

    def mk(l):
        def hook(mod, args):
            got[l] = args[0][0].float().cpu()
        return hook

    hs = [lm.down_proj(l).register_forward_pre_hook(mk(l)) for l in layers]
    try:
        lm.model(input_ids=torch.tensor([ids], device=lm.device))
    finally:
        for h in hs:
            h.remove()
    return torch.stack([got[l][:, n] for l, n in neurons], dim=1)


@torch.no_grad()
def next_token_logprobs(lm: LoadedModel, steerer: Steerer, ids: list[int], row_scale: list[float],
                        k: int = 10) -> list[list[tuple[str, float]]]:
    """Top-k next tokens after `ids`, once per entry of `row_scale` (e.g. [0, 1] = baseline vs steered)."""
    batch = _batch(lm, [ids] * len(row_scale))
    with steerer.scaled(row_scale):
        logits = lm.model(**batch).logits[:, -1].float()
    lp = torch.log_softmax(logits, dim=-1)
    top = lp.topk(k, dim=-1)
    return [[(lm.tok.decode([int(i)]), float(v)) for v, i in zip(vals, idx)]
            for vals, idx in zip(top.values, top.indices)]
