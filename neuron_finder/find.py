"""`nf find`: from a natural-language property to a verified steering recipe.

1. contrast set   target / near-miss / control texts (Claude, or a JSON file)
2. readouts       every neuron's max / min / mean / summary-position activation on each text, shown to the model
                  as its own reply
3. scoring        per neuron pole: AUROC target-vs-near-miss and target-vs-control, chosen on one half of the
                  texts and confirmed on the other; ranked by effect size against the neuron's own spread
4. composites     the T−N difference in each layer's MLP output splits exactly into per-neuron pieces
                  Δa_n · W_down[:, n]; neurons whose piece points along the residual property direction form
                  top-k composites
5. verification   steer singles, composites and random-neuron controls on neutral prompts; a blind judge rates
                  property and coherence
6. compose        greedily stack verified levers at reduced doses
"""
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import generate as G
from .model import LoadedModel
from .steer import Intervention, Recipe, Steerer

NEUTRAL_USER = "Say something."
SUMMARY_CHARS = set(".!?;:,\n。！？，")
READOUTS = ("max", "min", "mean", "summ")
# (feature name, readout, sign): the negative pole of a neuron is the same readout sign-flipped (max -> -min)
FEATURES = (("max+", "max", 1.0), ("mean+", "mean", 1.0), ("summ+", "summ", 1.0),
            ("max-", "min", -1.0), ("mean-", "mean", -1.0), ("summ-", "summ", -1.0))


@dataclass
class FindConfig:
    n_singles: int = 16
    composite_ks: tuple = (4, 8, 16, 32)
    n_controls: int = 6
    doses: tuple = (0.4, 0.8, 1.5)
    n_prompts: int = 5
    max_new: int = 80
    batch: int = 48
    judge: str = "claude"  # "claude" or "local" (the loaded model answers yes/no; for offline testing)
    min_coherence: float = 6.0
    min_lift: float = 1.5
    compose_max: int = 3
    refine_rounds: int = 2
    seed: int = 0
    auroc_discover: tuple = (0.85, 0.90)  # (vs near-miss, vs control) on the discovery half
    auroc_confirm: tuple = (0.75, 0.80)  # on the held-out half


@dataclass
class Candidate:
    name: str
    kind: str  # single | composite | control
    interventions: list  # list[Intervention] at dose 1.0 (singles/controls: rel; composites: raw, normalized)
    info: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)  # dose -> {"property", "coherence", "texts"}

    def at(self, dose: float) -> list[Intervention]:
        return [Intervention(iv.layer, iv.neuron, iv.pole, iv.mode, iv.dose * dose, iv.unit) for iv in self.interventions]


# -- 2. readouts ---------------------------------------------------------------------------------------------------
def _wrap(lm: LoadedModel, text: str) -> tuple[list[int], list[int], list[int]]:
    """Token ids for `text` as the assistant's reply, the content positions (text + end-of-turn), and the summary
    positions (sentence punctuation, last text token, end-of-turn)."""
    prefix = lm.chat_ids([{"role": "user", "content": NEUTRAL_USER}], thinking=False)
    body = lm.tok(text, add_special_tokens=False)["input_ids"]
    end = lm.tok("<|im_end|>\n", add_special_tokens=False)["input_ids"]
    ids = prefix + body + end
    s, e = len(prefix), len(prefix) + len(body)
    content = list(range(s, e + 1))
    summ = [s + i for i, t in enumerate(body) if (lm.tok.decode([t]).rstrip(" ")[-1:] or "") in SUMMARY_CHARS]
    summ = sorted(set(summ + [e - 1, e]))
    return ids, content, summ


@torch.no_grad()
def readouts(lm: LoadedModel, texts: list[str], batch: int = 16) -> tuple[dict, torch.Tensor]:
    """Per text, layer and neuron: max / min / mean over the content and mean over summary positions
    ({name: float32 [n_texts, L, D] on cpu}), plus the residual stream leaving each layer averaged over the content
    ([n_texts, L, d_model])."""
    L, D = lm.n_layers, lm.d_mlp
    R = {r: torch.zeros(len(texts), L, D) for r in READOUTS}
    resid = torch.zeros(len(texts), L, lm.d_model)
    wrapped = [_wrap(lm, t) for t in texts]
    st = {}

    def mlp_hook(l):
        def hook(mod, args):
            a = args[0].float()
            cm, sm = st["content"][..., None], st["summ"][..., None]
            sl = st["slice"]
            R["max"][sl, l] = a.masked_fill(~cm, -torch.inf).amax(1).cpu()
            R["min"][sl, l] = a.masked_fill(~cm, torch.inf).amin(1).cpu()
            R["mean"][sl, l] = ((a * cm).sum(1) / cm.sum(1)).cpu()
            R["summ"][sl, l] = ((a * sm).sum(1) / sm.sum(1)).cpu()
        return hook

    def layer_hook(l):
        def hook(mod, args, out):
            h = (out[0] if isinstance(out, tuple) else out).float()
            cm = st["content"][..., None]
            resid[st["slice"], l] = ((h * cm).sum(1) / cm.sum(1)).cpu()
        return hook

    hs = [lm.down_proj(l).register_forward_pre_hook(mlp_hook(l)) for l in range(L)]
    hs += [layer.register_forward_hook(layer_hook(l)) for l, layer in enumerate(lm.layers)]
    try:
        for b in range(0, len(texts), batch):
            chunk = wrapped[b:b + batch]
            T = max(len(ids) for ids, _, _ in chunk)
            content = torch.zeros(len(chunk), T, dtype=torch.bool)
            summ = torch.zeros_like(content)
            for i, (ids, c, s) in enumerate(chunk):
                off = T - len(ids)  # left padding
                content[i, [off + p for p in c]] = True
                summ[i, [off + p for p in s]] = True
            st.update(content=content.to(lm.device), summ=summ.to(lm.device), slice=slice(b, b + len(chunk)))
            enc = lm.tok.pad({"input_ids": [ids for ids, _, _ in chunk]}, padding=True, return_tensors="pt")
            lm.model(**enc.to(lm.device))
    finally:
        for h in hs:
            h.remove()
    return R, resid


# -- 3. scoring ----------------------------------------------------------------------------------------------------
def auroc(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """P(random row of a > random row of b), per column. a: [na, F], b: [nb, F]."""
    na, nb = a.shape[0], b.shape[0]
    ranks = torch.cat([a, b]).argsort(0).argsort(0).float() + 1
    return (ranks[:na].sum(0) - na * (na + 1) / 2) / (na * nb)


def score_neurons(lm: LoadedModel, R: dict, groups: dict, cfg: FindConfig, log=print) -> list[dict]:
    """Rank neuron poles that separate target from both near-miss and control texts."""
    dev = lm.device
    L, D = lm.n_layers, lm.d_mlp
    rng = np.random.default_rng(cfg.seed)
    T, N, C = (np.array(groups[k]) for k in ("target", "near_miss", "control"))
    pair_perm = rng.permutation(len(T))  # target/near-miss are matched pairs: split them together
    half = len(T) // 2
    cperm = rng.permutation(len(C))
    ix = lambda a: torch.as_tensor(a, dtype=torch.long, device=dev)
    disc = {"T": ix(T[pair_perm[:half]]), "N": ix(N[pair_perm[:half]]), "C": ix(C[cperm[:len(C) // 2]])}
    held = {"T": ix(T[pair_perm[half:]]), "N": ix(N[pair_perm[half:]]), "C": ix(C[cperm[len(C) // 2:]])}
    T, N, C, NC = ix(T), ix(N), ix(C), ix(np.concatenate([N, C]))

    index = lm.index
    best = {}  # (layer, neuron, pole) -> row
    n_disc = n_conf = 0
    for fname, rname, sign in FEATURES:
        X = (R[rname].to(dev).view(R[rname].shape[0], -1) * sign)  # [n_texts, L*D]
        X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        a_tn = auroc(X[disc["T"]], X[disc["N"]])
        a_tc = auroc(X[disc["T"]], X[disc["C"]])
        keep = (a_tn >= cfg.auroc_discover[0]) & (a_tc >= cfg.auroc_discover[1])
        n_disc += int(keep.sum())
        idx = keep.nonzero().squeeze(1)
        if len(idx) == 0:
            continue
        Xk = X[:, idx]
        h_tn = auroc(Xk[held["T"]], Xk[held["N"]])
        h_tc = auroc(Xk[held["T"]], Xk[held["C"]])
        ok = (h_tn >= cfg.auroc_confirm[0]) & (h_tc >= cfg.auroc_confirm[1])
        n_conf += int(ok.sum())
        idx, Xk = idx[ok], Xk[:, ok]
        if len(idx) == 0:
            continue
        # effect size against the neuron's own spread on non-target text, with a floor so near-constant neurons
        # don't win by dividing by ~0
        rest = Xk[NC]
        scale = Xk.abs().amax(0)
        if index is not None:
            pole = fname[-1]
            ref = torch.from_numpy(index.max if pole == "+" else -index.min).to(dev).view(-1)[idx]
            scale = torch.maximum(scale, ref)
        spread = rest.std(0) + 0.05 * scale + 1e-6
        mT = Xk[T].mean(0)
        d_tn = (mT - Xk[N].mean(0)) / spread
        d_tc = (mT - Xk[C].mean(0)) / spread
        eff = torch.minimum(d_tn, d_tc)
        a_all_tn, a_all_tc = auroc(Xk[T], Xk[N]), auroc(Xk[T], Xk[C])
        for j in range(len(idx)):
            flat = int(idx[j])
            key = (flat // D, flat % D, fname[-1])
            row = {"layer": key[0], "neuron": key[1], "pole": key[2], "readout": fname[:-1],
                   "effect": float(eff[j]), "auroc_tn": float(a_all_tn[j]), "auroc_tc": float(a_all_tc[j]),
                   "mean_target": float(mT[j])}
            if index is not None:
                row["frac_of_corpus_max"] = float(mT[j] / max(index.abs_max(*key), 1e-6))
            if key not in best or row["effect"] > best[key]["effect"]:
                best[key] = row
    rows = sorted(best.values(), key=lambda r: -r["effect"])
    log(f"scoring: {n_disc} feature hits on the discovery half, {n_conf} confirmed on held-out, "
        f"{len(rows)} distinct neuron poles")
    return rows


def _stratified(rows: list[dict], n: int, n_layers: int, bands: int = 4) -> list[dict]:
    """Top rows taken round-robin from early / mid / late layer bands. Raw effect size favours early-layer token
    detectors, which rarely steer well; mid-layer concept neurons get a turn this way."""
    by_band = [[r for r in rows if r["layer"] * bands // n_layers == b] for b in range(bands)]
    out = []
    while len(out) < n and any(by_band):
        for band in by_band:
            if band and len(out) < n:
                out.append(band.pop(0))
    return out


# -- 4. composites -------------------------------------------------------------------------------------------------
def composites(lm: LoadedModel, R: dict, resid: torch.Tensor, groups: dict, ks: tuple) -> tuple[list[Candidate], dict]:
    """Top-k neurons by their share of the property direction, as raw-unit interventions normalized so that dose 1.0
    pushes the residual stream by 1.0x its median norm (in the most-pushed layer)."""
    T, N = groups["target"], groups["near_miss"]
    dA = R["mean"][T].mean(0) - R["mean"][N].mean(0)  # [L, D]
    v = resid[T].mean(0) - resid[N].mean(0)  # [L, d_model]
    contrib = torch.zeros_like(dA)
    for l in range(lm.n_layers):
        vl = v[l]
        if vl.norm() == 0:
            continue
        proj = (vl / vl.norm()).to(lm.device) @ lm.down_proj(l).weight.float()  # [D]: W_down[:, n] · v̂
        contrib[l] = dA[l] * proj.cpu() / vl.norm()  # share of |v_l| written along v_l by neuron n
    pos = contrib.clamp(min=0)
    total = float(pos.sum()) or 1.0
    order = pos.view(-1).argsort(descending=True)
    D = lm.d_mlp
    cands, conc = [], {}
    for k in ks:
        sel = [(int(f) // D, int(f) % D) for f in order[:k]]
        conc[k] = float(sum(pos[l, n] for l, n in sel)) / total
        # push per layer at g = 1, then normalize g so the strongest layer's push = 1.0 x its median resid norm
        push = {}
        for l, n in sel:
            push[l] = push.get(l, 0) + float(dA[l, n]) * lm.down_proj(l).weight[:, n].float()
        worst = max(float(p.norm()) / lm.resid_norm[l] for l, p in push.items())
        g = 1.0 / max(worst, 1e-9)
        ivs = [Intervention(l, n, "+" if dA[l, n] > 0 else "-", "add", abs(float(dA[l, n])) * g, "raw") for l, n in sel]
        cands.append(Candidate(f"top{k}", "composite", ivs,
                               {"k": k, "share": conc[k], "layers": sorted({l for l, _ in sel})}))
    return cands, {"positive_share_by_k": conc, "best_layers": sorted(
        range(lm.n_layers), key=lambda l: -float(pos[l].sum()))[:5]}


# -- 5. verification -----------------------------------------------------------------------------------------------
def _rep4(ids: list[int]) -> float:
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


@torch.no_grad()
def _nll(lm: LoadedModel, steerer: Steerer, prompts: list[list[int]], gens: list[list[int]], batch: int) -> list[float]:
    """Mean per-token NLL of each generation under the unsteered model."""
    out = []
    with steerer.disabled():
        for b in range(0, len(gens), batch):
            seqs = [p + g for p, g in zip(prompts[b:b + batch], gens[b:b + batch])]
            enc = lm.tok.pad({"input_ids": seqs}, padding=True, return_tensors="pt").to(lm.device)
            lp = torch.log_softmax(lm.model(**enc).logits[:, :-1].float(), -1)
            tgt = enc["input_ids"][:, 1:]
            tok_lp = lp.gather(-1, tgt[..., None])[..., 0]
            Tn = enc["input_ids"].shape[1]
            for i, g in enumerate(gens[b:b + batch]):
                if not g:
                    out.append(10.0)
                    continue
                out.append(float(-tok_lp[i, Tn - 1 - len(g):].mean()))
    return out


@torch.no_grad()
def _local_judge(lm: LoadedModel, steerer: Steerer, prop: str, texts: list[str], batch: int) -> list[float]:
    """P(yes) x 10 from the loaded model itself. Only for offline testing: small models are weak judges."""
    yes = [lm.tok.convert_tokens_to_ids(t) for t in ("yes", "Yes")]
    no = [lm.tok.convert_tokens_to_ids(t) for t in ("no", "No")]
    seqs = [lm.chat_ids([{"role": "user", "content":
                          f"Text:\n\"\"\"\n{t.strip()}\n\"\"\"\n\nDoes this text clearly show the property "
                          f"\"{prop}\"? Answer with one word, yes or no."}]) for t in texts]
    out = []
    with steerer.disabled():
        for b in range(0, len(seqs), batch):
            enc = lm.tok.pad({"input_ids": seqs[b:b + batch]}, padding=True, return_tensors="pt").to(lm.device)
            p = torch.softmax(lm.model(**enc).logits[:, -1].float(), -1)
            py, pn = p[:, yes].sum(-1), p[:, no].sum(-1)
            out += (10 * py / (py + pn + 1e-9)).tolist()
    return out


def rollouts(lm: LoadedModel, steerer: Steerer, jobs: list[tuple[list[Intervention], int]], prompts: list[str],
             prop: str, rubric: str, cfg: FindConfig, log=print) -> list[dict]:
    """Run (interventions, prompt index) jobs, then judge everything blind. Returns one dict per job."""
    pids = [lm.chat_ids([{"role": "user", "content": p}]) for p in prompts]
    gens = []
    t0 = time.time()
    for b in range(0, len(jobs), cfg.batch):
        chunk = jobs[b:b + cfg.batch]
        gens += G.generate_per_row(lm, steerer, [pids[p] for _, p in chunk], [ivs for ivs, _ in chunk], cfg.max_new)
    log(f"  generated {len(jobs)} rollouts ({time.time() - t0:.0f}s)")
    texts = [lm.tok.decode(g, skip_special_tokens=True) for g in gens]
    reps = [_rep4(g) for g in gens]
    nlls = _nll(lm, steerer, [pids[p] for _, p in jobs], gens, cfg.batch)
    order = list(range(len(texts)))
    random.Random(cfg.seed).shuffle(order)  # the judge sees candidates, controls and baselines interleaved
    shuffled = [texts[i] for i in order]
    if cfg.judge == "claude":
        from . import llm
        got = llm.judge(prop, rubric, shuffled)
        ratings = [None] * len(texts)
        for j, i in enumerate(order):
            ratings[i] = got[j]
        props = [r[0] for r in ratings]
        cohs = [min(r[1], 3.0) if rep > 0.6 else r[1] for r, rep in zip(ratings, reps)]
    else:
        props = _local_judge(lm, steerer, prop, texts, cfg.batch)
        cohs = [10 * (1 - rep) * min(max((6 - nll) / 3, 0), 1) for rep, nll in zip(reps, nlls)]
    log(f"  judged ({cfg.judge}) ({time.time() - t0:.0f}s)")
    return [{"text": t, "property": p, "coherence": c, "rep4": r, "nll": n}
            for t, p, c, r, n in zip(texts, props, cohs, reps, nlls)]


def _summarize(rs: list[dict]) -> dict:
    return {"property": float(np.mean([r["property"] for r in rs])),
            "coherence": float(np.mean([r["coherence"] for r in rs])),
            "texts": [r["text"] for r in rs]}


def verify(lm: LoadedModel, steerer: Steerer, cands: list[Candidate], prompts: list[str], prop: str, rubric: str,
           cfg: FindConfig, log=print) -> dict:
    """Steer every candidate at every dose on every prompt (plus unsteered baselines) and score them."""
    jobs, keys = [], []
    for ci, c in enumerate(cands):
        for d in cfg.doses:
            for p in range(len(prompts)):
                jobs.append((c.at(d), p))
                keys.append((ci, d))
    for p in range(len(prompts)):
        jobs.append(([], p))
        keys.append((-1, 0.0))
    log(f"verification: {len(cands)} candidates x {len(cfg.doses)} doses x {len(prompts)} prompts")
    res = rollouts(lm, steerer, jobs, prompts, prop, rubric, cfg, log)
    grouped = {}
    for k, r in zip(keys, res):
        grouped.setdefault(k, []).append(r)
    baseline = _summarize(grouped[(-1, 0.0)])
    for ci, c in enumerate(cands):
        for d in cfg.doses:
            c.results[d] = _summarize(grouped[(ci, d)])
    ctrl = {d: float(np.mean([c.results[d]["property"] for c in cands if c.kind == "control"] or [0.0]))
            for d in cfg.doses}
    ver = {"baseline": baseline, "control_property": ctrl}
    for c in cands:
        _set_best(c, ver, cfg)
    return ver


def _reference(d: float, ver: dict) -> float:
    """Property score a meaningless push reaches at dose d: the better of the unsteered baseline and the random
    controls at the nearest measured dose at or above d (conservative for doses between measured ones)."""
    ctrl = ver["control_property"]
    above = [x for x in sorted(ctrl) if x >= d - 1e-9] or [max(ctrl)]
    return max(ver["baseline"]["property"], ctrl[above[0]])


def _set_best(c: Candidate, ver: dict, cfg: FindConfig):
    best = None
    for d in sorted(c.results):
        r = c.results[d]
        lift = r["property"] - _reference(d, ver)
        if r["coherence"] >= cfg.min_coherence and (best is None or lift > best["lift"]):
            best = {"dose": d, "lift": lift, **{k: r[k] for k in ("property", "coherence")}}
    c.info["best"] = best


def refine_doses(lm: LoadedModel, steerer: Steerer, cands: list[Candidate], prompts: list[str], prop: str,
                 rubric: str, ver: dict, cfg: FindConfig, log=print, max_cands: int = 8):
    """Where a candidate is coherent at one dose but only shows the property once it has broken down at the next,
    the usable window lies in between: try two intermediate doses there."""
    gaps = []
    for ci, c in enumerate(cands):
        if c.kind == "control":
            continue
        ds = sorted(c.results)
        for lo, hi in zip(ds, ds[1:]):
            rl, rh = c.results[lo], c.results[hi]
            if rl["coherence"] >= cfg.min_coherence > rh["coherence"] and rh["property"] >= rl["property"] + 1:
                gaps.append((rh["property"], ci, lo, hi))
    gaps = sorted(gaps, reverse=True)[:max_cands]
    if not gaps:
        return False
    jobs, keys = [], []
    for _, ci, lo, hi in gaps:
        for d in (round(lo + (hi - lo) / 3, 3), round(lo + 2 * (hi - lo) / 3, 3)):
            for p in range(len(prompts)):
                jobs.append((cands[ci].at(d), p))
                keys.append((ci, d))
    log(f"dose refinement: {len(gaps)} candidates jump from coherent to broken between doses; "
        f"trying intermediate doses")
    res = rollouts(lm, steerer, jobs, prompts, prop, rubric, cfg, log)
    grouped = {}
    for k, r in zip(keys, res):
        grouped.setdefault(k, []).append(r)
    for (ci, d), rs in grouped.items():
        cands[ci].results[d] = _summarize(rs)
    for ci in {ci for _, ci, _, _ in gaps}:
        _set_best(cands[ci], ver, cfg)
    return True


# -- 6. compose ----------------------------------------------------------------------------------------------------
def _merge(ivs: list[Intervention]) -> list[Intervention] | None:
    keys = [iv.key for iv in ivs]
    return None if len(keys) != len(set(keys)) else ivs


def compose(lm: LoadedModel, steerer: Steerer, levers: list[Candidate], prompts: list[str], prop: str, rubric: str,
            ref_prop: float, cfg: FindConfig, log=print) -> tuple[list[Intervention], dict]:
    cur_ivs = levers[0].at(levers[0].info["best"]["dose"])
    cur = dict(levers[0].info["best"], members=[levers[0].name])
    tried = 0
    for lev in levers[1:]:
        if tried >= cfg.compose_max:
            break
        lev_ivs = lev.at(lev.info["best"]["dose"])
        if {iv.key for iv in lev_ivs} & {iv.key for iv in cur_ivs}:
            continue  # overlaps what we already have (e.g. top8 contains top4)
        tried += 1
        trials = []
        for s in (0.6, 0.85):
            ivs = _merge([Intervention(iv.layer, iv.neuron, iv.pole, iv.mode, iv.dose * s, iv.unit)
                          for iv in cur_ivs + lev_ivs])
            if ivs:
                trials.append((s, ivs))
        if not trials:
            continue
        log(f"compose: trying {cur['members']} + {lev.name}")
        jobs = [(ivs, p) for _, ivs in trials for p in range(len(prompts))]
        res = rollouts(lm, steerer, jobs, prompts, prop, rubric, cfg, log)
        for t, (s, ivs) in enumerate(trials):
            r = _summarize(res[t * len(prompts):(t + 1) * len(prompts)])
            lift = r["property"] - ref_prop
            log(f"  scale {s}: property {r['property']:.1f}, coherence {r['coherence']:.1f}, lift {lift:+.1f}")
            if r["coherence"] >= cfg.min_coherence and lift >= cur["lift"] + 0.5:
                cur_ivs = ivs
                cur = {"dose": None, "lift": lift, "property": r["property"], "coherence": r["coherence"],
                       "members": cur["members"] + [lev.name], "samples": r["texts"]}
    return cur_ivs, cur


# -- driver --------------------------------------------------------------------------------------------------------
def find(lm: LoadedModel, steerer: Steerer, prop: str, cfg: FindConfig = FindConfig(), contrast: dict | None = None,
         log=print) -> tuple[Recipe | None, dict]:
    t0 = time.time()
    if contrast is None:
        from . import llm
        log(f"asking {llm.MODEL} for a contrast set ...")
        contrast = llm.contrast_set(prop)
    texts = contrast["target"] + contrast["near_miss"] + contrast["control"]
    nT, nN = len(contrast["target"]), len(contrast["near_miss"])
    groups = {"target": list(range(nT)), "near_miss": list(range(nT, nT + nN)),
              "control": list(range(nT + nN, len(texts)))}
    log(f"contrast set: {nT} target, {nN} near-miss, {len(contrast['control'])} control texts")
    if nT != nN or nT < 8:
        raise ValueError("need matched target/near_miss lists of at least 8 texts each")
    if lm.index is None:
        log("note: no neuron index for this model (`nf index`); effect sizes use only the contrast texts")

    R, resid = readouts(lm, texts)
    rows = score_neurons(lm, R, groups, cfg, log)
    singles = [Candidate(f"L{r['layer']}N{r['neuron']}{r['pole']}", "single",
                         [Intervention(r["layer"], r["neuron"], r["pole"], "add", 1.0, "rel")], dict(r))
               for r in _stratified(rows, cfg.n_singles, lm.n_layers)]
    comps, conc = composites(lm, R, resid, groups, cfg.composite_ks)
    log("composites: top-k share of the positive MLP contribution: " +
        ", ".join(f"k={k}: {s:.0%}" for k, s in conc["positive_share_by_k"].items()))
    rng = random.Random(cfg.seed + 1)
    ctrl_layers = [c.interventions[0].layer for c in singles] or list(range(lm.n_layers))
    controls = [Candidate(f"ctrl{i}", "control",
                          [Intervention(rng.choice(ctrl_layers), rng.randrange(lm.d_mlp), rng.choice("+-"),
                                        "add", 1.0, "rel")])
                for i in range(cfg.n_controls)]
    cands = singles + comps + controls
    prompts = contrast["steer_prompts"][:cfg.n_prompts]
    rubric = contrast.get("rubric", "")
    ver = verify(lm, steerer, cands, prompts, contrast.get("property", prop), rubric, cfg, log)
    for _ in range(cfg.refine_rounds):
        if not refine_doses(lm, steerer, cands, prompts, contrast.get("property", prop), rubric, ver, cfg, log):
            break

    levers = sorted([c for c in cands if c.kind != "control" and c.info.get("best")
                     and c.info["best"]["lift"] >= cfg.min_lift], key=lambda c: -c.info["best"]["lift"])
    log(f"verified levers: {len(levers)} (lift >= {cfg.min_lift} over controls/baseline at coherence >= "
        f"{cfg.min_coherence})")
    report = {
        "property": prop, "model": lm.model_id, "config": asdict(cfg), "contrast": contrast,
        "scored": rows[:200], "concentration": conc, "baseline": ver["baseline"],
        "control_property": {str(k): v for k, v in ver["control_property"].items()},
        "candidates": [{"name": c.name, "kind": c.kind, "info": c.info,
                        "interventions": [asdict(iv) for iv in c.interventions],
                        "results": {str(d): r for d, r in c.results.items()}} for c in cands],
    }
    recipe = None
    if levers:
        # same reference as verification: the better of the unsteered baseline and the random-neuron controls
        ref = max(ver["baseline"]["property"], float(np.mean(list(ver["control_property"].values()))))
        ivs, best = compose(lm, steerer, levers, prompts, contrast.get("property", prop), rubric, ref, cfg, log)
        samples = best.pop("samples", None) or levers[0].results[levers[0].info["best"]["dose"]]["texts"]
        recipe = Recipe(model=lm.model_id, interventions=ivs, property=prop,
                        scores={**best, "baseline_property": ver["baseline"]["property"]},
                        notes=json.dumps({"prompts": prompts, "samples": samples}, ensure_ascii=False))
        report["recipe"] = asdict(recipe)
    report["seconds"] = time.time() - t0
    return recipe, report


def save_report(report: dict, path: str | Path):
    Path(path).write_text(json.dumps(report, indent=1, ensure_ascii=False, default=float) + "\n")
