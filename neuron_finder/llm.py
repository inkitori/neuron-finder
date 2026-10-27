"""Claude API calls: contrast-set generation and blind judging of steered rollouts.

Model: `claude-opus-5-5` (override with NF_CLAUDE_MODEL). Credentials come from the environment
(ANTHROPIC_API_KEY or an `ant auth login` profile). Requests use structured outputs so replies are valid JSON, and
opt into server-side refusal fallbacks. Compute nodes share one outbound proxy, so judge calls are batched and run
with low concurrency.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor

MODEL = os.environ.get("NF_CLAUDE_MODEL", "claude-opus-5-5")
MAX_CONCURRENCY = 4

CONTRAST_SCHEMA = {
    "type": "object",
    "properties": {
        "property": {"type": "string"},
        "target": {"type": "array", "items": {"type": "string"}},
        "near_miss": {"type": "array", "items": {"type": "string"}},
        "control": {"type": "array", "items": {"type": "string"}},
        "steer_prompts": {"type": "array", "items": {"type": "string"}},
        "rubric": {"type": "string"},
    },
    "required": ["property", "target", "near_miss", "control", "steer_prompts", "rubric"],
    "additionalProperties": False,
}

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "ratings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "property": {"type": "integer"},
                    "coherence": {"type": "integer"},
                },
                "required": ["id", "property", "coherence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["ratings"],
    "additionalProperties": False,
}

CONTRAST_PROMPT = """\
I'm searching for neurons in a small open language model (Qwen) that make it exhibit this property:

    {prop}

I'll run three groups of texts through the model and look for neurons that fire on the first group but not the
other two, so the groups need to be carefully matched. Each text will be shown to the model as its *own reply* in
a chat, so write them in the voice of an assistant or a person replying, 1-3 sentences each.

- target ({n}): texts that clearly exhibit the property. Vary topic, length and phrasing widely so the only thing
  they share is the property itself.
- near_miss ({n}): one per target text, in the same order. Same topic, much of the same vocabulary, similar length,
  but WITHOUT the property. These separate the property from its usual topics and words, so they matter most.
- control ({n}): unrelated, neutral texts on varied subjects (science, cooking, travel, code, history, small talk).

Also give:
- steer_prompts (6): short, open-ended user messages that don't invite the property on their own (e.g. "Tell me
  about your weekend."), used to test whether pushing a neuron makes the property appear unprompted.
- rubric: one or two sentences telling a judge how to rate, 0-10, how strongly a piece of text exhibits the
  property (0 = not at all, 10 = unmistakably and throughout).
- property: a short restatement of the property.
"""

JUDGE_PROMPT = """\
Rate each text below on two 0-10 scales.

property: how strongly the text exhibits this property: {prop}
  {rubric}
coherence: how fluent and readable the text is as English (or another natural language). 10 = fluent, 5 = odd
  but understandable, 0 = gibberish or a degenerate loop repeating the same words.

Rate the property independently of coherence where possible. Some texts are cut off mid-sentence because of a
length limit; don't penalize that. Return one rating per id.

{items}
"""


def _client():
    import anthropic
    return anthropic.Anthropic()


def available() -> bool:
    """True if credentials for the Claude API appear to be configured."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return os.path.isdir(os.path.expanduser("~/.config/anthropic"))


def _structured(prompt: str, schema: dict, effort: str, max_tokens: int = 16000) -> dict:
    import anthropic

    try:
        resp = _client().beta.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            betas=["server-side-fallback-2026-07-01"],
            extra_body={"fallbacks": "default"},
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.AuthenticationError as e:
        raise RuntimeError("Claude API authentication failed; set ANTHROPIC_API_KEY or run `ant auth login`") from e
    except anthropic.RateLimitError as e:
        raise RuntimeError(f"Claude API rate limit: {e.message}") from e
    except anthropic.APIStatusError as e:
        raise RuntimeError(f"Claude API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise RuntimeError(f"could not reach the Claude API: {e}") from e
    if resp.stop_reason == "refusal":
        cat = getattr(resp.stop_details, "category", None) if resp.stop_details else None
        raise RuntimeError(f"Claude declined the request (category: {cat})")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("Claude's reply hit max_tokens before the JSON was complete")
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text)


def contrast_set(prop: str, n: int = 30) -> dict:
    """Target / near-miss / control texts, neutral steering prompts, and a judge rubric for `prop`."""
    d = _structured(CONTRAST_PROMPT.format(prop=prop, n=n), CONTRAST_SCHEMA, effort="medium")
    m = min(len(d["target"]), len(d["near_miss"]))
    d["target"], d["near_miss"] = d["target"][:m], d["near_miss"][:m]
    return d


def judge(prop: str, rubric: str, texts: list[str], batch: int = 20) -> list[tuple[float, float]]:
    """(property, coherence) ratings 0-10 for each text. The judge sees only the texts, never which neuron or
    dose produced them; callers should shuffle baselines and controls in with the candidates."""
    chunks = [list(range(s, min(s + batch, len(texts)))) for s in range(0, len(texts), batch)]

    def run(ids):
        items = "\n\n".join(f"<text id={i}>\n{texts[i].strip() or '(empty)'}\n</text>" for i in ids)
        d = _structured(JUDGE_PROMPT.format(prop=prop, rubric=rubric, items=items), JUDGE_SCHEMA, effort="low")
        return {r["id"]: (float(r["property"]), float(r["coherence"])) for r in d["ratings"]}

    out = {}
    with ThreadPoolExecutor(MAX_CONCURRENCY) as ex:
        for got in ex.map(run, chunks):
            out.update(got)
    for i in [i for i in range(len(texts)) if i not in out]:  # the judge occasionally skips an id
        out.update(run([i]))
    return [out.get(i, (0.0, 0.0)) for i in range(len(texts))]
