"""Running extraction against the DeepSeek API.

DeepSeek is OpenAI-compatible, so this uses the ``openai`` client pointed at
``api.deepseek.com``. Two things about that API shape the design here, and both
were established by probing it rather than assuming:

**There is no strict schema mode.** ``response_format={"type": "json_schema"}``
is rejected outright ("This response_format type is unavailable now"), so the
schema is delivered as a function's parameters and enforced client-side by
Pydantic. The model can and does violate it — ``deepseek-reasoner`` will return
``"elevation_fov_deg": "null"`` as a quoted string where the schema says
number-or-null — so validation is a real step, not a formality.

**Forced tool_choice only works in non-thinking mode.** Every canonical model id
runs in thinking mode and rejects ``tool_choice="required"`` or a named function.
``tool_choice="auto"`` is the only mechanism that works across all of them, so
that is what is used, and a turn where the model declines to call the tool is
recorded as a failure.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import openai
from pydantic import ValidationError

from . import prompts
from .cost import Usage, estimate_cost
from .schema import SensorSpec, strict_json_schema

DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"

#: Extraction output is one flat object — a few hundred tokens. Thinking-mode
#: models spend tokens reasoning before the tool call, so this is well clear of
#: the answer's own size rather than snug against it.
MAX_TOKENS = 4096

TOOL_NAME = "record_spec"

#: Strings a model reaches for when it means "nothing here". The schema says to
#: return null; these are what comes back instead. Mapping them to None measures
#: extraction rather than JSON etiquette — but the count is reported, so the
#: serialisation defect stays visible instead of being silently absorbed.
NULLISH = frozenset({
    "", "null", "none", "n/a", "na", "nil",
    "unknown", "not specified", "not stated",
})


def build_client(api_key: str | None = None, base_url: str | None = None) -> openai.OpenAI:
    """Build a DeepSeek client.

    The key is read from ``DEEPSEEK_API_KEY`` and has no default on purpose. A
    hardcoded fallback key in a public repository is scraped within minutes of
    the push and cannot be removed from git history afterwards.
    """
    key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. Export it (or put it in a .env you do "
            "not commit) — never hardcode it in the source."
        )
    return openai.OpenAI(
        api_key=key,
        base_url=base_url or os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
    )


def _tool_definition() -> dict:
    """The schema, delivered as a function the model can call.

    Passing SensorSpec's JSON schema as the parameters is what carries the field
    descriptions — the unit rules, the half-angle convention, "null if not
    stated" — to the model. It is prompt engineering that lives in the type.
    """
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Record the specifications extracted from the datasheet.",
            "parameters": strict_json_schema(SensorSpec),
        },
    }


def repair_nullish(raw: dict) -> tuple[dict, list[str]]:
    """Map null-ish strings to None. Returns the repaired dict and what changed."""
    repaired: dict = {}
    touched: list[str] = []
    for key, value in raw.items():
        if isinstance(value, str) and value.strip().casefold() in NULLISH:
            repaired[key] = None
            touched.append(key)
        else:
            repaired[key] = value
    return repaired, touched


@dataclass
class Extraction:
    """One document's extraction attempt, successful or not."""

    doc_id: str
    model: str
    variant: str
    spec: SensorSpec | None
    usage: Usage
    latency_s: float
    error: str | None = None
    repaired_fields: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.spec is not None

    @property
    def cost_usd(self) -> float:
        return estimate_cost(self.model, self.usage)


def _failed(doc_id, model, variant, usage, started, error) -> Extraction:
    return Extraction(
        doc_id=doc_id,
        model=model,
        variant=variant,
        spec=None,
        usage=usage,
        latency_s=time.perf_counter() - started,
        error=error,
    )


def extract_one(
    client: openai.OpenAI,
    doc_id: str,
    document_text: str,
    model: str = DEFAULT_MODEL,
    variant: str = prompts.DEFAULT_VARIANT,
) -> Extraction:
    """Extract one datasheet. Never raises — failures land in ``Extraction.error``.

    A run over a corpus should not lose 40 good extractions because the 41st hit
    a rate limit, so API failures are captured per document. The evaluator scores
    a failed extraction as a miss on every field.
    """
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": prompts.get(variant)},
                {"role": "user", "content": prompts.user_message(document_text)},
            ],
            tools=[_tool_definition()],
            tool_choice="auto",
        )
    except openai.APIStatusError as exc:
        return _failed(doc_id, model, variant, Usage(), started, f"{type(exc).__name__}: {exc}")
    except openai.APIConnectionError as exc:
        return _failed(doc_id, model, variant, Usage(), started, f"APIConnectionError: {exc}")

    latency = time.perf_counter() - started
    usage = Usage.from_response(response.usage)
    choice = response.choices[0]

    calls = choice.message.tool_calls
    if not calls:
        # tool_choice="auto" is the only portable option, so this is reachable:
        # the model answered in prose instead of calling the tool.
        preview = (choice.message.content or "")[:120]
        return _failed(
            doc_id, model, variant, usage, started,
            f"no tool call (finish_reason={choice.finish_reason}): {preview!r}",
        )

    try:
        raw = json.loads(calls[0].function.arguments)
    except json.JSONDecodeError as exc:
        return _failed(
            doc_id, model, variant, usage, started, f"tool arguments were not JSON: {exc}"
        )

    if not isinstance(raw, dict):
        return _failed(
            doc_id, model, variant, usage, started,
            f"tool arguments were {type(raw).__name__}, not an object",
        )

    repaired, touched = repair_nullish(raw)
    try:
        spec = SensorSpec.model_validate(repaired)
    except ValidationError as exc:
        # DeepSeek does not enforce the schema server-side, so this is a real
        # failure mode rather than defensive padding.
        first = exc.errors()[0]
        return _failed(
            doc_id, model, variant, usage, started,
            f"schema violation at {'.'.join(str(p) for p in first['loc'])}: {first['msg']}",
        )

    return Extraction(
        doc_id=doc_id,
        model=model,
        variant=variant,
        spec=spec,
        usage=usage,
        latency_s=latency,
        repaired_fields=tuple(touched),
    )


def extract_corpus(
    client: openai.OpenAI,
    documents: dict[str, str],
    model: str = DEFAULT_MODEL,
    variant: str = prompts.DEFAULT_VARIANT,
    on_result=None,
) -> list[Extraction]:
    """Extract every document, one request each."""
    results = []
    for doc_id, text in documents.items():
        result = extract_one(client, doc_id, text, model=model, variant=variant)
        results.append(result)
        if on_result is not None:
            on_result(result)
    return results
