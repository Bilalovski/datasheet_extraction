"""Running extraction against the Claude API.

Uses structured outputs (``messages.parse``) rather than asking for JSON in
prose and parsing it: the schema is enforced server-side, so a response either
validates against :class:`~datasheet_extraction.schema.SensorSpec` or the call
fails loudly. There is no regex-the-model's-JSON path to go wrong.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import anthropic

from . import prompts
from .cost import Usage, estimate_cost
from .schema import SensorSpec, strict_json_schema

DEFAULT_MODEL = "claude-opus-4-8"

#: Extraction output is one flat object — a few hundred tokens at most. This is
#: sized to leave headroom, not to be reached; a response that hits it comes
#: back with stop_reason "max_tokens" and is recorded as a failure.
MAX_TOKENS = 2048

#: Shortest prefix each model will cache, in tokens. A shorter prefix is not an
#: error — it silently does not cache, reporting cache_creation_input_tokens: 0.
#: Sonnet 5 is absent from the published table, so it is not listed here rather
#: than guessed at; caching is skipped for models with no known threshold.
MIN_CACHEABLE_TOKENS: dict[str, int] = {
    "claude-opus-4-8": 4096,
    "claude-opus-4-7": 4096,
    "claude-opus-4-6": 4096,
    "claude-haiku-4-5": 4096,
    "claude-fable-5": 2048,
    "claude-sonnet-4-6": 2048,
}


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

    @property
    def ok(self) -> bool:
        return self.spec is not None

    @property
    def cost_usd(self) -> float:
        return estimate_cost(self.model, self.usage)


def _system_blocks(variant: str, model: str, cache: bool) -> list[dict] | str:
    """Build the system prompt, optionally marked for caching.

    Caching is only requested when the prompt could plausibly clear the model's
    minimum cacheable prefix. Marking a shorter prefix is not harmful, just
    inert — but silently inert, which is worse than not asking for it.
    """
    text = prompts.get(variant)
    if not cache:
        return text

    threshold = MIN_CACHEABLE_TOKENS.get(model)
    # ~4 chars per token is rough, but the decision only needs the order of
    # magnitude: these prompts are ~2k characters against a 4k-token floor.
    if threshold is None or len(text) / 4 < threshold:
        return text

    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def extract_one(
    client: anthropic.Anthropic,
    doc_id: str,
    document_text: str,
    model: str = DEFAULT_MODEL,
    variant: str = prompts.DEFAULT_VARIANT,
    cache_system: bool = False,
) -> Extraction:
    """Extract one datasheet. Never raises — failures land in ``Extraction.error``.

    A run over a corpus should not lose 40 good extractions because the 41st hit
    a rate limit, so API failures are captured per document. The evaluator scores
    a failed extraction as a miss on every field.
    """
    started = time.perf_counter()
    try:
        response = client.messages.parse(
            model=model,
            max_tokens=MAX_TOKENS,
            system=_system_blocks(variant, model, cache_system),
            messages=[{"role": "user", "content": prompts.user_message(document_text)}],
            output_format=SensorSpec,
        )
    except anthropic.APIStatusError as exc:
        return Extraction(
            doc_id=doc_id,
            model=model,
            variant=variant,
            spec=None,
            usage=Usage(),
            latency_s=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
    except anthropic.APIConnectionError as exc:
        return Extraction(
            doc_id=doc_id,
            model=model,
            variant=variant,
            spec=None,
            usage=Usage(),
            latency_s=time.perf_counter() - started,
            error=f"APIConnectionError: {exc}",
        )

    latency = time.perf_counter() - started
    usage = Usage.from_response(response.usage)

    # A refusal or a truncated response both yield no usable object; record the
    # reason rather than letting a None parsed_output look like a clean null.
    if response.stop_reason not in ("end_turn", None) or response.parsed_output is None:
        return Extraction(
            doc_id=doc_id,
            model=model,
            variant=variant,
            spec=None,
            usage=usage,
            latency_s=latency,
            error=f"no parsed output (stop_reason={response.stop_reason})",
        )

    return Extraction(
        doc_id=doc_id,
        model=model,
        variant=variant,
        spec=response.parsed_output,
        usage=usage,
        latency_s=latency,
    )


def extract_corpus(
    client: anthropic.Anthropic,
    documents: dict[str, str],
    model: str = DEFAULT_MODEL,
    variant: str = prompts.DEFAULT_VARIANT,
    cache_system: bool = False,
    on_result=None,
) -> list[Extraction]:
    """Extract every document, one request each.

    For corpora past a few dozen documents, prefer :func:`submit_batch` — the
    Batch API halves the price for work that doesn't need to be interactive.
    """
    results = []
    for doc_id, text in documents.items():
        result = extract_one(
            client, doc_id, text, model=model, variant=variant, cache_system=cache_system
        )
        results.append(result)
        if on_result is not None:
            on_result(result)
    return results


def submit_batch(
    client: anthropic.Anthropic,
    documents: dict[str, str],
    model: str = DEFAULT_MODEL,
    variant: str = prompts.DEFAULT_VARIANT,
) -> str:
    """Queue a corpus on the Batch API and return the batch id.

    The Batch API has no ``parse()`` helper, so the schema goes over the wire as
    ``output_config.format`` and the response comes back as JSON text to validate
    client-side — same schema, same guarantees, more assembly.
    """
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    schema = strict_json_schema(SensorSpec)
    system = prompts.get(variant)

    batch = client.messages.batches.create(
        requests=[
            Request(
                custom_id=doc_id,
                params=MessageCreateParamsNonStreaming(
                    model=model,
                    max_tokens=MAX_TOKENS,
                    system=system,
                    messages=[
                        {"role": "user", "content": prompts.user_message(text)}
                    ],
                    output_config={"format": {"type": "json_schema", "schema": schema}},
                ),
            )
            for doc_id, text in documents.items()
        ]
    )
    return batch.id


def collect_batch(
    client: anthropic.Anthropic, batch_id: str, model: str, variant: str
) -> list[Extraction]:
    """Read a finished batch's results into :class:`Extraction` objects.

    Results arrive in arbitrary order, so they are keyed by ``custom_id`` — the
    document id submitted with each request — never by position.
    """
    extractions = []
    for result in client.messages.batches.results(batch_id):
        doc_id = result.custom_id
        if result.result.type != "succeeded":
            extractions.append(
                Extraction(
                    doc_id=doc_id,
                    model=model,
                    variant=variant,
                    spec=None,
                    usage=Usage(),
                    latency_s=0.0,
                    error=f"batch result: {result.result.type}",
                )
            )
            continue

        message = result.result.message
        usage = Usage.from_response(message.usage)
        text = next((b.text for b in message.content if b.type == "text"), None)
        try:
            spec = SensorSpec.model_validate_json(text) if text else None
            error = None if spec else "batch result had no text block"
        except Exception as exc:  # pydantic validation
            spec, error = None, f"schema validation failed: {exc}"

        extractions.append(
            Extraction(
                doc_id=doc_id,
                model=model,
                variant=variant,
                spec=spec,
                usage=usage,
                latency_s=0.0,
                error=error,
            )
        )
    return extractions
