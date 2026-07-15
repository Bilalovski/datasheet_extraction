"""Extraction tests. No API key, no network — the client is faked."""

from types import SimpleNamespace

import anthropic
import httpx
import pytest

from datasheet_extraction.extract import (
    MIN_CACHEABLE_TOKENS,
    _system_blocks,
    extract_corpus,
    extract_one,
)
from datasheet_extraction.schema import SensorSpec


class FakeMessages:
    def __init__(self, outcome):
        self._outcome = outcome
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


class FakeClient:
    def __init__(self, outcome):
        self.messages = FakeMessages(outcome)


def response(spec=None, stop_reason="end_turn", **usage):
    return SimpleNamespace(
        parsed_output=spec,
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=usage.get("input_tokens", 1200),
            output_tokens=usage.get("output_tokens", 180),
            cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
            cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
        ),
    )


class TestHappyPath:
    def test_returns_the_parsed_spec(self):
        spec = SensorSpec(part_number="RDX-7700", max_range_m=250.0)
        client = FakeClient(response(spec))

        result = extract_one(client, "rdx-7700", "some datasheet text")

        assert result.ok
        assert result.spec.part_number == "RDX-7700"
        assert result.error is None

    def test_records_usage_and_cost(self):
        client = FakeClient(response(SensorSpec(), input_tokens=1000, output_tokens=100))

        result = extract_one(client, "d", "text", model="claude-opus-4-8")

        assert result.usage.input_tokens == 1000
        assert result.cost_usd > 0

    def test_sends_the_schema_and_the_document(self):
        client = FakeClient(response(SensorSpec()))

        extract_one(client, "d", "DATASHEET BODY", variant="strict")

        call = client.messages.calls[0]
        assert call["output_format"] is SensorSpec
        assert "DATASHEET BODY" in call["messages"][0]["content"]


class TestFailuresAreCapturedNotRaised:
    """A corpus run must not lose good extractions to one bad request."""

    def test_api_status_error_is_recorded(self):
        error = anthropic.APIStatusError(
            "rate limited",
            response=httpx.Response(429, request=httpx.Request("POST", "http://x")),
            body=None,
        )
        result = extract_one(FakeClient(error), "d", "text")

        assert not result.ok
        assert "rate limited" in result.error

    def test_connection_error_is_recorded(self):
        error = anthropic.APIConnectionError(request=httpx.Request("POST", "http://x"))
        result = extract_one(FakeClient(error), "d", "text")

        assert not result.ok
        assert "APIConnectionError" in result.error

    def test_refusal_is_not_mistaken_for_a_clean_extraction(self):
        client = FakeClient(response(None, stop_reason="refusal"))

        result = extract_one(client, "d", "text")

        assert not result.ok
        assert "refusal" in result.error

    def test_truncated_response_is_recorded(self):
        client = FakeClient(response(None, stop_reason="max_tokens"))

        result = extract_one(client, "d", "text")

        assert not result.ok
        assert "max_tokens" in result.error

    def test_corpus_run_continues_past_a_failure(self):
        error = anthropic.APIConnectionError(request=httpx.Request("POST", "http://x"))
        results = extract_corpus(FakeClient(error), {"a": "x", "b": "y"})

        assert len(results) == 2
        assert all(not r.ok for r in results)


class TestSystemPromptCaching:
    def test_caching_off_by_default_sends_a_plain_string(self):
        blocks = _system_blocks("strict", "claude-opus-4-8", cache=False)
        assert isinstance(blocks, str)

    def test_short_prefix_is_not_marked_for_caching(self):
        # Every prompt variant here is far below Opus 4.8's 4096-token floor, so
        # a cache_control marker would be silently inert. Don't ask for it.
        blocks = _system_blocks("strict", "claude-opus-4-8", cache=True)
        assert isinstance(blocks, str)

    def test_unknown_model_threshold_skips_caching(self):
        assert "claude-sonnet-5" not in MIN_CACHEABLE_TOKENS
        blocks = _system_blocks("strict", "claude-sonnet-5", cache=True)
        assert isinstance(blocks, str)

    def test_long_prefix_over_the_threshold_is_marked(self, monkeypatch):
        from datasheet_extraction import prompts

        monkeypatch.setitem(prompts.VARIANTS, "huge", "x" * 40_000)
        blocks = _system_blocks("huge", "claude-opus-4-8", cache=True)

        assert isinstance(blocks, list)
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}


class TestPromptVariants:
    def test_unknown_variant_names_the_valid_ones(self):
        from datasheet_extraction import prompts

        with pytest.raises(KeyError, match="minimal"):
            prompts.get("nope")
