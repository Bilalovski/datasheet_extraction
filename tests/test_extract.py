"""Extraction tests. No API key, no network — the client is faked.

The failure modes exercised here are the ones the DeepSeek API actually
exhibits, established by probing it: no strict schema enforcement (so a model
can return a quoted "null" where a number belongs), and tool_choice="auto" being
the only portable option (so the model can decline to call the tool at all).
"""

import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from datasheet_extraction.extract import (
    NULLISH,
    build_client,
    extract_corpus,
    extract_one,
    repair_nullish,
)


class FakeCompletions:
    def __init__(self, outcome):
        self._outcome = outcome
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


class FakeClient:
    def __init__(self, outcome):
        self.chat = SimpleNamespace(completions=FakeCompletions(outcome))

    @property
    def calls(self):
        return self.chat.completions.calls


def response(arguments=None, content=None, finish_reason="tool_calls", **usage):
    """Build an OpenAI-shaped chat completion, with DeepSeek's cache fields."""
    tool_calls = None
    if arguments is not None:
        tool_calls = [
            SimpleNamespace(
                function=SimpleNamespace(
                    name="record_spec",
                    arguments=arguments if isinstance(arguments, str) else json.dumps(arguments),
                )
            )
        ]
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(tool_calls=tool_calls, content=content),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=usage.get("prompt_tokens", 400),
            completion_tokens=usage.get("completion_tokens", 80),
            prompt_cache_hit_tokens=usage.get("prompt_cache_hit_tokens", 0),
            prompt_cache_miss_tokens=usage.get("prompt_cache_miss_tokens", 400),
        ),
    )


class TestHappyPath:
    def test_returns_the_parsed_spec(self):
        client = FakeClient(response({"part_number": "RDX-7700", "max_range_m": 250}))

        result = extract_one(client, "rdx-7700", "text")

        assert result.ok
        assert result.spec.part_number == "RDX-7700"
        assert result.error is None

    def test_records_usage_and_cost(self):
        client = FakeClient(
            response({"part_number": "X"}, prompt_cache_miss_tokens=1000, completion_tokens=100)
        )

        result = extract_one(client, "d", "text", model="deepseek-v4-flash")

        assert result.usage.cache_miss_tokens == 1000
        assert result.cost_usd > 0

    def test_sends_the_schema_as_the_tool_parameters(self):
        # The field descriptions carrying the unit rules only reach the model if
        # the schema goes over as the function's parameters.
        client = FakeClient(response({"part_number": "X"}))

        extract_one(client, "d", "DATASHEET BODY")

        call = client.calls[0]
        params = call["tools"][0]["function"]["parameters"]
        assert "center_frequency_ghz" in params["properties"]
        assert "midpoint" in params["properties"]["center_frequency_ghz"]["description"]
        assert "DATASHEET BODY" in call["messages"][1]["content"]

    def test_uses_auto_tool_choice(self):
        # Forced tool_choice is rejected by every thinking-mode model, so auto is
        # the only portable option. Pinning this so it isn't "tidied" to required.
        client = FakeClient(response({"part_number": "X"}))

        extract_one(client, "d", "text")

        assert client.calls[0]["tool_choice"] == "auto"


class TestSchemaViolationsAreCaught:
    """DeepSeek has no strict schema mode, so client-side validation is real."""

    def test_quoted_null_in_a_numeric_field_is_repaired(self):
        # Observed from deepseek-reasoner: "elevation_fov_deg": "null".
        client = FakeClient(response({"part_number": "X", "elevation_fov_deg": "null"}))

        result = extract_one(client, "d", "text")

        assert result.ok
        assert result.spec.elevation_fov_deg is None
        assert "elevation_fov_deg" in result.repaired_fields

    def test_a_genuine_type_error_is_a_failure_not_a_repair(self):
        client = FakeClient(response({"max_range_m": "quite far"}))

        result = extract_one(client, "d", "text")

        assert not result.ok
        assert "schema violation at max_range_m" in result.error

    def test_unknown_field_is_rejected(self):
        client = FakeClient(response({"part_number": "X", "invented_field": 1}))

        result = extract_one(client, "d", "text")

        assert not result.ok
        assert "schema violation" in result.error

    def test_malformed_json_arguments_are_a_failure(self):
        client = FakeClient(response("{not json"))

        result = extract_one(client, "d", "text")

        assert not result.ok
        assert "not JSON" in result.error


class TestRepairNullish:
    def test_maps_nullish_strings_to_none(self):
        repaired, touched = repair_nullish({"a": "null", "b": "N/A", "c": "  none  "})
        assert repaired == {"a": None, "b": None, "c": None}
        assert set(touched) == {"a", "b", "c"}

    def test_leaves_real_values_alone(self):
        repaired, touched = repair_nullish({"interface": "CAN", "max_range_m": 250})
        assert repaired == {"interface": "CAN", "max_range_m": 250}
        assert touched == []

    def test_does_not_touch_actual_none(self):
        repaired, touched = repair_nullish({"a": None})
        assert repaired == {"a": None}
        assert touched == []

    def test_nullish_set_is_matched_case_insensitively(self):
        assert all(token == token.casefold() for token in NULLISH)
        repaired, _ = repair_nullish({"a": "NULL"})
        assert repaired["a"] is None


class TestFailuresAreCapturedNotRaised:
    def test_model_answering_in_prose_is_recorded(self):
        # Reachable because tool_choice must be "auto" on thinking-mode models.
        client = FakeClient(
            response(None, content="I could not find a datasheet here.", finish_reason="stop")
        )

        result = extract_one(client, "d", "text")

        assert not result.ok
        assert "no tool call" in result.error

    def test_api_status_error_is_recorded(self):
        error = openai.APIStatusError(
            "rate limited",
            response=httpx.Response(429, request=httpx.Request("POST", "http://x")),
            body=None,
        )
        result = extract_one(FakeClient(error), "d", "text")

        assert not result.ok
        assert "rate limited" in result.error

    def test_connection_error_is_recorded(self):
        error = openai.APIConnectionError(request=httpx.Request("POST", "http://x"))
        result = extract_one(FakeClient(error), "d", "text")

        assert not result.ok
        assert "APIConnectionError" in result.error

    def test_corpus_run_continues_past_a_failure(self):
        error = openai.APIConnectionError(request=httpx.Request("POST", "http://x"))
        results = extract_corpus(FakeClient(error), {"a": "x", "b": "y"})

        assert len(results) == 2
        assert all(not r.ok for r in results)


class TestClientConstruction:
    def test_missing_key_fails_loudly(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
            build_client()

    def test_no_hardcoded_key_exists_in_the_source(self):
        # The point of the check: a fallback key in a public repo is scraped
        # within minutes of the push and lives in git history forever.
        from pathlib import Path

        src = Path(__file__).resolve().parents[1] / "src" / "datasheet_extraction"
        for path in src.glob("*.py"):
            assert "sk-" not in path.read_text(encoding="utf-8"), f"possible key in {path.name}"

    def test_key_is_read_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-not-a-real-key")
        client = build_client()
        assert str(client.base_url).startswith("https://api.deepseek.com")


class TestPromptVariants:
    def test_unknown_variant_names_the_valid_ones(self):
        from datasheet_extraction import prompts

        with pytest.raises(KeyError, match="minimal"):
            prompts.get("nope")
