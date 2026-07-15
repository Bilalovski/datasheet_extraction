import pytest
from pydantic import ValidationError

from datasheet_extraction.cost import (
    CACHE_READ_MULTIPLIER,
    PRICING,
    Usage,
    estimate_cost,
)
from datasheet_extraction.schema import (
    FIELDS,
    NUMERIC_FIELDS,
    SensorSpec,
    strict_json_schema,
)


class TestSchema:
    def test_every_field_defaults_to_null(self):
        # "Not stated" has to be expressible for every field, or the evaluator
        # can't score abstention.
        spec = SensorSpec()
        assert all(getattr(spec, name) is None for name in FIELDS)

    def test_numeric_fields_derived_from_annotations(self):
        assert "center_frequency_ghz" in NUMERIC_FIELDS
        assert "tx_channels" in NUMERIC_FIELDS  # int, not float
        assert "part_number" not in NUMERIC_FIELDS
        assert "sensor_type" not in NUMERIC_FIELDS

    def test_rejects_unknown_fields(self):
        with pytest.raises(ValidationError):
            SensorSpec(bogus_field=1)


class TestStrictJsonSchema:
    def test_objects_forbid_additional_properties(self):
        schema = strict_json_schema(SensorSpec)
        assert schema["additionalProperties"] is False

    def test_every_property_is_required(self):
        # Structured outputs express optionality as a nullable type, not an
        # absent key — so every field must be listed in `required`.
        schema = strict_json_schema(SensorSpec)
        assert set(schema["required"]) == set(schema["properties"])

    def test_optional_fields_admit_null(self):
        schema = strict_json_schema(SensorSpec)
        arms = schema["properties"]["max_range_m"]["anyOf"]
        assert {"type": "null"} in arms


class TestUsage:
    def test_addition_accumulates_every_counter(self):
        total = Usage(input_tokens=10, output_tokens=1) + Usage(
            input_tokens=5, output_tokens=2, cache_read_input_tokens=7
        )
        assert total.input_tokens == 15
        assert total.output_tokens == 3
        assert total.cache_read_input_tokens == 7

    def test_total_prompt_tokens_includes_cached_reads(self):
        # input_tokens alone is the uncached remainder; reading it as the prompt
        # size understates a cached run.
        usage = Usage(input_tokens=100, cache_read_input_tokens=900)
        assert usage.total_prompt_tokens == 1000

    def test_from_response_tolerates_missing_cache_fields(self):
        class Bare:
            input_tokens = 10
            output_tokens = 2

        usage = Usage.from_response(Bare())
        assert usage.cache_read_input_tokens == 0


class TestEstimateCost:
    def test_uncached_cost_is_rate_times_tokens(self):
        rate_in, rate_out = PRICING["claude-opus-4-8"]
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        assert estimate_cost("claude-opus-4-8", usage) == pytest.approx(rate_in + rate_out)

    def test_cache_reads_are_cheaper_than_fresh_input(self):
        fresh = estimate_cost("claude-opus-4-8", Usage(input_tokens=1_000_000))
        cached = estimate_cost("claude-opus-4-8", Usage(cache_read_input_tokens=1_000_000))
        assert cached == pytest.approx(fresh * CACHE_READ_MULTIPLIER)

    def test_cache_writes_cost_more_than_fresh_input(self):
        fresh = estimate_cost("claude-opus-4-8", Usage(input_tokens=1_000_000))
        written = estimate_cost(
            "claude-opus-4-8", Usage(cache_creation_input_tokens=1_000_000)
        )
        assert written > fresh

    def test_batch_halves_the_bill(self):
        usage = Usage(input_tokens=1000, output_tokens=500)
        assert estimate_cost("claude-opus-4-8", usage, batch=True) == pytest.approx(
            estimate_cost("claude-opus-4-8", usage) / 2
        )

    def test_unknown_model_raises_rather_than_guessing(self):
        with pytest.raises(KeyError):
            estimate_cost("claude-not-a-model", Usage(input_tokens=1))
