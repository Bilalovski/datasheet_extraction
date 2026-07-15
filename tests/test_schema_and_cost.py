import pytest
from pydantic import ValidationError

from datasheet_extraction.cost import PRICING, Usage, estimate_cost, resolve
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
        assert strict_json_schema(SensorSpec)["additionalProperties"] is False

    def test_every_property_is_required(self):
        schema = strict_json_schema(SensorSpec)
        assert set(schema["required"]) == set(schema["properties"])

    def test_optional_fields_admit_null(self):
        schema = strict_json_schema(SensorSpec)
        assert {"type": "null"} in schema["properties"]["max_range_m"]["anyOf"]

    def test_descriptions_survive_into_the_schema(self):
        # The schema is how the unit rules reach the model, since it's sent as
        # the tool's parameters. Losing descriptions would silently gut the prompt.
        props = strict_json_schema(SensorSpec)["properties"]
        assert "midpoint" in props["center_frequency_ghz"]["description"]
        assert "120" in props["azimuth_fov_deg"]["description"]


class TestUsage:
    def test_addition_accumulates_every_counter(self):
        total = Usage(cache_miss_tokens=10, output_tokens=1) + Usage(
            cache_miss_tokens=5, output_tokens=2, cache_hit_tokens=7
        )
        assert total.cache_miss_tokens == 15
        assert total.output_tokens == 3
        assert total.cache_hit_tokens == 7

    def test_prompt_tokens_is_hits_plus_misses(self):
        assert Usage(cache_miss_tokens=100, cache_hit_tokens=900).prompt_tokens == 1000

    def test_cache_hit_rate(self):
        assert Usage(cache_miss_tokens=100, cache_hit_tokens=900).cache_hit_rate == 0.9
        assert Usage().cache_hit_rate == 0.0

    def test_from_response_reads_deepseek_cache_split(self):
        class DeepSeekUsage:
            prompt_tokens = 1000
            completion_tokens = 50
            prompt_cache_hit_tokens = 900
            prompt_cache_miss_tokens = 100

        usage = Usage.from_response(DeepSeekUsage())
        assert usage.cache_hit_tokens == 900
        assert usage.cache_miss_tokens == 100
        assert usage.output_tokens == 50

    def test_from_response_without_cache_fields_assumes_all_miss(self):
        # Conservative: over-state cost rather than quietly under-state it.
        class PlainOpenAIUsage:
            prompt_tokens = 1000
            completion_tokens = 50

        usage = Usage.from_response(PlainOpenAIUsage())
        assert usage.cache_miss_tokens == 1000
        assert usage.cache_hit_tokens == 0


class TestEstimateCost:
    def test_cost_is_rate_times_tokens(self):
        hit_rate, miss_rate, out_rate = PRICING["deepseek-v4-flash"]
        usage = Usage(cache_miss_tokens=1_000_000, output_tokens=1_000_000)
        assert estimate_cost("deepseek-v4-flash", usage) == pytest.approx(miss_rate + out_rate)

    def test_cache_hits_are_far_cheaper_than_misses(self):
        miss = estimate_cost("deepseek-v4-flash", Usage(cache_miss_tokens=1_000_000))
        hit = estimate_cost("deepseek-v4-flash", Usage(cache_hit_tokens=1_000_000))
        assert hit < miss / 10  # published rates put a hit at ~2% of a miss

    def test_pro_costs_more_than_flash_for_the_same_work(self):
        usage = Usage(cache_miss_tokens=10_000, output_tokens=500)
        assert estimate_cost("deepseek-v4-pro", usage) > estimate_cost("deepseek-v4-flash", usage)

    def test_unknown_model_raises_rather_than_guessing(self):
        with pytest.raises(KeyError):
            estimate_cost("gpt-9", Usage(cache_miss_tokens=1))


class TestDeprecatedAliases:
    def test_aliases_resolve_for_pricing(self):
        # deepseek-chat / deepseek-reasoner retire 2026-07-24; keep them priceable
        # so an old id doesn't crash, while the canonical ids are the default.
        assert resolve("deepseek-chat") == "deepseek-v4-flash"
        assert resolve("deepseek-reasoner") == "deepseek-v4-flash"

    def test_canonical_ids_pass_through(self):
        assert resolve("deepseek-v4-pro") == "deepseek-v4-pro"

    def test_deprecated_alias_still_prices(self):
        assert estimate_cost("deepseek-chat", Usage(cache_miss_tokens=1_000_000)) > 0
