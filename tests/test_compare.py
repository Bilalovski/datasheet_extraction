from datasheet_extraction.compare import (
    Outcome,
    compare_field,
    normalise_text,
    numbers_match,
)


class TestNumbersMatch:
    def test_exact(self):
        assert numbers_match(77.0, 77.0)

    def test_within_default_tolerance(self):
        # 1% of 250 is 2.5, so 251 is a rounding difference, not an error.
        assert numbers_match(250.0, 251.0)

    def test_outside_default_tolerance(self):
        assert not numbers_match(250.0, 260.0)

    def test_tolerance_is_relative_to_gold(self):
        # 0.0004 is 1% of 0.04 — the same absolute slack would be absurd at 250 m.
        assert numbers_match(0.04, 0.0404)
        assert not numbers_match(0.04, 0.05)

    def test_zero_gold_uses_absolute_comparison(self):
        # Everything is within 1% of zero, so relative tolerance is meaningless.
        assert numbers_match(0.0, 0.0)
        assert not numbers_match(0.0, 0.5)

    def test_negative_values(self):
        assert numbers_match(-40.0, -40.2)
        assert not numbers_match(-40.0, -45.0)


class TestNormaliseText:
    def test_casefold_and_strip(self):
        assert normalise_text("  Ethernet  ") == "ethernet"

    def test_collapses_internal_whitespace(self):
        assert normalise_text("ACME   Microwave\nSystems") == "acme microwave systems"

    def test_drops_trailing_punctuation(self):
        assert normalise_text("LVDS.") == normalise_text("LVDS")


class TestCompareField:
    def test_both_null_is_a_true_negative(self):
        assert compare_field("field_of_view_deg", None, None) is Outcome.TRUE_NEGATIVE

    def test_invented_value_is_a_hallucination(self):
        assert compare_field("field_of_view_deg", None, 30.0) is Outcome.HALLUCINATION

    def test_missed_value_is_a_false_negative(self):
        assert compare_field("field_of_view_deg", 30.0, None) is Outcome.FALSE_NEGATIVE

    def test_agreeing_values_are_a_true_positive(self):
        assert compare_field("max_range_m", 78.5, 78.5) is Outcome.TRUE_POSITIVE

    def test_disagreeing_values_are_a_wrong_value(self):
        assert compare_field("max_range_m", 78.5, 77.0) is Outcome.WRONG_VALUE

    def test_numeric_field_tolerates_int_float_mismatch(self):
        # A field typed loosely still matches 15 vs 15.0.
        assert compare_field("field_of_view_deg", 15, 15.0) is Outcome.TRUE_POSITIVE

    def test_string_field_compares_normalised(self):
        outcome = compare_field(
            "manufacturer", "ACME Microwave Systems", "acme  microwave systems"
        )
        assert outcome is Outcome.TRUE_POSITIVE

    def test_string_field_does_not_match_a_different_name(self):
        outcome = compare_field("manufacturer", "Northgate Sensing", "Northgate Systems")
        assert outcome is Outcome.WRONG_VALUE

    def test_enum_field_compares_exactly(self):
        assert compare_field("sensor_type", "radar", "radar") is Outcome.TRUE_POSITIVE
        assert compare_field("sensor_type", "radar", "lidar") is Outcome.WRONG_VALUE

    def test_unparseable_number_is_a_wrong_value_not_a_crash(self):
        assert compare_field("max_range_m", 250.0, "far") is Outcome.WRONG_VALUE
