import pytest

from datasheet_extraction.evaluate import evaluate
from datasheet_extraction.schema import FIELDS, SensorSpec


def spec(**fields) -> SensorSpec:
    return SensorSpec(**fields)


class TestPerfectAndEmpty:
    def test_perfect_prediction_scores_one(self):
        gold = {"a": spec(part_number="RDX-7700", max_range_m=250.0)}
        report = evaluate(gold, {"a": spec(part_number="RDX-7700", max_range_m=250.0)})

        assert report.total.f1 == 1.0
        assert report.total.hallucination_rate == 0.0

    def test_abstaining_everywhere_gold_is_null_is_free(self):
        # A document where nothing is stated and the model says nothing should
        # not be scored as 14 correct answers — there is nothing to be right about.
        gold = {"a": spec()}
        report = evaluate(gold, {"a": spec()})

        assert report.total.support == 0
        assert report.total.true_positives == 0
        assert report.total.absent == len(FIELDS)


class TestWrongValueAccounting:
    def test_wrong_value_counts_as_both_a_miss_and_a_spurious_answer(self):
        gold = {"a": spec(max_range_m=250.0)}
        report = evaluate(gold, {"a": spec(max_range_m=100.0)})
        score = report.fields["max_range_m"]

        assert score.false_positives == 1
        assert score.false_negatives == 1
        assert score.precision == 0.0
        assert score.recall == 0.0

    def test_confidently_wrong_scores_worse_than_abstaining(self):
        # The point of double-counting: a wrong spec propagates silently, a null
        # does not, so the metric must not rank them equally.
        gold = {"a": spec(max_range_m=250.0), "b": spec(max_range_m=100.0)}

        abstained = evaluate(gold, {"a": spec(max_range_m=250.0), "b": spec()})
        wrong = evaluate(gold, {"a": spec(max_range_m=250.0), "b": spec(max_range_m=999.0)})

        assert wrong.fields["max_range_m"].f1 < abstained.fields["max_range_m"].f1


class TestHallucinationRate:
    def test_measured_only_over_fields_gold_leaves_null(self):
        gold = {"a": spec(part_number="X")}  # 1 stated, the rest null
        report = evaluate(gold, {"a": spec(part_number="X", max_range_m=250.0)})

        # One invented value out of every field the datasheet doesn't state.
        assert report.fields["max_range_m"].hallucination_rate == 1.0
        assert report.fields["part_number"].hallucination_rate == 0.0
        assert report.total.hallucination_rate == pytest.approx(1 / (len(FIELDS) - 1))

    def test_perfect_precision_can_coexist_with_hallucination(self):
        # This is exactly why the metric exists: precision on stated fields is
        # 1.0 here, and the model still invented a spec that isn't in the doc.
        gold = {"a": spec(part_number="X")}
        report = evaluate(gold, {"a": spec(part_number="X", supply_voltage_v=3.3)})

        assert report.fields["part_number"].precision == 1.0
        assert report.fields["supply_voltage_v"].hallucination_rate == 1.0


class TestFailedExtractions:
    def test_missing_document_is_a_miss_on_every_stated_field(self):
        gold = {"a": spec(part_number="X", max_range_m=250.0)}
        report = evaluate(gold, {})  # extraction crashed

        assert report.total.false_negatives == 2
        assert report.total.recall == 0.0

    def test_missing_document_is_not_credited_for_abstaining(self):
        # An extraction that never ran must not score true negatives on the
        # fields it never had a chance to invent.
        gold = {"a": spec(part_number="X")}
        report = evaluate(gold, {})

        assert report.total.counts.get("hallucination", 0) == 0
        assert report.fields["part_number"].false_negatives == 1


class TestReportRows:
    def test_rows_cover_every_field_plus_a_total(self):
        gold = {"a": spec(part_number="X")}
        rows = evaluate(gold, {"a": spec(part_number="X")}).as_rows()

        assert len(rows) == len(FIELDS) + 1
        assert rows[-1]["field"] == "ALL"
