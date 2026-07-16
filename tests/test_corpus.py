import json
from pathlib import Path

import pytest

from datasheet_extraction.corpus import load_corpus, load_gold
from datasheet_extraction.schema import FIELDS

DEMO = Path(__file__).resolve().parents[1] / "corpus" / "demo"


class TestDemoCorpus:
    """The shipped demo corpus must stay loadable and internally consistent."""

    def test_loads(self):
        documents, gold = load_corpus(DEMO)
        assert set(documents) == set(gold)
        assert len(documents) == 4

    def test_every_document_has_text(self):
        documents, _ = load_corpus(DEMO)
        assert all(text.strip() for text in documents.values())

    def test_gold_labels_validate_against_the_schema(self):
        gold = load_gold(DEMO)
        assert gold["syn-tof"].field_of_view_deg == 20.0
        assert gold["syn-temp"].temperature_accuracy_c == 0.25  # worst-case
        assert gold["syn-baro"].pressure_max_hpa == 1260

    def test_corpus_contains_fields_that_are_genuinely_absent(self):
        # Without unstated fields there is nothing to measure hallucination on,
        # so the demo corpus is only useful if it has a decent supply of nulls.
        gold = load_gold(DEMO)
        absent = sum(
            1
            for spec in gold.values()
            for name in FIELDS
            if getattr(spec, name) is None
        )
        assert absent >= 20

    def test_ranging_fields_are_null_for_point_sensors(self):
        # A temperature sensor states no range — a model that fills one in is
        # hallucinating, not merely wrong. This cross-family null structure is
        # what the hallucination metric feeds on.
        gold = load_gold(DEMO)
        assert gold["syn-temp"].max_range_m is None
        assert gold["syn-temp"].field_of_view_deg is None


class TestCorpusValidation:
    def test_document_without_a_label_is_an_error(self, tmp_path):
        (tmp_path / "a.txt").write_text("text", encoding="utf-8")
        (tmp_path / "gold.json").write_text("{}", encoding="utf-8")

        with pytest.raises(ValueError, match="no gold label"):
            load_corpus(tmp_path)

    def test_label_without_a_document_is_an_error(self, tmp_path):
        (tmp_path / "gold.json").write_text(
            json.dumps({"ghost": {"part_number": "X"}}), encoding="utf-8"
        )

        with pytest.raises(ValueError, match="no document"):
            load_corpus(tmp_path)

    def test_invalid_label_names_the_document(self, tmp_path):
        (tmp_path / "a.txt").write_text("text", encoding="utf-8")
        (tmp_path / "gold.json").write_text(
            json.dumps({"a": {"max_range_m": "not a number"}}), encoding="utf-8"
        )

        with pytest.raises(ValueError, match="gold label for 'a'"):
            load_corpus(tmp_path)

    def test_missing_gold_file_is_an_error(self, tmp_path):
        (tmp_path / "a.txt").write_text("text", encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            load_corpus(tmp_path)
