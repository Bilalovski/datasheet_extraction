"""Turning per-field outcomes into precision, recall, F1, and hallucination rate.

The one judgement call worth knowing about: a ``WRONG_VALUE`` counts as both a
false positive and a false negative. The model both failed to produce the right
answer (a miss) and asserted a wrong one (a spurious answer), and a metric that
charged it only once would rank "confidently wrong" the same as "said nothing" —
which is backwards, because a wrong spec silently propagates into whatever reads
it while a null is visibly missing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from dataclasses import field as dataclass_field

from .compare import Outcome, compare_field
from .schema import FIELDS, SensorSpec


@dataclass
class FieldScore:
    """Counts and metrics for one field, aggregated over documents."""

    name: str
    counts: Counter[Outcome] = dataclass_field(default_factory=Counter)

    @property
    def true_positives(self) -> int:
        return self.counts[Outcome.TRUE_POSITIVE]

    @property
    def false_positives(self) -> int:
        """Spurious answers: invented values plus wrong ones."""
        return self.counts[Outcome.HALLUCINATION] + self.counts[Outcome.WRONG_VALUE]

    @property
    def false_negatives(self) -> int:
        """Missed answers: nulls where a value existed, plus wrong ones."""
        return self.counts[Outcome.FALSE_NEGATIVE] + self.counts[Outcome.WRONG_VALUE]

    @property
    def support(self) -> int:
        """Documents whose gold label states this field."""
        return (
            self.counts[Outcome.TRUE_POSITIVE]
            + self.counts[Outcome.FALSE_NEGATIVE]
            + self.counts[Outcome.WRONG_VALUE]
        )

    @property
    def absent(self) -> int:
        """Documents whose gold label leaves this field unstated."""
        return self.counts[Outcome.TRUE_NEGATIVE] + self.counts[Outcome.HALLUCINATION]

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 1.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def hallucination_rate(self) -> float:
        """Share of "not stated" fields where the model invented a value.

        Scored only over the fields the gold label leaves null — the chances the
        model had to hallucinate. Precision alone hides this: a model that
        answers every field correctly *and* invents three specs that aren't in
        the document is the one you must not ship, and this is the number that
        says so.
        """
        return self.counts[Outcome.HALLUCINATION] / self.absent if self.absent else 0.0


@dataclass
class Report:
    """Scores for every field, plus the micro-average across all of them."""

    fields: dict[str, FieldScore]

    @property
    def total(self) -> FieldScore:
        """Micro-average: every (document, field) pair weighted equally."""
        combined = FieldScore(name="ALL")
        for score in self.fields.values():
            combined.counts.update(score.counts)
        return combined

    def as_rows(self) -> list[dict[str, object]]:
        """Per-field rows plus a trailing total, ready to render or serialise."""
        rows = [
            {
                "field": score.name,
                "support": score.support,
                "precision": round(score.precision, 4),
                "recall": round(score.recall, 4),
                "f1": round(score.f1, 4),
                "hallucination_rate": round(score.hallucination_rate, 4),
            }
            for score in self.fields.values()
        ]
        total = self.total
        rows.append(
            {
                "field": "ALL",
                "support": total.support,
                "precision": round(total.precision, 4),
                "recall": round(total.recall, 4),
                "f1": round(total.f1, 4),
                "hallucination_rate": round(total.hallucination_rate, 4),
            }
        )
        return rows


def evaluate(
    gold: dict[str, SensorSpec], predicted: dict[str, SensorSpec]
) -> Report:
    """Score predictions against gold labels, keyed by document id.

    A document in ``gold`` with no entry in ``predicted`` scores as a miss on
    every field rather than being skipped — an extraction that crashed is a
    failure, not an absence of evidence.
    """
    scores = {name: FieldScore(name=name) for name in FIELDS}

    for doc_id, gold_spec in gold.items():
        pred_spec = predicted.get(doc_id)
        for name in FIELDS:
            gold_value = getattr(gold_spec, name)
            pred_value = getattr(pred_spec, name) if pred_spec is not None else None
            scores[name].counts[compare_field(name, gold_value, pred_value)] += 1

    return Report(fields=scores)
