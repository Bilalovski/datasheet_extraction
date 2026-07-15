"""Deciding whether one extracted field matches the gold label.

The comparison is per-field and type-aware: numbers match within a relative
tolerance, strings match after normalisation, enums match exactly. The part that
matters for the metrics is the five-way outcome — in particular that "the
datasheet doesn't say" is a real answer a model can get right or wrong, not a
gap in the data.
"""

from __future__ import annotations

import re
from enum import Enum

from .schema import NUMERIC_FIELDS

#: Numbers match within this relative tolerance. A datasheet saying "77 GHz"
#: and an extraction saying "77.0" are the same answer; 76.9 vs 77.0 is a
#: rounding difference, not an error worth flagging.
DEFAULT_REL_TOLERANCE = 0.01

#: Below this magnitude, relative tolerance is meaningless (everything is
#: within 1% of zero), so fall back to an absolute comparison.
ABS_TOLERANCE_FLOOR = 1e-9


class Outcome(str, Enum):
    """What happened to one field on one document."""

    TRUE_POSITIVE = "true_positive"
    """Gold has a value, the model found it, they agree."""

    TRUE_NEGATIVE = "true_negative"
    """The datasheet doesn't state it and the model correctly said nothing."""

    FALSE_NEGATIVE = "false_negative"
    """The datasheet states it and the model returned null. A miss."""

    HALLUCINATION = "hallucination"
    """The datasheet does not state it and the model returned a value anyway."""

    WRONG_VALUE = "wrong_value"
    """Both have a value and they disagree."""


_WHITESPACE = re.compile(r"\s+")
_TRAILING_NOISE = re.compile(r"[.,;:]+$")


def normalise_text(value: str) -> str:
    """Casefold, collapse internal whitespace, drop trailing punctuation."""
    return _TRAILING_NOISE.sub("", _WHITESPACE.sub(" ", value.strip())).casefold()


def numbers_match(
    gold: float, pred: float, rel_tolerance: float = DEFAULT_REL_TOLERANCE
) -> bool:
    """True when two numbers agree within ``rel_tolerance`` of the gold value."""
    scale = abs(gold)
    if scale < ABS_TOLERANCE_FLOOR:
        return abs(pred - gold) < ABS_TOLERANCE_FLOOR
    return abs(pred - gold) / scale <= rel_tolerance


def values_match(
    field: str, gold: object, pred: object, rel_tolerance: float = DEFAULT_REL_TOLERANCE
) -> bool:
    """True when a non-null prediction agrees with a non-null gold value."""
    if field in NUMERIC_FIELDS:
        try:
            return numbers_match(float(gold), float(pred), rel_tolerance)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
    if isinstance(gold, str) and isinstance(pred, str):
        return normalise_text(gold) == normalise_text(pred)
    return gold == pred


def compare_field(
    field: str, gold: object, pred: object, rel_tolerance: float = DEFAULT_REL_TOLERANCE
) -> Outcome:
    """Classify one field of one document into an :class:`Outcome`."""
    gold_missing = gold is None
    pred_missing = pred is None

    if gold_missing and pred_missing:
        return Outcome.TRUE_NEGATIVE
    if gold_missing:
        return Outcome.HALLUCINATION
    if pred_missing:
        return Outcome.FALSE_NEGATIVE
    if values_match(field, gold, pred, rel_tolerance):
        return Outcome.TRUE_POSITIVE
    return Outcome.WRONG_VALUE
