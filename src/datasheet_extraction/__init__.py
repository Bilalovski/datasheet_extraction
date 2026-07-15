"""Structured extraction of sensor specifications from datasheets, with an eval harness."""

from .compare import Outcome, compare_field
from .evaluate import FieldScore, Report, evaluate
from .schema import FIELDS, SensorSpec

__all__ = [
    "FIELDS",
    "FieldScore",
    "Outcome",
    "Report",
    "SensorSpec",
    "compare_field",
    "evaluate",
]
