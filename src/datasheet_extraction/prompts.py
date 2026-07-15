"""System prompts for extraction, written as variants so they can be ablated.

Each variant is a hypothesis about what actually drives the score. ``minimal``
is the floor — the schema's field descriptions carry all the instruction.
``strict`` adds explicit abstention and unit rules. ``strict_with_examples``
adds worked cases for the two conversions the model gets wrong most often
(half-angle FoV, frequency bands). Which one wins, and whether the extra tokens
pay for themselves, is what ``ablate`` measures rather than assumes.
"""

from __future__ import annotations

MINIMAL = """\
Extract the sensor's specifications from the datasheet text into the given schema.\
"""

STRICT = """\
You extract sensor specifications from datasheet text into a structured schema.

Rules:
- Report only what the document states. If a field is not stated, return null \
for it. Do not infer a value from a similar product, a typical value for the \
part's category, or your own knowledge of the device.
- Convert to the unit named in each field. A value in kHz goes into an MHz field \
divided by 1000.
- Fields of view are totals. A datasheet quoting "±60°" is a 120° total field of \
view; one quoting "120° (±60°)" is also 120°.
- For a frequency band, report the midpoint: "76-81 GHz" is 78.5.
- Where a value has minimum, typical, and maximum columns, take the typical one.
- Copy part numbers and manufacturer names exactly as printed, without expanding \
abbreviations or adding suffixes like "Inc.".

A null is a correct answer when the document is silent. A plausible guess is not.\
"""

STRICT_WITH_EXAMPLES = (
    STRICT
    + """

Worked examples:

  "Operating band: 76 GHz to 81 GHz"      -> center_frequency_ghz = 78.5
  "Frequency: 77 GHz"                     -> center_frequency_ghz = 77.0
  "Azimuth coverage: ±45°"                -> azimuth_fov_deg = 90.0
  "Horizontal FoV: 100 degrees"           -> azimuth_fov_deg = 100.0
  "Range resolution: 4 cm"                -> range_resolution_m = 0.04
  "Supply: 3.3 V (typ), 3.0 V (min)"      -> supply_voltage_v = 3.3
  (elevation not mentioned anywhere)      -> elevation_fov_deg = null\
"""
)

VARIANTS: dict[str, str] = {
    "minimal": MINIMAL,
    "strict": STRICT,
    "strict_with_examples": STRICT_WITH_EXAMPLES,
}

DEFAULT_VARIANT = "strict_with_examples"


def get(variant: str) -> str:
    """Look up a prompt variant by name."""
    try:
        return VARIANTS[variant]
    except KeyError:
        raise KeyError(
            f"unknown prompt variant {variant!r}; have: {', '.join(sorted(VARIANTS))}"
        ) from None


def user_message(document_text: str) -> str:
    """Wrap the document for the user turn.

    The document goes last, after any cacheable prefix, and is delimited so the
    model can tell datasheet prose from instructions.
    """
    return f"<datasheet>\n{document_text.strip()}\n</datasheet>"
