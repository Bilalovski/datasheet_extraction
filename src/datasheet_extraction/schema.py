"""The extraction target: a sensor's specifications as structured data.

Every field is optional, and that is load-bearing rather than lazy typing. A
datasheet that never states a value should produce ``None`` for that field, not a
plausible-looking number. The evaluator scores those absences, so the schema has
to express "not stated" for every field.

The field set is a union across the sensor families in the corpus (ranging,
environmental, magnetic), so most fields are null for any given part — an
ultrasonic sensor nulls the pressure fields, a barometer nulls the ranging ones.
That cross-family null structure is what the hallucination metric feeds on.

Field *descriptions* carry the labelling conventions verbatim, because the model
reads them (they are sent as the extraction tool's parameters) and must be judged
by the same rules the gold labels were written under. Where a convention was a
real judgement call, it is stated in the description.
"""

from __future__ import annotations

from typing import Any, Literal, get_args

from pydantic import BaseModel, Field

SensorType = Literal[
    "lidar", "tof", "ultrasonic", "temperature", "pressure",
    "magnetometer", "environmental", "other",
]


class SensorSpec(BaseModel):
    """Specifications extracted from one sensor datasheet."""

    model_config = {"extra": "forbid"}

    # --- identity ---
    part_number: str | None = Field(
        default=None,
        description=(
            "Manufacturer's part number, exactly as printed (e.g. 'VL53L1X'). "
            "Null if not stated."
        ),
    )
    manufacturer: str | None = Field(
        default=None,
        description=(
            "Company that makes the sensor (e.g. 'STMicroelectronics'). Null if the "
            "datasheet does not name one — do not infer it from the part number."
        ),
    )
    sensor_type: SensorType | None = Field(
        default=None,
        description="What kind of sensor this is. Null if it cannot be determined.",
    )

    # --- electrical / interface ---
    interface: str | None = Field(
        default=None,
        description=(
            "Primary digital data interface (e.g. 'I2C', 'SPI', 'UART'). Join two "
            "co-equal buses as 'I2C/SPI'. Null if the device has no named digital bus "
            "(e.g. a raw trigger/echo pulse interface)."
        ),
    )
    supply_voltage_min_v: float | None = Field(
        default=None,
        description=(
            "Minimum of the main supply voltage you provide to the device (the VDD "
            "you plug in), in volts. If the datasheet gives a single nominal supply, "
            "use it for both min and max. Null if not stated."
        ),
    )
    supply_voltage_max_v: float | None = Field(
        default=None,
        description="Maximum of the main supply voltage, in volts. Null if not stated.",
    )
    active_current_ma: float | None = Field(
        default=None,
        description=(
            "Typical active/ranging-mode supply current, in milliamps. Null if only "
            "given as a family of mode/rate-dependent values with no single typical."
        ),
    )

    # --- operating environment ---
    operating_temp_min_c: float | None = Field(
        default=None,
        description=(
            "Minimum operating (ambient) temperature in degrees Celsius. "
            "Null if not stated."
        ),
    )
    operating_temp_max_c: float | None = Field(
        default=None,
        description=(
            "Maximum operating (ambient) temperature in degrees Celsius. "
            "Null if not stated."
        ),
    )

    # --- ranging (lidar / tof / ultrasonic) ---
    min_range_m: float | None = Field(
        default=None,
        description="Minimum measurable distance in metres. Null if not stated.",
    )
    max_range_m: float | None = Field(
        default=None,
        description="Maximum measurable distance in metres. Null if not stated.",
    )
    range_resolution_m: float | None = Field(
        default=None,
        description="Distance resolution in metres. Null if not stated.",
    )
    field_of_view_deg: float | None = Field(
        default=None,
        description=(
            "Full field of view / beam angle in degrees. These sensors have a "
            "symmetric cone, so this is a single angle, not separate azimuth and "
            "elevation. Null if not stated."
        ),
    )
    max_output_rate_hz: float | None = Field(
        default=None,
        description=(
            "Maximum measurement/ranging output rate in Hz. Null if not stated, or "
            "only given under a narrow condition (e.g. 'short range only')."
        ),
    )

    # --- point-measurement sensors (temperature / pressure / magnetometer) ---
    temperature_accuracy_c: float | None = Field(
        default=None,
        description=(
            "Temperature measurement accuracy in degrees Celsius, worst case over the "
            "stated range (the '±X max' figure, not the best-case headline). Null if "
            "not stated."
        ),
    )
    pressure_min_hpa: float | None = Field(
        default=None,
        description="Minimum measurable pressure in hPa. Null if not stated.",
    )
    pressure_max_hpa: float | None = Field(
        default=None,
        description="Maximum measurable pressure in hPa. Null if not stated.",
    )
    pressure_accuracy_hpa: float | None = Field(
        default=None,
        description=(
            "Absolute pressure accuracy in hPa, worst case (use the absolute-accuracy "
            "figure, not the smaller relative-accuracy headline). Null if not stated."
        ),
    )
    mag_range_ut: float | None = Field(
        default=None,
        description=(
            "Magnetic field measurement range in microtesla (the ± magnitude). Where "
            "it differs by axis, use the maximum full-scale across axes. Null if not stated."
        ),
    )


#: Field names in the order the report renders them.
FIELDS: list[str] = list(SensorSpec.model_fields)


def _numeric_fields(model: type[BaseModel]) -> frozenset[str]:
    """Fields whose annotation admits int or float, ignoring the None arm."""
    numeric = set()
    for name, field in model.model_fields.items():
        arms = get_args(field.annotation) or (field.annotation,)
        if any(arm in (int, float) for arm in arms):
            numeric.add(name)
    return frozenset(numeric)


#: Fields compared with a relative tolerance rather than exact equality.
#: Derived from the annotations so adding a field to the schema is enough.
NUMERIC_FIELDS: frozenset[str] = _numeric_fields(SensorSpec)


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Render a Pydantic model as a schema for the extraction tool's parameters.

    Every object gets ``additionalProperties: false`` and every property is listed
    in ``required`` — optionality is expressed as a nullable type, which is why
    every field is ``X | None``. DeepSeek does not enforce this server-side, so the
    schema is a guide to the model and Pydantic does the real enforcement on the
    way back in.
    """
    schema = model.model_json_schema()

    def tighten(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
                if "properties" in node:
                    node["required"] = list(node["properties"])
            for value in node.values():
                tighten(value)
        elif isinstance(node, list):
            for item in node:
                tighten(item)

    tighten(schema)
    return schema
