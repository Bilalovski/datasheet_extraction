"""The extraction target: a sensor's specifications as structured data.

Every field is optional, and that is load-bearing rather than lazy typing. A
datasheet that never states its elevation field of view should produce
``elevation_fov_deg=None``, not a plausible-looking number. The evaluator scores
those absences (see :mod:`datasheet_extraction.evaluate`), so the schema has to
be able to express "not stated" for every field.
"""

from __future__ import annotations

from typing import Any, Literal, get_args

from pydantic import BaseModel, Field

SensorType = Literal["radar", "lidar", "camera", "imu", "ultrasonic", "other"]


class SensorSpec(BaseModel):
    """Specifications extracted from one sensor datasheet."""

    model_config = {"extra": "forbid"}

    part_number: str | None = Field(
        default=None,
        description=(
            "Manufacturer's part number, exactly as printed (e.g. 'AWR1843'). "
            "Null if not stated."
        ),
    )
    manufacturer: str | None = Field(
        default=None,
        description="Company that makes the sensor (e.g. 'Texas Instruments'). Null if not stated.",
    )
    sensor_type: SensorType | None = Field(
        default=None,
        description="What kind of sensor this is. Null if it cannot be determined.",
    )

    center_frequency_ghz: float | None = Field(
        default=None,
        description=(
            "Operating centre frequency in GHz. If the datasheet gives a band "
            "(e.g. '76-81 GHz'), report the midpoint. Null if not stated."
        ),
    )
    bandwidth_mhz: float | None = Field(
        default=None,
        description="Usable sweep bandwidth in MHz. Null if not stated.",
    )
    max_range_m: float | None = Field(
        default=None,
        description="Maximum detection range in metres. Null if not stated.",
    )
    range_resolution_m: float | None = Field(
        default=None,
        description="Range resolution in metres. Null if not stated.",
    )
    azimuth_fov_deg: float | None = Field(
        default=None,
        description=(
            "Total azimuth field of view in degrees. A '±60°' spec is a 120° "
            "total FoV. Null if not stated."
        ),
    )
    elevation_fov_deg: float | None = Field(
        default=None,
        description=(
            "Total elevation field of view in degrees. A '±15°' spec is a 30° "
            "total FoV. Null if not stated."
        ),
    )
    tx_channels: int | None = Field(
        default=None,
        description="Number of transmit channels/antennas. Null if not stated.",
    )
    rx_channels: int | None = Field(
        default=None,
        description="Number of receive channels/antennas. Null if not stated.",
    )
    supply_voltage_v: float | None = Field(
        default=None,
        description="Nominal supply voltage in volts. Null if not stated.",
    )
    power_consumption_w: float | None = Field(
        default=None,
        description="Typical power consumption in watts. Null if not stated.",
    )
    interface: str | None = Field(
        default=None,
        description=(
            "Primary data interface (e.g. 'CAN', 'Ethernet', 'SPI', 'LVDS'). "
            "Null if not stated."
        ),
    )
    operating_temp_min_c: float | None = Field(
        default=None,
        description="Minimum operating temperature in degrees Celsius. Null if not stated.",
    )
    operating_temp_max_c: float | None = Field(
        default=None,
        description="Maximum operating temperature in degrees Celsius. Null if not stated.",
    )


def _numeric_fields(model: type[BaseModel]) -> frozenset[str]:
    """Fields whose annotation admits int or float, ignoring the None arm."""
    numeric = set()
    for name, field in model.model_fields.items():
        arms = get_args(field.annotation) or (field.annotation,)
        if any(arm in (int, float) for arm in arms):
            numeric.add(name)
    return frozenset(numeric)


#: Field names in the order the report renders them.
FIELDS: list[str] = list(SensorSpec.model_fields)

#: Fields compared with a relative tolerance rather than exact equality.
#: Derived from the annotations so adding a field to the schema is enough.
NUMERIC_FIELDS: frozenset[str] = _numeric_fields(SensorSpec)


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Render a Pydantic model as a schema the structured-outputs API accepts.

    Two things differ from Pydantic's default output. Every object needs
    ``additionalProperties: false``, and every property must appear in
    ``required`` — optional fields are expressed as a null-able type rather than
    an absent key, which is why every field here is ``X | None``.

    Needed only for the Batch API, where responses go through
    ``output_config.format`` instead of ``messages.parse()``.
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
