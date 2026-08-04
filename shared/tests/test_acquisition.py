from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from prithvi_shared.acquisition import PayloadAcquisitionCommand
from pydantic import ValidationError


def _command(**source_overrides: object) -> dict[str, object]:
    source: dict[str, object] = {
        "provider": "earth_engine",
        "bbox_wgs84": [23.10, 42.50, 23.15, 42.55],
        "start_date": "2026-07-01",
        "end_date": "2026-07-29",
        "selection_policy": "target_cloud_range",
        "target_cloud_min_percent": 15,
        "target_cloud_max_percent": 35,
        "target_cloud_ideal_percent": 25,
    }
    source.update(source_overrides)
    return {
        "schema_version": "1.0",
        "job_id": "field_42_20260715_ab12cd34",
        "region_id": "field_42",
        "source": source,
    }


def test_target_cloud_command_matches_schema_contract() -> None:
    command = PayloadAcquisitionCommand.model_validate(_command())

    assert command.source.bbox_wgs84 == (23.10, 42.50, 23.15, 42.55)
    assert command.source.start_date == date(2026, 7, 1)
    assert command.source.end_date == date(2026, 7, 29)
    schema = json.loads(
        (
            Path(__file__).parents[2] / "shared/schemas/payload_acquisition_command.schema.json"
        ).read_text(encoding="utf-8")
    )
    assert schema["additionalProperties"] is False
    assert schema["properties"]["source"]["$ref"] == "#/$defs/earthEngineSource"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_cloud_min_percent", 4.99),
        ("target_cloud_max_percent", 50.01),
        ("target_cloud_min_percent", 35),
        ("target_cloud_ideal_percent", 14.99),
        ("target_cloud_ideal_percent", 35.01),
    ],
)
def test_target_cloud_bounds_are_rejected(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        PayloadAcquisitionCommand.model_validate(_command(**{field: value}))


@pytest.mark.parametrize(
    "bbox",
    [
        [23.15, 42.50, 23.10, 42.55],
        [23.10, 42.55, 23.15, 42.50],
        [-181, 42.50, 23.15, 42.55],
        [23.10, -91, 23.15, 42.55],
        [float("nan"), 42.50, 23.15, 42.55],
    ],
)
def test_invalid_bbox_is_rejected(bbox: list[float]) -> None:
    with pytest.raises(ValidationError):
        PayloadAcquisitionCommand.model_validate(_command(bbox_wgs84=bbox))


def test_date_interval_is_inclusive_and_limited_to_92_days() -> None:
    accepted = PayloadAcquisitionCommand.model_validate(
        _command(start_date="2026-01-01", end_date="2026-04-02")
    )
    assert (accepted.source.end_date - accepted.source.start_date).days + 1 == 92

    with pytest.raises(ValidationError, match="92 days"):
        PayloadAcquisitionCommand.model_validate(
            _command(start_date="2026-01-01", end_date="2026-04-03")
        )
    with pytest.raises(ValidationError, match="on or before"):
        PayloadAcquisitionCommand.model_validate(
            _command(start_date="2026-07-02", end_date="2026-07-01")
        )


def test_unknown_fields_and_unsafe_identifiers_are_rejected() -> None:
    value = _command()
    value["command"] = "python"
    with pytest.raises(ValidationError):
        PayloadAcquisitionCommand.model_validate(value)

    value = _command()
    value["job_id"] = "../unsafe"
    with pytest.raises(ValidationError):
        PayloadAcquisitionCommand.model_validate(value)


def test_least_cloudy_uses_fixed_safe_defaults() -> None:
    value = _command(selection_policy="least_cloudy")
    source = value["source"]
    assert isinstance(source, dict)
    for field in (
        "target_cloud_min_percent",
        "target_cloud_max_percent",
        "target_cloud_ideal_percent",
    ):
        source.pop(field)

    command = PayloadAcquisitionCommand.model_validate(value)

    assert command.source.selection_policy == "least_cloudy"
    assert command.source.target_cloud_min_percent == 15
    assert command.source.target_cloud_max_percent == 35
    assert command.source.target_cloud_ideal_percent == 25
