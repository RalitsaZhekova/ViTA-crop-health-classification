"""Validated coordinate commands exchanged between ground and payload."""

from __future__ import annotations

import re
from datetime import date
from math import isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ACQUISITION_SCHEMA_VERSION = "1.0"
EARTH_ENGINE_PROVIDER = "earth_engine"
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
MAX_SEARCH_INTERVAL_DAYS = 92


class EarthEngineSourceCommand(BaseModel):
    """Fixed-provider search parameters supplied by the ground mission client."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["earth_engine"]
    bbox_wgs84: tuple[float, float, float, float]
    start_date: date
    end_date: date
    selection_policy: Literal["target_cloud_range", "least_cloudy"]
    target_cloud_min_percent: float = Field(default=15.0, ge=5.0, le=50.0)
    target_cloud_max_percent: float = Field(default=35.0, ge=5.0, le=50.0)
    target_cloud_ideal_percent: float = Field(default=25.0, ge=0.0, le=100.0)

    @field_validator("bbox_wgs84")
    @classmethod
    def validate_bbox(
        cls, value: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        west, south, east, north = value
        if not all(isfinite(coordinate) for coordinate in value):
            raise ValueError("bbox coordinates must be finite")
        if not (-180 <= west <= 180 and -180 <= east <= 180):
            raise ValueError("bbox longitudes must be within -180..180")
        if not (-90 <= south <= 90 and -90 <= north <= 90):
            raise ValueError("bbox latitudes must be within -90..90")
        if west >= east:
            raise ValueError("bbox must not cross the antimeridian and west must be less than east")
        if south >= north:
            raise ValueError("bbox south must be less than north")
        return value

    @model_validator(mode="after")
    def validate_interval_and_cloud_target(self) -> EarthEngineSourceCommand:
        if self.start_date > self.end_date:
            raise ValueError("start_date must be on or before end_date")
        inclusive_days = (self.end_date - self.start_date).days + 1
        if inclusive_days > MAX_SEARCH_INTERVAL_DAYS:
            raise ValueError(f"search interval must not exceed {MAX_SEARCH_INTERVAL_DAYS} days")
        if self.target_cloud_min_percent >= self.target_cloud_max_percent:
            raise ValueError("target cloud minimum must be less than maximum")
        if not (
            self.target_cloud_min_percent
            <= self.target_cloud_ideal_percent
            <= self.target_cloud_max_percent
        ):
            raise ValueError("target cloud ideal must be between minimum and maximum")
        return self


class PayloadAcquisitionCommand(BaseModel):
    """Schema 1.0 payload acquisition job command."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"]
    job_id: str
    region_id: str
    source: EarthEngineSourceCommand

    @field_validator("job_id", "region_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        if not IDENTIFIER_PATTERN.fullmatch(value):
            raise ValueError(
                "identifier must be 1-80 characters using only letters, numbers, "
                "underscore or hyphen"
            )
        return value
