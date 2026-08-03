"""Bounded-memory validation for a preprocessed multispectral scene."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine

SCHEMA_VERSION = "0.1-draft"
SUPPORTED_SENSORS = ("sentinel-2", "balkan-1")

_BAND_ALIASES = {
    "BLUE": {"BLUE", "B02", "B2"},
    "GREEN": {"GREEN", "B03", "B3"},
    "RED": {"RED", "B04", "B4"},
    "NIR_BROAD": {"NIR", "NIRBROAD", "B08", "B8"},
    "NIR_NARROW": {"NIRNARROW", "B8A"},
    "PANCHROMATIC": {"PAN", "PANCHROMATIC"},
}


def _normalise_description(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def _logical_role(description: str | None) -> str | None:
    normalised = _normalise_description(description)
    if normalised is None:
        return None
    for role, aliases in _BAND_ALIASES.items():
        if normalised in aliases:
            return role
    return None


def _validate_acquired_at(value: str | None) -> str | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("acquired_at must include a UTC offset or trailing Z")
    return parsed.isoformat()


def _validate_scene_id(value: str) -> str:
    if not value or value in {".", ".."} or any(separator in value for separator in ("/", "\\")):
        raise ValueError("scene_id must be a non-empty filename-safe identifier")
    return value


def _sample_band(dataset: rasterio.DatasetReader, index: int) -> dict[str, Any]:
    height = min(dataset.height, 256)
    width = min(dataset.width, 256)
    sample = dataset.read(
        index,
        out_shape=(height, width),
        masked=True,
        resampling=Resampling.nearest,
    )
    finite = np.asarray(sample.compressed())
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"valid_pixels": 0, "minimum": None, "maximum": None, "mean": None}
    return {
        "valid_pixels": int(finite.size),
        "minimum": float(finite.min()),
        "maximum": float(finite.max()),
        "mean": float(finite.mean()),
    }


def _radiometry(
    sensor: str,
    stats: list[dict[str, Any]],
) -> dict[str, Any]:
    minima = [item["minimum"] for item in stats if item["minimum"] is not None]
    maxima = [item["maximum"] for item in stats if item["maximum"] is not None]
    if not minima or not maxima:
        return {"status": "NO_VALID_SAMPLES", "cloud_reflectance_divisor": None}
    minimum = min(minima)
    maximum = max(maxima)
    if minimum >= -0.05 and maximum <= 2.0:
        return {
            "status": "REFLECTANCE_0_1",
            "sample_minimum": minimum,
            "sample_maximum": maximum,
            "cloud_reflectance_divisor": 1.0,
        }
    if sensor == "sentinel-2" and minimum >= 0 and maximum <= 20000:
        return {
            "status": "SENTINEL_L1C_SCALED_10000",
            "sample_minimum": minimum,
            "sample_maximum": maximum,
            "cloud_reflectance_divisor": 10000.0,
        }
    return {
        "status": "CALIBRATION_REQUIRED",
        "sample_minimum": minimum,
        "sample_maximum": maximum,
        "cloud_reflectance_divisor": None,
    }


def inspect_scene(
    path: str | Path,
    *,
    sensor: str,
    acquired_at: str | None = None,
    scene_id: str | None = None,
    band_order: Sequence[str] | None = None,
    allow_provisional_balkan_crop: bool = False,
) -> dict[str, Any]:
    """Inspect a scene without loading the full raster into memory."""
    if sensor not in SUPPORTED_SENSORS:
        raise ValueError(f"sensor must be one of {SUPPORTED_SENSORS}")
    if allow_provisional_balkan_crop and sensor != "balkan-1":
        raise ValueError("The provisional Balkan crop adapter is only valid for balkan-1")
    explicit_band_order = None if band_order is None else list(band_order)
    if explicit_band_order is not None and any(
        not isinstance(item, str) or not item.strip() for item in explicit_band_order
    ):
        raise ValueError("band_order entries must be non-empty strings")
    raster_path = Path(path)
    if not raster_path.is_file():
        raise FileNotFoundError(raster_path)

    errors: list[str] = []
    warnings: list[str] = []
    mapping: dict[str, dict[str, Any]] = {}
    bands: list[dict[str, Any]] = []

    with rasterio.open(raster_path) as dataset:
        if dataset.driver != "GTiff":
            errors.append(f"Expected GeoTIFF driver, found {dataset.driver}")
        if dataset.crs is None:
            errors.append("CRS is missing")
        if dataset.transform == Affine.identity():
            errors.append("Affine georeferencing transform is missing")
        bounds = [float(value) for value in dataset.bounds]
        if not all(math.isfinite(value) for value in bounds):
            errors.append("Raster bounds are not finite")
        if dataset.width <= 0 or dataset.height <= 0:
            errors.append("Raster dimensions must be positive")

        source_descriptions = dataset.descriptions
        if explicit_band_order is not None and len(explicit_band_order) != dataset.count:
            errors.append(
                "Explicit band order must contain exactly one entry per raster band: "
                f"expected {dataset.count}, received {len(explicit_band_order)}"
            )
        use_explicit_band_order = (
            explicit_band_order is not None and len(explicit_band_order) == dataset.count
        )
        effective_descriptions = (
            tuple(explicit_band_order) if use_explicit_band_order else source_descriptions
        )
        band_order_source = (
            "explicit_override" if use_explicit_band_order else "geotiff_descriptions"
        )

        for index, (description, source_description, dtype) in enumerate(
            zip(
                effective_descriptions,
                source_descriptions,
                dataset.dtypes,
                strict=True,
            ),
            start=1,
        ):
            role = _logical_role(description)
            stats = _sample_band(dataset, index)
            record = {
                "index": index,
                "description": description,
                "source_description": source_description,
                "logical_role": role,
                "mapping_source": band_order_source,
                "dtype": dtype,
                "sample": stats,
            }
            bands.append(record)
            if role is not None:
                if role in mapping:
                    errors.append(f"Multiple bands resolve to {role}")
                else:
                    mapping[role] = {
                        "index": index,
                        "description": description,
                        "source_description": source_description,
                        "mapping_source": band_order_source,
                    }

        if not use_explicit_band_order and any(
            description is None for description in source_descriptions
        ):
            errors.append("Every input band must have a description")
        unrecognised = [
            item["index"] for item in bands if item["logical_role"] is None
        ]
        if unrecognised:
            warnings.append(f"Unrecognised extra band indices: {unrecognised}")
        if dataset.nodata is None:
            warnings.append("No explicit nodata value is declared")

        raster = {
            "driver": dataset.driver,
            "width": dataset.width,
            "height": dataset.height,
            "band_count": dataset.count,
            "crs": str(dataset.crs) if dataset.crs else None,
            "transform": list(dataset.transform)[:6],
            "bounds": bounds,
            "resolution": [abs(float(dataset.res[0])), abs(float(dataset.res[1]))],
            "nodata": dataset.nodata,
        }

    missing_rgb = [role for role in ("BLUE", "GREEN", "RED") if role not in mapping]
    has_nir_broad = "NIR_BROAD" in mapping
    has_nir_narrow = "NIR_NARROW" in mapping
    if missing_rgb:
        errors.append(f"Missing required RGB roles: {missing_rgb}")
    if not (has_nir_broad or has_nir_narrow):
        errors.append("Missing a recognised NIR band")

    if sensor == "balkan-1":
        if raster["band_count"] != 5:
            errors.append(
                "Balkan-1 preprocessed input must contain exactly five bands"
            )
        if "PANCHROMATIC" not in mapping:
            errors.append("Balkan-1 preprocessed input is missing PANCHROMATIC")
        if unrecognised:
            errors.append("Balkan-1 input contains unrecognised band roles")

    model_roles = {"BLUE", "GREEN", "RED", "NIR_BROAD", "NIR_NARROW"}
    radiometry = _radiometry(
        sensor,
        [item["sample"] for item in bands if item["logical_role"] in model_roles],
    )
    if radiometry["status"] == "CALIBRATION_REQUIRED":
        warnings.append("Radiometric calibration is required before model inference")
    normalised_acquired_at = _validate_acquired_at(acquired_at)
    if normalised_acquired_at is None:
        warnings.append("Acquisition time is missing")

    if sensor == "sentinel-2":
        cloud_status = "READY" if not missing_rgb and has_nir_broad else "MISSING_B08"
        if not missing_rgb and has_nir_narrow:
            crop_status = "READY"
        elif not missing_rgb and has_nir_broad:
            crop_status = "B8_TO_NIR_NARROW_HARMONISATION_REQUIRED"
        else:
            crop_status = "MISSING_REQUIRED_BANDS"
    else:
        cloud_status = "BALKAN_1_CLOUD_ADAPTER_REQUIRED"
        if missing_rgb or not (has_nir_broad or has_nir_narrow):
            crop_status = "MISSING_REQUIRED_BANDS"
        elif allow_provisional_balkan_crop:
            crop_status = "READY_PROVISIONAL"
            warnings.extend(
                [
                    "Balkan-1 NIR transfer into the crop model is enabled for execution "
                    "testing only",
                    "Balkan-1 crop output is not accuracy or sensor-compatibility evidence",
                ]
            )
        else:
            crop_status = "BALKAN_1_SPECTRAL_HARMONISATION_REQUIRED"

    cloud_order = ("NIR_BROAD", "RED", "GREEN", "BLUE")
    crop_order = ("BLUE", "GREEN", "RED", "NIR_NARROW")
    cloud_indices = (
        [mapping[role]["index"] for role in cloud_order]
        if all(role in mapping for role in cloud_order)
        else None
    )
    native_crop_indices = (
        [mapping[role]["index"] for role in crop_order]
        if all(role in mapping for role in crop_order)
        else None
    )
    crop_approximate_indices = None
    if native_crop_indices is None and not missing_rgb and has_nir_broad:
        crop_approximate_indices = [
            mapping[role]["index"]
            for role in ("BLUE", "GREEN", "RED", "NIR_BROAD")
        ]
    crop_indices = native_crop_indices
    crop_source_roles = list(crop_order) if native_crop_indices is not None else None
    spectral_adapter = None
    if sensor == "balkan-1" and allow_provisional_balkan_crop:
        if native_crop_indices is not None:
            crop_source_roles = list(crop_order)
        elif crop_approximate_indices is not None:
            crop_indices = crop_approximate_indices
            crop_source_roles = ["BLUE", "GREEN", "RED", "NIR_BROAD"]
        if crop_indices is not None:
            spectral_adapter = {
                "mode": "PROVISIONAL_BALKAN_1_NIR_TRANSFER",
                "source_nir_role": crop_source_roles[-1],
                "model_nir_role": "NIR_NARROW",
                "validation_status": "EXECUTION_ONLY_UNVALIDATED",
            }

    resolved_scene_id = _validate_scene_id(scene_id or raster_path.stem)
    return {
        "schema_version": SCHEMA_VERSION,
        "scene_id": resolved_scene_id,
        "source_path": str(raster_path.resolve()),
        "source_bytes": raster_path.stat().st_size,
        "sensor": sensor,
        "acquired_at": normalised_acquired_at,
        "sensor_contract": {
            "expected_roles": (
                ["BLUE", "GREEN", "RED", "NIR", "PANCHROMATIC"]
                if sensor == "balkan-1"
                else ["BLUE", "GREEN", "RED", "NIR_BROAD or NIR_NARROW"]
            ),
            "pan_used_by_current_models": False,
            "band_order_source": band_order_source,
            "provisional_crop_adapter_enabled": allow_provisional_balkan_crop,
        },
        "raster": raster,
        "bands": bands,
        "logical_band_mapping": mapping,
        "radiometry": radiometry,
        "readiness": {
            "intake": "READY" if not errors else "REJECT",
            "cloud": cloud_status,
            "crop": crop_status,
        },
        "model_band_routes": {
            "cloud_detection": {
                "expected_logical_order": list(cloud_order),
                "source_band_indices": cloud_indices,
            },
            "crop_classification": {
                "expected_logical_order": list(crop_order),
                "source_band_indices": crop_indices,
                "source_logical_order": crop_source_roles,
                "unharmonised_broad_nir_indices": crop_approximate_indices,
                "spectral_adapter": spectral_adapter,
            },
        },
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a preprocessed multispectral GeoTIFF without running models."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--sensor", required=True, choices=SUPPORTED_SENSORS)
    parser.add_argument("--acquired-at")
    parser.add_argument("--scene-id")
    parser.add_argument(
        "--band-order",
        nargs="+",
        help="Explicit source-band labels, one per raster band",
    )
    parser.add_argument(
        "--allow-provisional-balkan-crop",
        action="store_true",
        help="Enable unvalidated Balkan NIR transfer for execution testing only",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect_scene(
        args.input,
        sensor=args.sensor,
        acquired_at=args.acquired_at,
        scene_id=args.scene_id,
        band_order=args.band_order,
        allow_provisional_balkan_crop=args.allow_provisional_balkan_crop,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if report["readiness"]["intake"] == "READY" else 2)


if __name__ == "__main__":
    main()
