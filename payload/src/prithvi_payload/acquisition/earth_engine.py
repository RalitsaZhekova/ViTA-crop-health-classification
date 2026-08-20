"""Google Earth Engine Sentinel-2 SR acquisition for the warm payload service."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import requests

from prithvi_payload.acquisition.errors import AcquisitionError
from prithvi_payload.acquisition.grid import TargetGrid, calculate_target_grid

EARTH_ENGINE_SCOPE = "https://www.googleapis.com/auth/earthengine"
EARTH_ENGINE_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
EARTH_ENGINE_SOURCE_BANDS = ("B2", "B3", "B4", "B8", "B8A")
EARTH_ENGINE_BANDS = ("B02", "B03", "B04", "B08", "B8A")
REFLECTANCE_SCALE = 10_000
MAXIMUM_SEARCH_DAYS = 92
MAXIMUM_CANDIDATES = 3
MAXIMUM_DOWNLOAD_BYTES = 32 * 1024 * 1024
MINIMUM_VALID_PIXEL_FRACTION = 0.8
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass(frozen=True)
class Candidate:
    system_index: str
    acquired_at: str
    acquired_at_millis: int
    metadata_cloud_percentage: float | None
    product_id: str | None
    rank: int = 0


@dataclass(frozen=True)
class AcquiredSentinelScene:
    local_tiff_path: Path
    provider_scene_id: str
    product_id: str | None
    acquired_at: str
    metadata_cloud_percentage: float | None
    requested_bbox_wgs84: tuple[float, float, float, float]
    output_crs: str
    output_transform: tuple[float, float, float, float, float, float]
    sha256: str
    byte_size: int
    candidate_rank: int
    candidate_attempt_count: int
    timing: dict[str, float]

    def safe_provenance(self) -> dict[str, Any]:
        return {
            "provider": "earth_engine",
            "collection": EARTH_ENGINE_COLLECTION,
            "provider_scene_id": self.provider_scene_id,
            "product_id": self.product_id,
            "acquired_at": self.acquired_at,
            "requested_bbox_wgs84": list(self.requested_bbox_wgs84),
            "source_crs": self.output_crs,
            "source_transform": list(self.output_transform),
            "source_scale": REFLECTANCE_SCALE,
            "source_sha256": self.sha256,
            "source_bytes": self.byte_size,
            "selection_policy": "least_cloudy",
            "target_cloud_range": None,
            "earth_engine_metadata_cloud_percentage": self.metadata_cloud_percentage,
            "candidate_rank": self.candidate_rank,
            "candidate_attempt_count": self.candidate_attempt_count,
            "resampling_policy": "earth_engine_default_nearest",
        }


def earth_engine_configuration() -> dict[str, Any]:
    project = os.environ.get("VITA_EE_PROJECT", "").strip()
    credential_value = os.environ.get(
        "VITA_EE_CREDENTIALS", "/earth-engine-auth/credentials.json"
    ).strip()
    credentials = Path(credential_value) if credential_value else None
    configured = bool(project and credentials is not None and credentials.is_file())
    return {
        "configured": configured,
        "project": project or None,
        "credentials_present": bool(credentials is not None and credentials.is_file()),
        "maximum_search_days": MAXIMUM_SEARCH_DAYS,
        "maximum_dimension_pixels": 1000,
        "resolution_metres": 10,
        "minimum_valid_pixel_fraction": MINIMUM_VALID_PIXEL_FRACTION,
    }


def initialize_earth_engine() -> None:
    configuration = earth_engine_configuration()
    if not configuration["configured"]:
        raise AcquisitionError(
            "EARTH_ENGINE_NOT_CONFIGURED",
            "Live Sentinel acquisition is not configured on the payload computer",
        )
    try:
        import ee
        import google.auth

        credentials, _ = google.auth.load_credentials_from_file(
            os.environ.get("VITA_EE_CREDENTIALS", "/earth-engine-auth/credentials.json"),
            scopes=[EARTH_ENGINE_SCOPE],
        )
        ee.Initialize(credentials=credentials, project=configuration["project"])
    except Exception:
        raise AcquisitionError(
            "EARTH_ENGINE_AUTHENTICATION_FAILED",
            "Earth Engine authentication failed on the payload computer",
        ) from None


def _valid_cloud_percentage(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and 0 <= number <= 100 else None


def _parse_candidates(features: list[dict[str, Any]]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for feature in features:
        properties = feature.get("properties") if isinstance(feature, dict) else None
        if not isinstance(properties, dict):
            continue
        system_index = properties.get("system:index")
        if not isinstance(system_index, str) or not system_index:
            feature_id = feature.get("id")
            system_index = feature_id.rsplit("/", 1)[-1] if isinstance(feature_id, str) else ""
        timestamp = properties.get("system:time_start")
        if (
            not system_index
            or isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
        ):
            continue
        milliseconds = int(timestamp)
        product_id = properties.get("PRODUCT_ID")
        candidates.append(
            Candidate(
                system_index=system_index,
                acquired_at=datetime.fromtimestamp(
                    milliseconds / 1000.0, tz=timezone.utc
                ).isoformat(),
                acquired_at_millis=milliseconds,
                metadata_cloud_percentage=_valid_cloud_percentage(
                    properties.get("CLOUDY_PIXEL_PERCENTAGE")
                ),
                product_id=product_id if isinstance(product_id, str) else None,
            )
        )
    candidates.sort(
        key=lambda candidate: (
            candidate.metadata_cloud_percentage is None,
            candidate.metadata_cloud_percentage
            if candidate.metadata_cloud_percentage is not None
            else math.inf,
            -candidate.acquired_at_millis,
            candidate.system_index,
        )
    )
    return [replace(candidate, rank=index) for index, candidate in enumerate(candidates, 1)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_directory_name(system_index: str) -> str:
    safe = _SAFE_COMPONENT.sub("_", system_index).strip("_")[:64] or "scene"
    suffix = hashlib.sha256(system_index.encode("utf-8")).hexdigest()[:10]
    return f"{safe}_{suffix}"


def _normalise_geotiff(
    raw_path: Path,
    destination_path: Path,
    *,
    grid: TargetGrid,
    acquired_at: str,
) -> None:
    temporary = destination_path.with_suffix(".partial")
    temporary.unlink(missing_ok=True)
    try:
        with rasterio.open(raw_path) as source:
            if source.count != len(EARTH_ENGINE_BANDS):
                raise AcquisitionError(
                    "EARTH_ENGINE_INVALID_RASTER",
                    "Downloaded Sentinel raster does not contain the required five bands",
                )
            if source.width != grid.width or source.height != grid.height:
                raise AcquisitionError(
                    "EARTH_ENGINE_INVALID_RASTER",
                    "Downloaded Sentinel raster dimensions do not match the requested area",
                )
            if source.crs is None or source.crs.to_string() != grid.crs:
                raise AcquisitionError(
                    "EARTH_ENGINE_INVALID_RASTER",
                    "Downloaded Sentinel raster projection does not match the requested area",
                )
            if not source.transform.almost_equals(grid.transform):
                raise AcquisitionError(
                    "EARTH_ENGINE_INVALID_RASTER",
                    "Downloaded Sentinel raster grid does not match the requested area",
                )
            if any(np.dtype(dtype).kind not in {"i", "u"} for dtype in source.dtypes):
                raise AcquisitionError(
                    "EARTH_ENGINE_INVALID_RASTER",
                    "Downloaded Sentinel reflectance values are not integer scaled",
                )
            profile = source.profile.copy()
            profile.update(
                driver="GTiff",
                interleave="band",
                photometric="MINISBLACK",
                compress="DEFLATE",
                zlevel=1,
                num_threads="ALL_CPUS",
                BIGTIFF="IF_SAFER",
            )
            with rasterio.open(temporary, "w", **profile) as destination:
                valid_pixels = 0
                total_pixels = 0
                for _, window in source.block_windows(1):
                    values = source.read(window=window)
                    destination.write(values, window=window)
                    valid = np.any(values != 0, axis=0)
                    valid_pixels += int(np.count_nonzero(valid))
                    total_pixels += int(valid.size)
                valid_fraction = valid_pixels / total_pixels if total_pixels else 0.0
                if valid_fraction < MINIMUM_VALID_PIXEL_FRACTION:
                    raise AcquisitionError(
                        "EARTH_ENGINE_INCOMPLETE_COVERAGE",
                        "The selected Sentinel scene does not sufficiently cover this area",
                        details={"valid_pixel_fraction": valid_fraction},
                    )
                destination.descriptions = EARTH_ENGINE_BANDS
                destination.update_tags(
                    ACQUIRED_AT=acquired_at,
                    REFLECTANCE_SCALE=str(REFLECTANCE_SCALE),
                    SENSOR="sentinel-2",
                )
        temporary.replace(destination_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


class EarthEngineAcquisitionProvider:
    def __init__(
        self,
        *,
        max_candidates: int = MAXIMUM_CANDIDATES,
        max_download_bytes: int = MAXIMUM_DOWNLOAD_BYTES,
        session: requests.Session | None = None,
    ) -> None:
        if not 1 <= max_candidates <= MAXIMUM_CANDIDATES:
            raise ValueError("max_candidates must be within 1..3")
        self.max_candidates = max_candidates
        self.max_download_bytes = max_download_bytes
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", "ViTA-Payload/1.0 EarthEngineAcquisition")

    def initialize(self) -> None:
        initialize_earth_engine()

    def search(
        self,
        bbox_wgs84: tuple[float, float, float, float],
        start_date: date,
        end_date: date,
    ) -> tuple[list[Candidate], TargetGrid, float]:
        if start_date > end_date:
            raise AcquisitionError("INVALID_DATE_RANGE", "Start date must not follow end date")
        if (end_date - start_date).days + 1 > MAXIMUM_SEARCH_DAYS:
            raise AcquisitionError(
                "INVALID_DATE_RANGE",
                f"Search dates must span no more than {MAXIMUM_SEARCH_DAYS} days",
            )
        grid = calculate_target_grid(bbox_wgs84)
        started = time.perf_counter()
        try:
            import ee

            region = ee.Geometry.Rectangle(list(bbox_wgs84), geodesic=False)
            end_exclusive = end_date + timedelta(days=1)
            response = (
                ee.ImageCollection(EARTH_ENGINE_COLLECTION)
                .filterBounds(region)
                .filterDate(start_date.isoformat(), end_exclusive.isoformat())
                .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 50))
                .sort("CLOUDY_PIXEL_PERCENTAGE")
                .limit(self.max_candidates)
                .getInfo()
            )
        except Exception:
            raise AcquisitionError(
                "EARTH_ENGINE_QUERY_FAILED",
                "Earth Engine could not search for Sentinel-2 scenes",
            ) from None
        features = response.get("features", []) if isinstance(response, dict) else []
        candidates = _parse_candidates(features if isinstance(features, list) else [])
        if not candidates:
            raise AcquisitionError(
                "EARTH_ENGINE_NO_SCENE",
                "No sufficiently clear Sentinel-2 scene was found for this area and date range",
            )
        return candidates, grid, time.perf_counter() - started

    @staticmethod
    def _download_parameters(grid: TargetGrid) -> dict[str, Any]:
        return {
            "bands": list(EARTH_ENGINE_BANDS),
            "crs": grid.crs,
            "crs_transform": list(grid.transform)[:6],
            "dimensions": [grid.width, grid.height],
            "format": "GEO_TIFF",
            "filePerBand": False,
        }

    def _download(self, image: Any, grid: TargetGrid, destination: Path) -> None:
        try:
            signed_url = image.getDownloadURL(self._download_parameters(grid))
            with self.session.get(signed_url, stream=True, timeout=(10, 180)) as response:
                if response.status_code != 200:
                    raise AcquisitionError(
                        "EARTH_ENGINE_DOWNLOAD_FAILED",
                        f"Earth Engine download returned HTTP {response.status_code}",
                    )
                content_type = response.headers.get("Content-Type", "").casefold()
                if "html" in content_type or "json" in content_type:
                    raise AcquisitionError(
                        "EARTH_ENGINE_INVALID_RESPONSE",
                        "Earth Engine returned an error document instead of a GeoTIFF",
                    )
                received = 0
                with destination.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        received += len(chunk)
                        if received > self.max_download_bytes:
                            raise AcquisitionError(
                                "EARTH_ENGINE_DOWNLOAD_TOO_LARGE",
                                "Earth Engine response exceeded the 32 MiB payload limit",
                            )
                        output.write(chunk)
            with destination.open("rb") as source:
                if source.read(4) not in {b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"}:
                    raise AcquisitionError(
                        "EARTH_ENGINE_INVALID_RESPONSE",
                        "Earth Engine response is not a GeoTIFF",
                    )
        except AcquisitionError:
            destination.unlink(missing_ok=True)
            raise
        except requests.RequestException:
            destination.unlink(missing_ok=True)
            raise AcquisitionError(
                "EARTH_ENGINE_DOWNLOAD_FAILED",
                "Earth Engine download did not complete",
            ) from None
        except Exception:
            destination.unlink(missing_ok=True)
            raise AcquisitionError(
                "EARTH_ENGINE_DOWNLOAD_FAILED",
                "Earth Engine could not prepare the Sentinel-2 download",
            ) from None

    def acquire(
        self,
        *,
        bbox_wgs84: tuple[float, float, float, float],
        start_date: date,
        end_date: date,
        destination_root: Path,
    ) -> AcquiredSentinelScene:
        total_started = time.perf_counter()
        self.initialize()
        candidates, grid, search_seconds = self.search(bbox_wgs84, start_date, end_date)
        failures: list[AcquisitionError] = []
        for attempt, candidate in enumerate(candidates, 1):
            candidate_root = destination_root / _candidate_directory_name(candidate.system_index)
            candidate_root.mkdir(parents=True, exist_ok=True)
            raw_path = candidate_root / "download.tif"
            scene_path = candidate_root / "scene.tif"
            try:
                import ee

                image = ee.Image(f"{EARTH_ENGINE_COLLECTION}/{candidate.system_index}").select(
                    list(EARTH_ENGINE_SOURCE_BANDS), list(EARTH_ENGINE_BANDS)
                )
                download_started = time.perf_counter()
                self._download(image, grid, raw_path)
                download_seconds = time.perf_counter() - download_started
                validation_started = time.perf_counter()
                _normalise_geotiff(
                    raw_path,
                    scene_path,
                    grid=grid,
                    acquired_at=candidate.acquired_at,
                )
                validation_seconds = time.perf_counter() - validation_started
                raw_path.unlink(missing_ok=True)
                timing = {
                    "earth_engine_search_seconds": search_seconds,
                    "earth_engine_download_seconds": download_seconds,
                    "geotiff_validation_seconds": validation_seconds,
                    "total_acquisition_seconds": time.perf_counter() - total_started,
                }
                acquired = AcquiredSentinelScene(
                    local_tiff_path=scene_path.resolve(),
                    provider_scene_id=candidate.system_index,
                    product_id=candidate.product_id,
                    acquired_at=candidate.acquired_at,
                    metadata_cloud_percentage=candidate.metadata_cloud_percentage,
                    requested_bbox_wgs84=bbox_wgs84,
                    output_crs=grid.crs,
                    output_transform=tuple(grid.transform)[:6],
                    sha256=_sha256(scene_path),
                    byte_size=scene_path.stat().st_size,
                    candidate_rank=candidate.rank,
                    candidate_attempt_count=attempt,
                    timing=timing,
                )
                (candidate_root / "acquisition.json").write_text(
                    json.dumps(
                        {
                            "status": "ACQUIRED",
                            "provider_scene_id": acquired.provider_scene_id,
                            "acquired_at": acquired.acquired_at,
                            "metadata_cloud_percentage": acquired.metadata_cloud_percentage,
                            "candidate_rank": acquired.candidate_rank,
                            "candidate_attempt_count": acquired.candidate_attempt_count,
                            "grid": {
                                "crs": grid.crs,
                                "transform": list(grid.transform)[:6],
                                "width": grid.width,
                                "height": grid.height,
                            },
                            "sha256": acquired.sha256,
                            "bytes": acquired.byte_size,
                            "timing_seconds": timing,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return acquired
            except AcquisitionError as error:
                raw_path.unlink(missing_ok=True)
                scene_path.unlink(missing_ok=True)
                failures.append(error)
        raise failures[-1] if failures else AcquisitionError(
            "EARTH_ENGINE_ACQUISITION_FAILED",
            "Earth Engine acquisition did not produce a usable Sentinel-2 scene",
        )
