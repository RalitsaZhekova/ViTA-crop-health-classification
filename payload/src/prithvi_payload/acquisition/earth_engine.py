"""Google Earth Engine Sentinel-2 SR acquisition for the payload runtime."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import requests
from prithvi_shared import PayloadAcquisitionCommand

from prithvi_payload.acquisition.errors import AcquisitionError
from prithvi_payload.acquisition.grid import TargetGrid, calculate_target_grid
from prithvi_payload.acquisition.models import AcquiredScene, CandidateMetadata

EARTH_ENGINE_PROJECT_ID = "vita-503208"
EARTH_ENGINE_SCOPE = "https://www.googleapis.com/auth/earthengine"
EARTH_ENGINE_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
# Earth Engine's Sentinel-2 catalog omits the leading zero used by the payload
# contract. Select the provider names, then rename them without changing pixels.
EARTH_ENGINE_SOURCE_BANDS = ("B2", "B3", "B4", "B8", "B8A")
EARTH_ENGINE_BANDS = ("B02", "B03", "B04", "B08", "B8A")
REFLECTANCE_SCALE = 10_000.0
DEFAULT_MAX_CANDIDATES = 50
DEFAULT_MAX_SCENE_ATTEMPTS = 5
DEFAULT_MAX_DOWNLOAD_BYTES = 32 * 1024 * 1024
DOWNLOAD_CONNECT_TIMEOUT_SECONDS = 10
DOWNLOAD_READ_TIMEOUT_SECONDS = 120
DOWNLOAD_ATTEMPTS = 3
TRANSIENT_HTTP_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_-]+")


def _safe_download_request_error(error: Exception) -> dict[str, Any]:
    """Classify a provider exception without retaining its potentially sensitive text."""
    message = str(error).casefold()
    response = getattr(error, "resp", None)
    raw_status = getattr(response, "status", None)
    status_code = raw_status if isinstance(raw_status, int) and 400 <= raw_status <= 599 else None

    if status_code in TRANSIENT_HTTP_STATUS_CODES or any(
        marker in message
        for marker in ("temporarily unavailable", "deadline exceeded", "timed out", "timeout")
    ):
        reason = "transient_provider_failure"
    elif status_code in {401, 403} or any(
        marker in message
        for marker in (
            "permission",
            "forbidden",
            "not authorized",
            "not authorised",
            "access denied",
            "not registered",
            "insufficient authentication",
        )
    ):
        reason = "permission_denied"
    elif status_code == 429 or any(
        marker in message for marker in ("quota", "resource exhausted", "rate limit")
    ):
        reason = "quota_exceeded"
    elif "crs" in message or "projection" in message:
        reason = "invalid_projection"
    elif "affine" in message or "transform" in message:
        reason = "invalid_transform"
    elif "dimension" in message or "pixel grid" in message:
        reason = "invalid_dimensions"
    elif "band" in message:
        reason = "invalid_bands"
    elif "not found" in message:
        reason = "scene_not_found"
    elif "cannot specify" in message or status_code == 400:
        reason = "invalid_request"
    else:
        reason = "request_rejected"

    safe_type = _SAFE_COMPONENT.sub("_", type(error).__name__).strip("_")[:80] or "Exception"
    details: dict[str, Any] = {
        "provider_reason": reason,
        "provider_error_type": safe_type,
    }
    if status_code is not None:
        details["provider_status_code"] = status_code
    return details


def initialize_earth_engine(project_id: str) -> None:
    """Initialize with Application Default Credentials and no interactive fallback."""
    if project_id != EARTH_ENGINE_PROJECT_ID:
        raise AcquisitionError(
            "EARTH_ENGINE_CONFIGURATION_ERROR",
            "Earth Engine project does not match the fixed payload project",
        )
    try:
        import ee
        import google.auth

        credentials, _ = google.auth.default(scopes=[EARTH_ENGINE_SCOPE])
        ee.Initialize(credentials=credentials, project=project_id)
    except Exception:
        raise AcquisitionError(
            "EARTH_ENGINE_AUTHENTICATION_FAILED",
            "Earth Engine initialization failed using runtime credentials",
        ) from None


def _valid_cloud_percent(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and 0 <= number <= 100 else None


def _timestamp(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def _iso_timestamp(milliseconds: int) -> str:
    return datetime.fromtimestamp(milliseconds / 1000.0, tz=timezone.utc).isoformat()


def parse_candidate_features(features: list[dict[str, Any]]) -> list[CandidateMetadata]:
    candidates: list[CandidateMetadata] = []
    for feature in features:
        properties = feature.get("properties") if isinstance(feature, dict) else None
        if not isinstance(properties, dict):
            continue
        system_index = properties.get("system:index") or feature.get("id", "").rsplit("/", 1)[-1]
        if not isinstance(system_index, str) or not system_index:
            continue
        acquired_at_millis = _timestamp(properties.get("system:time_start"))
        safe_metadata = {
            name: properties.get(name)
            for name in (
                "MGRS_TILE",
                "PROCESSING_BASELINE",
                "SPACECRAFT_NAME",
            )
            if properties.get(name) is not None
        }
        product_id = properties.get("PRODUCT_ID")
        candidates.append(
            CandidateMetadata(
                system_index=system_index,
                acquired_at=_iso_timestamp(acquired_at_millis),
                acquired_at_millis=acquired_at_millis,
                metadata_cloud_percent=_valid_cloud_percent(
                    properties.get("CLOUDY_PIXEL_PERCENTAGE")
                ),
                product_id=product_id if isinstance(product_id, str) else None,
                source_metadata=safe_metadata,
            )
        )
    return candidates


def order_candidates(
    candidates: list[CandidateMetadata],
    command: PayloadAcquisitionCommand,
) -> list[CandidateMetadata]:
    source = command.source
    if source.selection_policy == "target_cloud_range":
        candidates = [
            candidate
            for candidate in candidates
            if candidate.metadata_cloud_percent is not None
            and source.target_cloud_min_percent
            <= candidate.metadata_cloud_percent
            <= source.target_cloud_max_percent
        ]
        if not candidates:
            raise AcquisitionError(
                "EARTH_ENGINE_NO_TARGET_CLOUD_SCENE",
                "No Earth Engine scene has metadata cloud coverage in the requested range",
            )
        candidates.sort(
            key=lambda candidate: (
                abs(candidate.metadata_cloud_percent - source.target_cloud_ideal_percent),
                -candidate.acquired_at_millis,
                candidate.system_index,
            )
        )
    else:
        candidates.sort(
            key=lambda candidate: (
                candidate.metadata_cloud_percent is None,
                candidate.metadata_cloud_percent
                if candidate.metadata_cloud_percent is not None
                else math.inf,
                -candidate.acquired_at_millis,
                candidate.system_index,
            )
        )
    return [
        replace(candidate, candidate_rank=index) for index, candidate in enumerate(candidates, 1)
    ]


def _candidate_directory_name(system_index: str) -> str:
    safe = _SAFE_COMPONENT.sub("_", system_index).strip("_")[:64] or "scene"
    suffix = hashlib.sha256(system_index.encode("utf-8")).hexdigest()[:10]
    return f"{safe}_{suffix}"


def _select_payload_bands(image: Any) -> Any:
    """Map fixed Earth Engine catalog names to the payload's canonical names."""
    return image.select(list(EARTH_ENGINE_SOURCE_BANDS), list(EARTH_ENGINE_BANDS))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def normalize_and_validate_geotiff(
    raw_path: Path,
    destination_path: Path,
    *,
    grid: TargetGrid,
) -> None:
    """Copy pixels unchanged, assign the fixed band metadata, then validate."""
    temporary = destination_path.with_suffix(".partial")
    temporary.unlink(missing_ok=True)
    try:
        shutil.copyfile(raw_path, temporary)
        with rasterio.open(temporary, "r+") as dataset:
            if dataset.count != len(EARTH_ENGINE_BANDS):
                raise AcquisitionError(
                    "EARTH_ENGINE_INVALID_RASTER",
                    "Downloaded raster does not contain exactly five bands",
                )
            if dataset.width <= 0 or dataset.height <= 0:
                raise AcquisitionError(
                    "EARTH_ENGINE_INVALID_RASTER", "Downloaded raster has invalid dimensions"
                )
            if dataset.width != grid.width or dataset.height != grid.height:
                raise AcquisitionError(
                    "EARTH_ENGINE_INVALID_RASTER",
                    "Downloaded raster dimensions do not match the requested target grid",
                )
            if dataset.crs is None or dataset.crs.to_string() != grid.crs:
                raise AcquisitionError(
                    "EARTH_ENGINE_INVALID_RASTER",
                    "Downloaded raster CRS does not match the requested target grid",
                )
            if not dataset.transform.almost_equals(grid.transform):
                raise AcquisitionError(
                    "EARTH_ENGINE_INVALID_RASTER",
                    "Downloaded raster transform does not match the requested target grid",
                )
            if any(np.dtype(dtype).kind not in {"i", "u"} for dtype in dataset.dtypes):
                raise AcquisitionError(
                    "EARTH_ENGINE_INVALID_RASTER",
                    "Downloaded raster reflectance values must use an integer dtype",
                )
            dataset.descriptions = EARTH_ENGINE_BANDS
            dataset.update_tags(REFLECTANCE_SCALE=str(int(REFLECTANCE_SCALE)))
            sample = dataset.read(
                out_shape=(dataset.count, min(dataset.height, 128), min(dataset.width, 128))
            )
            if not np.isfinite(sample).all():
                raise AcquisitionError(
                    "EARTH_ENGINE_INVALID_RASTER",
                    "Downloaded raster contains non-finite sampled values",
                )
        with rasterio.open(temporary) as dataset:
            if dataset.descriptions != EARTH_ENGINE_BANDS:
                raise AcquisitionError(
                    "EARTH_ENGINE_INVALID_RASTER", "Normalized raster band order is invalid"
                )
            if dataset.tags().get("REFLECTANCE_SCALE") != str(int(REFLECTANCE_SCALE)):
                raise AcquisitionError(
                    "EARTH_ENGINE_INVALID_RASTER", "Normalized raster scale metadata is invalid"
                )
        temporary.replace(destination_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


class EarthEngineAcquisitionProvider:
    """Fixed Sentinel-2 SR provider used only inside the persistent payload."""

    def __init__(
        self,
        *,
        project_id: str = EARTH_ENGINE_PROJECT_ID,
        max_candidates: int | None = None,
        max_scene_attempts: int | None = None,
        max_download_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
        session: requests.Session | None = None,
    ) -> None:
        self.project_id = project_id
        self.max_candidates = (
            int(os.environ.get("EE_MAX_CANDIDATES", DEFAULT_MAX_CANDIDATES))
            if max_candidates is None
            else max_candidates
        )
        self.max_scene_attempts = (
            int(os.environ.get("EE_MAX_SCENE_ATTEMPTS", DEFAULT_MAX_SCENE_ATTEMPTS))
            if max_scene_attempts is None
            else max_scene_attempts
        )
        if not 1 <= self.max_candidates <= DEFAULT_MAX_CANDIDATES:
            raise ValueError("EE_MAX_CANDIDATES must be within 1..50")
        if not 1 <= self.max_scene_attempts <= DEFAULT_MAX_SCENE_ATTEMPTS:
            raise ValueError("EE_MAX_SCENE_ATTEMPTS must be within 1..5")
        if max_download_bytes <= 0:
            raise ValueError("max_download_bytes must be positive")
        self.max_download_bytes = max_download_bytes
        self.session = session or requests.Session()

    def initialize(self) -> None:
        initialize_earth_engine(self.project_id)

    def search_candidates(
        self, command: PayloadAcquisitionCommand
    ) -> tuple[list[CandidateMetadata], TargetGrid]:
        grid = calculate_target_grid(command.source.bbox_wgs84)
        try:
            import ee

            region = ee.Geometry.Rectangle(list(command.source.bbox_wgs84), geodesic=False)
            end_exclusive = command.source.end_date + timedelta(days=1)
            collection = (
                ee.ImageCollection(EARTH_ENGINE_COLLECTION)
                .filterBounds(region)
                .filterDate(command.source.start_date.isoformat(), end_exclusive.isoformat())
                .sort("system:time_start", False)
                .limit(self.max_candidates)
            )
            response = collection.getInfo()
        except Exception:
            raise AcquisitionError(
                "EARTH_ENGINE_QUERY_FAILED",
                "Earth Engine candidate search failed",
            ) from None
        features = response.get("features", []) if isinstance(response, dict) else []
        candidates = parse_candidate_features(features if isinstance(features, list) else [])
        if not candidates:
            raise AcquisitionError(
                "EARTH_ENGINE_NO_SCENE", "Earth Engine returned no scenes for the requested region"
            )
        return order_candidates(candidates, command), grid

    def _download_parameters(
        self,
        grid: TargetGrid,
    ) -> dict[str, Any]:
        return {
            "bands": list(EARTH_ENGINE_BANDS),
            "crs": grid.crs,
            "crs_transform": list(grid.transform)[:6],
            "dimensions": [grid.width, grid.height],
            "format": "GEO_TIFF",
            "filePerBand": False,
        }

    def _stream_candidate(self, image: Any, parameters: dict[str, Any], partial: Path) -> None:
        partial.unlink(missing_ok=True)
        for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
            try:
                try:
                    signed_url = image.getDownloadURL(parameters)
                except Exception as error:
                    details = _safe_download_request_error(error)
                    if (
                        details["provider_reason"] == "transient_provider_failure"
                        and attempt < DOWNLOAD_ATTEMPTS
                    ):
                        time.sleep(2 ** (attempt - 1))
                        continue
                    raise AcquisitionError(
                        "EARTH_ENGINE_DOWNLOAD_REQUEST_FAILED",
                        "Earth Engine rejected the fixed GeoTIFF download request",
                        details=details,
                    ) from None
                with self.session.get(
                    signed_url,
                    stream=True,
                    timeout=(DOWNLOAD_CONNECT_TIMEOUT_SECONDS, DOWNLOAD_READ_TIMEOUT_SECONDS),
                ) as response:
                    content_type = response.headers.get("Content-Type", "").lower()
                    if response.status_code in TRANSIENT_HTTP_STATUS_CODES:
                        raise requests.RequestException("transient provider response")
                    if response.status_code != 200:
                        raise AcquisitionError(
                            "EARTH_ENGINE_DOWNLOAD_FAILED",
                            f"Earth Engine download returned HTTP {response.status_code}",
                        )
                    if "html" in content_type or "json" in content_type:
                        raise AcquisitionError(
                            "EARTH_ENGINE_INVALID_RESPONSE",
                            "Earth Engine returned an error document instead of a GeoTIFF",
                        )
                    content_length = response.headers.get("Content-Length")
                    if content_length is not None:
                        try:
                            declared_bytes = int(content_length)
                        except ValueError:
                            declared_bytes = 0
                        if declared_bytes > self.max_download_bytes:
                            raise AcquisitionError(
                                "EARTH_ENGINE_DOWNLOAD_TOO_LARGE",
                                "Earth Engine response exceeded the fixed byte limit",
                            )
                    received = 0
                    with partial.open("wb") as output:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if not chunk:
                                continue
                            received += len(chunk)
                            if received > self.max_download_bytes:
                                raise AcquisitionError(
                                    "EARTH_ENGINE_DOWNLOAD_TOO_LARGE",
                                    "Earth Engine response exceeded the fixed byte limit",
                                )
                            output.write(chunk)
                if received < 4:
                    raise AcquisitionError(
                        "EARTH_ENGINE_INVALID_RESPONSE", "Earth Engine returned an empty response"
                    )
                with partial.open("rb") as source:
                    if source.read(4) not in {b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"}:
                        raise AcquisitionError(
                            "EARTH_ENGINE_INVALID_RESPONSE", "Earth Engine response is not a TIFF"
                        )
                return
            except AcquisitionError:
                partial.unlink(missing_ok=True)
                raise
            except requests.RequestException:
                partial.unlink(missing_ok=True)
                if attempt == DOWNLOAD_ATTEMPTS:
                    raise AcquisitionError(
                        "EARTH_ENGINE_DOWNLOAD_FAILED",
                        "Earth Engine download failed after bounded retries",
                    ) from None
                time.sleep(2 ** (attempt - 1))
            except Exception:
                partial.unlink(missing_ok=True)
                raise AcquisitionError(
                    "EARTH_ENGINE_DOWNLOAD_FAILED", "Earth Engine download failed"
                ) from None

    def acquire_candidate(
        self,
        command: PayloadAcquisitionCommand,
        candidate: CandidateMetadata,
        destination_directory: Path,
        *,
        grid: TargetGrid,
    ) -> AcquiredScene:
        acquisition_started = time.perf_counter()
        candidate_root = destination_directory / _candidate_directory_name(candidate.system_index)
        candidate_root.mkdir(parents=True, exist_ok=True)
        partial_path = candidate_root / "download.partial"
        raw_path = candidate_root / "source_raw.tif"
        scene_path = candidate_root / "scene.tif"
        record_path = candidate_root / "acquisition_record.json"
        try:
            import ee

            image = _select_payload_bands(
                ee.Image(f"{EARTH_ENGINE_COLLECTION}/{candidate.system_index}")
            )
            download_started = time.perf_counter()
            self._stream_candidate(image, self._download_parameters(grid), partial_path)
            download_seconds = time.perf_counter() - download_started
            partial_path.replace(raw_path)
            validation_started = time.perf_counter()
            normalize_and_validate_geotiff(raw_path, scene_path, grid=grid)
            validation_seconds = time.perf_counter() - validation_started
            acquisition_timing = {
                "earth_engine_download_seconds": download_seconds,
                "geotiff_validation_seconds": validation_seconds,
                "total_acquisition_seconds": time.perf_counter() - acquisition_started,
            }
            acquired = AcquiredScene(
                local_tiff_path=scene_path.resolve(),
                provider="earth_engine",
                collection=EARTH_ENGINE_COLLECTION,
                provider_scene_id=candidate.system_index,
                product_id=candidate.product_id,
                acquired_at=candidate.acquired_at,
                metadata_cloud_percent=candidate.metadata_cloud_percent,
                requested_bbox_wgs84=command.source.bbox_wgs84,
                output_crs=grid.crs,
                output_transform=tuple(grid.transform)[:6],
                width=grid.width,
                height=grid.height,
                band_names=EARTH_ENGINE_BANDS,
                reflectance_scale=REFLECTANCE_SCALE,
                source_metadata=candidate.source_metadata,
                sha256=_sha256(scene_path),
                byte_size=scene_path.stat().st_size,
                candidate_rank=candidate.candidate_rank,
                timing=acquisition_timing,
            )
            _write_json_atomic(
                record_path,
                {
                    "status": "acquired",
                    **candidate.safe_record(),
                    "local": {
                        "width": grid.width,
                        "height": grid.height,
                        "crs": grid.crs,
                        "transform": list(grid.transform)[:6],
                        "bands": list(EARTH_ENGINE_BANDS),
                        "reflectance_scale": REFLECTANCE_SCALE,
                        "sha256": acquired.sha256,
                        "byte_size": acquired.byte_size,
                        "resampling_policy": "earth_engine_default_nearest",
                    },
                    "timing": acquisition_timing,
                },
            )
            return acquired
        except AcquisitionError as error:
            partial_path.unlink(missing_ok=True)
            _write_json_atomic(
                record_path,
                {"status": "failed", **candidate.safe_record(), "error": error.safe_record()},
            )
            raise
        except Exception:
            partial_path.unlink(missing_ok=True)
            safe_error = AcquisitionError(
                "EARTH_ENGINE_ACQUISITION_FAILED", "Earth Engine candidate acquisition failed"
            )
            _write_json_atomic(
                record_path,
                {"status": "failed", **candidate.safe_record(), "error": safe_error.safe_record()},
            )
            raise safe_error from None

    def record_candidate_evaluation(
        self,
        acquired: AcquiredScene,
        evaluation: dict[str, Any],
    ) -> None:
        """Persist safe payload measurements beside the internal acquisition record."""
        record_path = acquired.local_tiff_path.parent / "acquisition_record.json"
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            record = {
                "status": "acquired",
                "provider_scene_id": acquired.provider_scene_id,
                "acquired_at": acquired.acquired_at,
                "metadata_cloud_percent": acquired.metadata_cloud_percent,
                "candidate_rank": acquired.candidate_rank,
            }
        record["payload_evaluation"] = evaluation
        _write_json_atomic(record_path, record)
