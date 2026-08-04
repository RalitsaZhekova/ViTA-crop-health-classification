"""Checksum-verified storage for compact crop-condition downlink bundles."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from PIL import Image

CATALOG_SCHEMA_VERSION = 1
BUNDLE_SCHEMA_VERSION = "1.0"
BUNDLE_PRODUCT_TYPE = "vita.crop-condition.web-bundle"
EXPECTED_FILES = frozenset({"scene.json", "scene.webp", "condition.png"})
CONDITION_LABELS = frozenset(
    {
        "Nominal",
        "Watch",
        "Moderate anomaly",
        "High anomaly",
        "Insufficient data",
    }
)
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SOURCE_MEASUREMENT_FIELDS = (
    "payload_measured_thick_cloud_percentage",
    "payload_measured_thin_cloud_percentage",
    "payload_measured_shadow_percentage",
    "payload_measured_cloud_percentage",
    "payload_measured_unusable_percentage",
)


class BundleValidationError(ValueError):
    """Raised when a downlink bundle does not satisfy the ground contract."""


class SceneConflictError(ValueError):
    """Raised when a scene identifier is reused for different content."""


@dataclass(frozen=True)
class ValidatedBundle:
    root: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    scene_id: str
    region_id: str
    sensor: str
    acquired_at: str
    status: str
    label: str
    score: float | None
    evidence_quality_score: float
    analysis_percentage: float
    bounds_wgs84: tuple[float, float, float, float]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BundleValidationError(f"{name} must be a JSON object")
    return value


def _require_identifier(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not SAFE_IDENTIFIER.fullmatch(value):
        raise BundleValidationError(f"{name} must be a safe non-empty identifier")
    return value


def _require_percentage(value: Any, *, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
        raise BundleValidationError(f"{name} must be a finite percentage within 0..100")
    return float(value)


def _normalise_datetime(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise BundleValidationError("acquired_at must be a non-empty ISO date-time")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise BundleValidationError("acquired_at is not a valid ISO date-time") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise BundleValidationError("acquired_at must include a UTC offset")
    return parsed.astimezone(timezone.utc).isoformat()


def _iter_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_strings(item)


def _validate_no_absolute_paths(manifest: dict[str, Any]) -> None:
    for value in _iter_strings(manifest):
        if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
            raise BundleValidationError("Bundle metadata must not contain absolute paths")


def _validate_source(value: Any) -> None:
    source = _require_object(value, name="source")
    provider = source.get("provider")
    if provider not in {"local_file", "earth_engine"}:
        raise BundleValidationError("source.provider is unsupported")
    for field in SOURCE_MEASUREMENT_FIELDS:
        _require_percentage(source.get(field), name=f"source.{field}")
    if provider == "local_file":
        allowed = {"provider", "acquired_at", *SOURCE_MEASUREMENT_FIELDS}
        if not set(source) <= allowed:
            raise BundleValidationError("Local-file source provenance contains unknown fields")
        return

    required = {
        "collection",
        "provider_scene_id",
        "product_id",
        "acquired_at",
        "requested_bbox_wgs84",
        "source_crs",
        "source_transform",
        "source_scale",
        "source_sha256",
        "source_bytes",
        "selection_policy",
        "target_cloud_range",
        "earth_engine_metadata_cloud_percentage",
        "candidate_rank",
        "candidate_attempt_count",
        "resampling_policy",
    }
    if not required <= set(source):
        raise BundleValidationError("Earth Engine source provenance is incomplete")
    allowed = {"provider", *required, *SOURCE_MEASUREMENT_FIELDS}
    if set(source) != allowed:
        raise BundleValidationError("Earth Engine source provenance contains unknown fields")
    if source.get("collection") != "COPERNICUS/S2_SR_HARMONIZED":
        raise BundleValidationError("source.collection is unsupported")
    if not isinstance(source.get("provider_scene_id"), str) or not source["provider_scene_id"]:
        raise BundleValidationError("source.provider_scene_id is invalid")
    if source.get("product_id") is not None and not isinstance(source["product_id"], str):
        raise BundleValidationError("source.product_id is invalid")
    _normalise_datetime(source.get("acquired_at"))
    bbox = source.get("requested_bbox_wgs84")
    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or any(not isinstance(item, (int, float)) or not math.isfinite(item) for item in bbox)
    ):
        raise BundleValidationError("source.requested_bbox_wgs84 is invalid")
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise BundleValidationError("source.requested_bbox_wgs84 is outside valid bounds")
    transform = source.get("source_transform")
    if (
        not isinstance(transform, list)
        or len(transform) != 6
        or any(not isinstance(item, (int, float)) or not math.isfinite(item) for item in transform)
    ):
        raise BundleValidationError("source.source_transform is invalid")
    if source.get("source_scale") != 10_000:
        raise BundleValidationError("source.source_scale is invalid")
    if not isinstance(source.get("source_crs"), str) or not source["source_crs"]:
        raise BundleValidationError("source.source_crs is invalid")
    if not isinstance(source.get("source_sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", source["source_sha256"]
    ):
        raise BundleValidationError("source.source_sha256 is invalid")
    if source.get("selection_policy") not in {"target_cloud_range", "least_cloudy"}:
        raise BundleValidationError("source.selection_policy is invalid")
    if (
        not isinstance(source.get("source_bytes"), int)
        or isinstance(source["source_bytes"], bool)
        or source["source_bytes"] <= 0
    ):
        raise BundleValidationError("source.source_bytes is invalid")
    if (
        not isinstance(source.get("candidate_rank"), int)
        or isinstance(source["candidate_rank"], bool)
        or source["candidate_rank"] < 1
    ):
        raise BundleValidationError("source.candidate_rank is invalid")
    if (
        not isinstance(source.get("candidate_attempt_count"), int)
        or isinstance(source["candidate_attempt_count"], bool)
        or not 1 <= source["candidate_attempt_count"] <= 5
    ):
        raise BundleValidationError("source.candidate_attempt_count is invalid")
    target = source.get("target_cloud_range")
    if source["selection_policy"] == "target_cloud_range":
        if not isinstance(target, dict) or set(target) != {
            "minimum_percent",
            "maximum_percent",
            "ideal_percent",
        }:
            raise BundleValidationError("source.target_cloud_range is invalid")
        minimum = _require_percentage(
            target["minimum_percent"], name="source.target_cloud_range.minimum_percent"
        )
        maximum = _require_percentage(
            target["maximum_percent"], name="source.target_cloud_range.maximum_percent"
        )
        ideal = _require_percentage(
            target["ideal_percent"], name="source.target_cloud_range.ideal_percent"
        )
        if not (5 <= minimum < maximum <= 50 and minimum <= ideal <= maximum):
            raise BundleValidationError("source.target_cloud_range is outside safe bounds")
    elif target is not None:
        raise BundleValidationError("least-cloudy source.target_cloud_range must be null")
    metadata_cloud = source.get("earth_engine_metadata_cloud_percentage")
    if metadata_cloud is not None:
        _require_percentage(metadata_cloud, name="source.earth_engine_metadata_cloud_percentage")
    if source.get("resampling_policy") != "earth_engine_default_nearest":
        raise BundleValidationError("source.resampling_policy is invalid")


def _validate_asset(
    root: Path,
    value: Any,
    *,
    name: str,
    expected_filename: str,
    expected_media_type: str,
    expected_format: str,
    expected_mode: str,
) -> tuple[int, int]:
    asset = _require_object(value, name=f"assets.{name}")
    href = asset.get("href")
    if href != expected_filename or Path(str(href)).name != href:
        raise BundleValidationError(f"assets.{name}.href must be {expected_filename}")
    if asset.get("media_type") != expected_media_type:
        raise BundleValidationError(f"assets.{name}.media_type is invalid")
    path = root / href
    if not path.is_file():
        raise BundleValidationError(f"Missing asset: {href}")
    size = path.stat().st_size
    if asset.get("bytes") != size:
        raise BundleValidationError(f"Byte count does not match {href}")
    expected_sha256 = asset.get("sha256")
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise BundleValidationError(f"assets.{name}.sha256 is invalid")
    if _sha256(path) != expected_sha256:
        raise BundleValidationError(f"Checksum does not match {href}")
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            if image.format != expected_format or image.mode != expected_mode:
                raise BundleValidationError(f"Image encoding does not match {href} metadata")
    except (OSError, SyntaxError) as error:
        raise BundleValidationError(f"Could not decode {href}") from error
    if asset.get("width") != width or asset.get("height") != height:
        raise BundleValidationError(f"Image dimensions do not match {href} metadata")
    return width, height


def validate_bundle(bundle_root: str | Path) -> ValidatedBundle:
    """Validate a complete three-file downlink without changing ground state."""
    root = Path(bundle_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    entries = {path.name for path in root.iterdir()}
    if entries != EXPECTED_FILES:
        raise BundleValidationError(
            f"Bundle must contain exactly {sorted(EXPECTED_FILES)}, found {sorted(entries)}"
        )
    metadata_path = root / "scene.json"
    try:
        manifest = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BundleValidationError("scene.json is not valid UTF-8 JSON") from error
    manifest = _require_object(manifest, name="scene.json")
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise BundleValidationError("Unsupported downlink schema_version")
    if manifest.get("product_type") != BUNDLE_PRODUCT_TYPE:
        raise BundleValidationError("Unsupported downlink product_type")
    _validate_no_absolute_paths(manifest)
    _validate_source(manifest.get("source"))

    scene_id = _require_identifier(manifest.get("scene_id"), name="scene_id")
    region_id = _require_identifier(manifest.get("region_id"), name="region_id")
    sensor = manifest.get("sensor")
    if sensor not in {"sentinel-2", "balkan-1"}:
        raise BundleValidationError("sensor is unsupported")
    acquired_at = _normalise_datetime(manifest.get("acquired_at"))
    status = manifest.get("status")
    if status not in {"MEASURED", "INSUFFICIENT_DATA"}:
        raise BundleValidationError("status is unsupported")

    assets = _require_object(manifest.get("assets"), name="assets")
    if set(assets) != {"rgb_preview", "condition_overlay"}:
        raise BundleValidationError("assets must contain only the two web images")
    rgb_size = _validate_asset(
        root,
        assets["rgb_preview"],
        name="rgb_preview",
        expected_filename="scene.webp",
        expected_media_type="image/webp",
        expected_format="WEBP",
        expected_mode="RGB",
    )
    overlay_size = _validate_asset(
        root,
        assets["condition_overlay"],
        name="condition_overlay",
        expected_filename="condition.png",
        expected_media_type="image/png",
        expected_format="PNG",
        expected_mode="RGBA",
    )
    if rgb_size != overlay_size:
        raise BundleValidationError("RGB and condition images must have matching dimensions")

    package = _require_object(manifest.get("package"), name="package")
    if package.get("file_count") != 3 or package.get("metadata_file") != "scene.json":
        raise BundleValidationError("package file contract is invalid")
    metadata_bytes = metadata_path.stat().st_size
    asset_bytes = sum((root / name).stat().st_size for name in ("scene.webp", "condition.png"))
    if package.get("metadata_bytes") != metadata_bytes:
        raise BundleValidationError("package metadata byte count is invalid")
    if package.get("asset_bytes") != asset_bytes:
        raise BundleValidationError("package asset byte count is invalid")
    if package.get("total_bytes") != metadata_bytes + asset_bytes:
        raise BundleValidationError("package total byte count is invalid")

    geospatial = _require_object(manifest.get("geospatial"), name="geospatial")
    bounds = geospatial.get("bounds_wgs84")
    if (
        not isinstance(bounds, list)
        or len(bounds) != 4
        or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in bounds)
    ):
        raise BundleValidationError("geospatial.bounds_wgs84 must contain four finite values")
    west, south, east, north = (float(value) for value in bounds)
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise BundleValidationError("geospatial.bounds_wgs84 is outside valid longitude/latitude")

    interaction = _require_object(manifest.get("interaction_grid"), name="interaction_grid")
    rows = interaction.get("rows")
    columns = interaction.get("columns")
    cells = interaction.get("cells")
    if (
        not isinstance(rows, int)
        or not isinstance(columns, int)
        or rows <= 0
        or columns <= 0
        or not isinstance(cells, list)
        or len(cells) != rows * columns
    ):
        raise BundleValidationError("interaction_grid dimensions do not match its cells")
    cell_ids = [cell.get("id") for cell in cells if isinstance(cell, dict)]
    if len(cell_ids) != len(cells) or len(set(cell_ids)) != len(cells):
        raise BundleValidationError("interaction_grid cell identifiers must be unique")

    condition = _require_object(manifest.get("condition"), name="condition")
    label = condition.get("label")
    if label not in CONDITION_LABELS:
        raise BundleValidationError("condition.label is unsupported")
    score = condition.get("condition_score")
    if score is not None and (
        not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 100
    ):
        raise BundleValidationError("condition.condition_score must be null or within 0..100")
    evidence_quality_score = _require_percentage(
        condition.get("evidence_quality_score"), name="condition.evidence_quality_score"
    )
    analysis_percentage = _require_percentage(
        condition.get("analysis_percentage"), name="condition.analysis_percentage"
    )
    return ValidatedBundle(
        root=root,
        manifest=manifest,
        manifest_sha256=_sha256(metadata_path),
        scene_id=scene_id,
        region_id=region_id,
        sensor=str(sensor),
        acquired_at=acquired_at,
        status=str(status),
        label=str(label),
        score=float(score) if score is not None else None,
        evidence_quality_score=evidence_quality_score,
        analysis_percentage=analysis_percentage,
        bounds_wgs84=(west, south, east, north),
    )


class SceneCatalog:
    """SQLite index and immutable file store for verified scene bundles."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.assets_root = self.root / "scenes"
        self.database_path = self.root / "catalog.sqlite3"
        self.assets_root.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS catalog_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS scenes (
                    scene_id TEXT PRIMARY KEY,
                    region_id TEXT NOT NULL,
                    sensor TEXT NOT NULL,
                    acquired_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    condition_label TEXT NOT NULL,
                    condition_score REAL,
                    evidence_quality_score REAL NOT NULL,
                    analysis_percentage REAL NOT NULL,
                    west REAL NOT NULL,
                    south REAL NOT NULL,
                    east REAL NOT NULL,
                    north REAL NOT NULL,
                    manifest_sha256 TEXT NOT NULL,
                    relative_bundle_path TEXT NOT NULL,
                    ingested_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_scenes_region_time
                    ON scenes(region_id, acquired_at DESC);
                """
            )
            existing = connection.execute(
                "SELECT value FROM catalog_metadata WHERE key = 'schema_version'"
            ).fetchone()
            if existing is None:
                connection.execute(
                    "INSERT INTO catalog_metadata(key, value) VALUES('schema_version', ?)",
                    (str(CATALOG_SCHEMA_VERSION),),
                )
            elif int(existing["value"]) != CATALOG_SCHEMA_VERSION:
                raise RuntimeError("Unsupported ground catalog schema version")

    @staticmethod
    def _summary(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "scene_id": row["scene_id"],
            "region_id": row["region_id"],
            "sensor": row["sensor"],
            "acquired_at": row["acquired_at"],
            "status": row["status"],
            "condition": {
                "label": row["condition_label"],
                "score": row["condition_score"],
                "evidence_quality_score": row["evidence_quality_score"],
                "analysis_percentage": row["analysis_percentage"],
            },
            "bounds_wgs84": [row["west"], row["south"], row["east"], row["north"]],
            "ingested_at": row["ingested_at"],
        }

    def ingest(self, bundle_root: str | Path) -> tuple[dict[str, Any], bool]:
        """Atomically ingest a valid bundle; identical repeats are idempotent."""
        bundle = validate_bundle(bundle_root)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM scenes WHERE scene_id = ?", (bundle.scene_id,)
            ).fetchone()
            if existing is not None:
                if existing["manifest_sha256"] != bundle.manifest_sha256:
                    raise SceneConflictError(
                        f"Scene {bundle.scene_id} already exists with different content"
                    )
                return self._summary(existing), False
            destination = self.assets_root / bundle.scene_id
            staging = self.assets_root / (
                f".{bundle.scene_id}.{bundle.manifest_sha256[:12]}.{uuid.uuid4().hex}.staging"
            )
            installed = False
            try:
                staging.mkdir()
                for filename in sorted(EXPECTED_FILES):
                    shutil.copy2(bundle.root / filename, staging / filename)
                validate_bundle(staging)
                if destination.exists():
                    raise SceneConflictError(f"Scene directory already exists: {bundle.scene_id}")
                os.replace(staging, destination)
                installed = True
                now = datetime.now(timezone.utc).isoformat()
                relative = destination.relative_to(self.root).as_posix()
                connection.execute(
                    """
                    INSERT INTO scenes(
                        scene_id, region_id, sensor, acquired_at, status,
                        condition_label, condition_score, evidence_quality_score,
                        analysis_percentage, west, south, east, north,
                        manifest_sha256, relative_bundle_path, ingested_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        bundle.scene_id,
                        bundle.region_id,
                        bundle.sensor,
                        bundle.acquired_at,
                        bundle.status,
                        bundle.label,
                        bundle.score,
                        bundle.evidence_quality_score,
                        bundle.analysis_percentage,
                        *bundle.bounds_wgs84,
                        bundle.manifest_sha256,
                        relative,
                        now,
                    ),
                )
            except Exception:
                if installed and destination.exists():
                    shutil.rmtree(destination)
                raise
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        record = self.get_scene(bundle.scene_id)
        if record is None:
            raise RuntimeError("Ingested scene could not be read back")
        return record, True

    def get_scene(self, scene_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM scenes WHERE scene_id = ?", (scene_id,)
            ).fetchone()
        return None if row is None else self._summary(row)

    def list_scenes(
        self,
        *,
        region_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        if not 1 <= limit <= 500 or offset < 0:
            raise ValueError("Invalid pagination")
        query = "SELECT * FROM scenes"
        parameters: list[Any] = []
        if region_id is not None:
            query += " WHERE region_id = ?"
            parameters.append(region_id)
        query += " ORDER BY acquired_at DESC, scene_id LIMIT ? OFFSET ?"
        parameters.extend((limit, offset))
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._summary(row) for row in rows]

    def list_regions(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT region_id, COUNT(*) AS scene_count,
                       MIN(acquired_at) AS first_acquired_at,
                       MAX(acquired_at) AS latest_acquired_at
                FROM scenes
                GROUP BY region_id
                ORDER BY region_id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def latest_for_region(self, region_id: str) -> dict[str, Any] | None:
        scenes = self.list_scenes(region_id=region_id, limit=1)
        return scenes[0] if scenes else None

    def history_for_region(self, region_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        scenes = self.list_scenes(region_id=region_id, limit=limit)
        return list(reversed(scenes))

    def manifest_path(self, scene_id: str) -> Path:
        path = self._bundle_path(scene_id) / "scene.json"
        if not path.is_file():
            raise FileNotFoundError(scene_id)
        return path

    def load_manifest(self, scene_id: str) -> dict[str, Any]:
        return json.loads(self.manifest_path(scene_id).read_text(encoding="utf-8"))

    def asset_path(self, scene_id: str, asset_name: str) -> Path:
        if asset_name not in {"scene.webp", "condition.png"}:
            raise ValueError("Unsupported scene asset")
        path = self._bundle_path(scene_id) / asset_name
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def _bundle_path(self, scene_id: str) -> Path:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT relative_bundle_path FROM scenes WHERE scene_id = ?", (scene_id,)
            ).fetchone()
        if row is None:
            raise FileNotFoundError(scene_id)
        path = (self.root / row["relative_bundle_path"]).resolve()
        if self.root not in path.parents:
            raise RuntimeError("Catalog contains an unsafe bundle path")
        return path
