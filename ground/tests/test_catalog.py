from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from prithvi_ground.catalog import (
    BundleValidationError,
    SceneCatalog,
    SceneConflictError,
    validate_bundle,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(path: Path, manifest: dict) -> None:
    metadata_bytes = -1
    while True:
        manifest["package"]["metadata_bytes"] = max(metadata_bytes, 0)
        manifest["package"]["total_bytes"] = manifest["package"]["asset_bytes"] + max(
            metadata_bytes, 0
        )
        encoded = (
            json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()
        if len(encoded) == metadata_bytes:
            path.write_bytes(encoded)
            return
        metadata_bytes = len(encoded)


def _build_bundle(
    root: Path,
    *,
    scene_id: str = "sentinel-scene-01",
    region_id: str = "demo-region",
    acquired_at: str = "2026-07-28T10:00:00+00:00",
    score: float = 68.5,
) -> Path:
    root.mkdir()
    rgb = np.zeros((12, 16, 3), dtype=np.uint8)
    rgb[..., 0] = 52
    rgb[..., 1] = 96
    rgb[..., 2] = 61
    overlay = np.zeros((12, 16, 4), dtype=np.uint8)
    overlay[2:10, 3:14] = (145, 207, 96, 205)
    Image.fromarray(rgb).save(root / "scene.webp", format="WEBP", quality=82, method=6)
    Image.fromarray(overlay).save(root / "condition.png", format="PNG", optimize=True)
    assets = {
        "rgb_preview": {
            "href": "scene.webp",
            "media_type": "image/webp",
            "width": 16,
            "height": 12,
            "bytes": (root / "scene.webp").stat().st_size,
            "sha256": _sha256(root / "scene.webp"),
        },
        "condition_overlay": {
            "href": "condition.png",
            "media_type": "image/png",
            "width": 16,
            "height": 12,
            "bytes": (root / "condition.png").stat().st_size,
            "sha256": _sha256(root / "condition.png"),
        },
    }
    asset_bytes = sum(asset["bytes"] for asset in assets.values())
    manifest = {
        "schema_version": "1.0",
        "product_type": "vita.crop-condition.web-bundle",
        "algorithm_version": "compact-downlink-v1",
        "scene_id": scene_id,
        "region_id": region_id,
        "sensor": "sentinel-2",
        "acquired_at": acquired_at,
        "status": "MEASURED",
        "claim": "relative crop-condition screening; not an agronomic diagnosis",
        "source": {
            "provider": "local_file",
            "acquired_at": acquired_at,
            "payload_measured_thick_cloud_percentage": 12.0,
            "payload_measured_thin_cloud_percentage": 3.0,
            "payload_measured_shadow_percentage": 4.0,
            "payload_measured_cloud_percentage": 15.0,
            "payload_measured_unusable_percentage": 20.0,
        },
        "assets": assets,
        "geospatial": {
            "bounds_wgs84": [23.0, 42.0, 23.2, 42.1],
            "native_crs": "EPSG:32634",
            "source_width": 160,
            "source_height": 120,
        },
        "quality": {"usable_percentage": 80.0, "analysis_percentage": 35.0},
        "metrics": {
            "ndvi": {"mean": 0.63, "median": 0.65},
            "gndvi": {"mean": 0.58, "median": 0.60},
        },
        "condition": {
            "status": "MEASURED",
            "label": "Watch",
            "condition_score": score,
            "evidence_quality_score": 82.0,
            "analysis_pixels": 5600,
            "analysis_percentage": 35.0,
            "explanations": ["Spectral vigor is below the nominal prototype range."],
            "limitations": ["Not an agronomic diagnosis."],
        },
        "interaction_grid": {
            "rows": 1,
            "columns": 1,
            "cell_order": "row-major",
            "cells": [
                {
                    "id": "r00c00",
                    "row": 0,
                    "column": 0,
                    "bounds_wgs84": [23.0, 42.0, 23.2, 42.1],
                    "condition_score": score,
                    "alert_percentage": 4.2,
                    "metrics": {"ndvi": 0.65, "gndvi": 0.60, "evi": 0.45, "savi": 0.52},
                    "quality": {"analysis_percentage": 35.0, "unusable_percentage": 20.0},
                }
            ],
        },
        "legend": {"condition_gradient": [], "classes": {}, "priority": []},
        "processing": {"condition_algorithm_version": "spectral-condition-v1"},
        "limitations": ["Not an agronomic diagnosis."],
        "warnings": [],
        "package": {
            "file_count": 3,
            "metadata_file": "scene.json",
            "asset_bytes": asset_bytes,
            "metadata_bytes": 0,
            "total_bytes": asset_bytes,
        },
    }
    _write_manifest(root / "scene.json", manifest)
    return root


def test_validate_and_ingest_bundle_atomically(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path / "bundle")
    validated = validate_bundle(bundle)

    assert validated.scene_id == "sentinel-scene-01"
    assert validated.label == "Watch"
    assert validated.score == pytest.approx(68.5)

    catalog = SceneCatalog(tmp_path / "store")
    record, created = catalog.ingest(bundle)
    repeated, repeated_created = catalog.ingest(bundle)

    assert created is True
    assert repeated_created is False
    assert repeated == record
    assert record["condition"]["score"] == pytest.approx(68.5)
    assert catalog.get_scene("sentinel-scene-01") == record
    assert catalog.load_manifest("sentinel-scene-01")["scene_id"] == "sentinel-scene-01"
    assert catalog.asset_path("sentinel-scene-01", "scene.webp").is_file()
    stored_names = sorted(
        path.name for path in (tmp_path / "store/scenes/sentinel-scene-01").iterdir()
    )
    assert stored_names == [
        "condition.png",
        "scene.json",
        "scene.webp",
    ]


def test_catalog_lists_regions_and_chronological_history(tmp_path: Path) -> None:
    catalog = SceneCatalog(tmp_path / "store")
    catalog.ingest(
        _build_bundle(
            tmp_path / "first",
            scene_id="scene-first",
            acquired_at="2026-07-27T10:00:00Z",
            score=42.0,
        )
    )
    catalog.ingest(
        _build_bundle(
            tmp_path / "second",
            scene_id="scene-second",
            acquired_at="2026-07-28T10:00:00Z",
            score=68.0,
        )
    )

    regions = catalog.list_regions()
    history = catalog.history_for_region("demo-region")

    assert regions[0]["scene_count"] == 2
    assert catalog.latest_for_region("demo-region")["scene_id"] == "scene-second"
    assert [scene["scene_id"] for scene in history] == ["scene-first", "scene-second"]
    assert [scene["condition"]["score"] for scene in history] == [42.0, 68.0]


def test_catalog_rejects_checksum_tampering(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path / "bundle")
    (bundle / "condition.png").write_bytes((bundle / "condition.png").read_bytes() + b"tamper")

    with pytest.raises(BundleValidationError, match="Byte count|Checksum"):
        validate_bundle(bundle)


def test_catalog_rejects_extra_files_and_absolute_paths(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path / "bundle")
    (bundle / "extra.txt").write_text("not part of the contract")
    with pytest.raises(BundleValidationError, match="exactly"):
        validate_bundle(bundle)
    (bundle / "extra.txt").unlink()

    manifest = json.loads((bundle / "scene.json").read_text())
    manifest["warnings"] = [r"C:\unsafe\source.tif"]
    _write_manifest(bundle / "scene.json", manifest)
    with pytest.raises(BundleValidationError, match="absolute paths"):
        validate_bundle(bundle)


def test_catalog_rejects_conflicting_scene_identity(tmp_path: Path) -> None:
    catalog = SceneCatalog(tmp_path / "store")
    catalog.ingest(_build_bundle(tmp_path / "first", score=40.0))
    conflicting = _build_bundle(tmp_path / "second", score=90.0)

    with pytest.raises(SceneConflictError, match="different content"):
        catalog.ingest(conflicting)


def test_catalog_rejects_malformed_interaction_grid(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path / "bundle")
    manifest = json.loads((bundle / "scene.json").read_text())
    manifest["interaction_grid"]["rows"] = 2
    _write_manifest(bundle / "scene.json", manifest)

    with pytest.raises(BundleValidationError, match="dimensions"):
        validate_bundle(bundle)


def test_concurrent_identical_ingest_is_idempotent(tmp_path: Path) -> None:
    bundle = _build_bundle(tmp_path / "bundle")
    catalog = SceneCatalog(tmp_path / "store")

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: catalog.ingest(bundle), range(2)))

    assert sorted(created for _record, created in results) == [False, True]
    assert results[0][0] == results[1][0]
    assert len(catalog.list_scenes()) == 1
