from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import rasterio
from prithvi_payload.acquisition.earth_engine import EarthEngineAcquisitionProvider
from prithvi_payload.acquisition.errors import AcquisitionError
from prithvi_payload.crop_stage import DEFAULT_MAX_CLOUD_PERCENTAGE
from prithvi_payload.runtime import PayloadRuntime
from prithvi_shared import PayloadAcquisitionCommand

LIVE_ENABLED = os.environ.get("VITA_RUN_EE_INTEGRATION") == "1"
HAS_RUNTIME_CONFIG = bool(
    os.environ.get("EE_PROJECT_ID") and os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
)

pytestmark = pytest.mark.skipif(
    not (LIVE_ENABLED and HAS_RUNTIME_CONFIG),
    reason=(
        "live Earth Engine test requires VITA_RUN_EE_INTEGRATION=1, EE_PROJECT_ID "
        "and GOOGLE_APPLICATION_CREDENTIALS"
    ),
)


def test_live_cloud_demonstration_uses_existing_payload_pipeline(tmp_path: Path) -> None:
    command = PayloadAcquisitionCommand.model_validate(
        {
            "schema_version": "1.0",
            "job_id": "field_42_live_20260729",
            "region_id": "field_42",
            "source": {
                "provider": "earth_engine",
                "bbox_wgs84": [23.10, 42.50, 23.15, 42.55],
                "start_date": "2026-05-01",
                "end_date": "2026-07-29",
                "selection_policy": "target_cloud_range",
                "target_cloud_min_percent": 15,
                "target_cloud_max_percent": 35,
                "target_cloud_ideal_percent": 25,
            },
        }
    )
    provider = EarthEngineAcquisitionProvider(max_candidates=50, max_scene_attempts=5)
    runtime = PayloadRuntime(provider=provider, cuda_required=False)
    runtime.initialize()
    states: list[str] = []

    try:
        result = runtime.process(
            command,
            tmp_path / command.job_id,
            lambda state, _: states.append(state),
        )
    except AcquisitionError as error:
        if error.code == "PAYLOAD_NO_SCENE_IN_TARGET_CLOUD_RANGE":
            pytest.xfail(
                "Five metadata-qualified candidates were evaluated, but actual AOI cloud "
                "coverage was outside 15-35%; scientific thresholds were not changed"
            )
        raise

    assert 15 <= result["metadata_cloud_percentage"] <= 35
    assert 15 <= result["payload_cloud_percentage"] <= 35
    assert 0 < result["payload_cloud_percentage"] < DEFAULT_MAX_CLOUD_PERCENTAGE
    assert "evaluating_cloud" in states
    assert result["artifacts"] == ["scene.json", "scene.webp", "condition.png"]

    job_root = tmp_path / command.job_id
    downlink = job_root / result["artifact_directory"]
    assert {path.name for path in downlink.iterdir()} == {
        "scene.json",
        "scene.webp",
        "condition.png",
    }
    manifest = json.loads((downlink / "scene.json").read_text(encoding="utf-8"))
    assert (
        manifest["source"]["earth_engine_metadata_cloud_percentage"]
        == (result["metadata_cloud_percentage"])
    )
    assert (
        manifest["source"]["payload_measured_cloud_percentage"]
        == (result["payload_cloud_percentage"])
    )

    payload_root = downlink.parent
    unusable_path = next(payload_root.glob("cloud_masks/*_unusable.tif"))
    semantic_path = next(payload_root.glob("cloud_masks/*_semantic.tif"))
    crop_path = next(payload_root.glob("crop_maps/*_binary.tif"))
    valid_crop_path = next(payload_root.glob("condition_analysis/condition/*_valid_crop.tif"))
    with (
        rasterio.open(unusable_path) as unusable_source,
        rasterio.open(semantic_path) as semantic_source,
        rasterio.open(crop_path) as crop_source,
        rasterio.open(valid_crop_path) as valid_crop_source,
    ):
        unusable = unusable_source.read(1) == 1
        semantic = semantic_source.read(1)
        crop = crop_source.read(1)
        valid_crop = valid_crop_source.read(1)
    assert np.any(np.isin(semantic, (1, 2)))
    assert np.any(unusable)
    if result["payload_shadow_percentage"] > 0:
        assert np.any(semantic == 3)
    assert np.all(crop[unusable] == 255)
    assert np.all(valid_crop[unusable] == 0)
