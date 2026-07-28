from __future__ import annotations

import json
from pathlib import Path

from cloud_detection.backend import TestBackend
from cloud_detection.cli import DEFAULT_CONFIG
from cloud_detection.config import load_config
from fastapi.testclient import TestClient
from prithvi_ground.api import create_app
from test_pipeline import AllCloudBackend, FakeCropModel, _write_sentinel_scene
from vita_integration.mvp import run_sentinel_mvp


def test_sentinel_mvp_reaches_verified_api_and_web_client(tmp_path: Path) -> None:
    source = tmp_path / "sentinel.tif"
    _write_sentinel_scene(source)
    output = tmp_path / "run"
    ground_store = tmp_path / "ground"

    result = run_sentinel_mvp(
        source,
        output_root=output,
        ground_store=ground_store,
        acquired_at="2026-07-28T12:00:00Z",
        reflectance_scale=10_000,
        cloud_backend=TestBackend(),
        cloud_config=load_config(DEFAULT_CONFIG),
        crop_model=FakeCropModel(),
        condition_tile_size=16,
        downlink_max_image_dimension=64,
        downlink_grid_size=4,
    )

    assert result["status"] == "MVP_READY"
    assert result["completed_stages"] == [
        "payload",
        "downlink",
        "ground_catalog",
        "api_ready",
    ]
    assert result["ground"]["created"] is True
    assert result["ground"]["scene"]["scene_id"] == "sentinel"
    assert result["ground"]["links"]["application"] == "/"
    assert result["ground"]["links"]["openapi"] == "/docs"
    assert json.loads((output / "mvp_result.json").read_text()) == result

    with TestClient(create_app(ground_store)) as client:
        assert client.get("/").status_code == 200
        scene = client.get("/api/v1/scenes/sentinel")
        assert scene.status_code == 200
        assert scene.json()["condition"]["label"] == "Watch"
        assert client.get("/api/v1/scenes/sentinel/manifest").status_code == 200
        assert client.get("/api/v1/scenes/sentinel/preview").headers[
            "content-type"
        ] == "image/webp"
        assert client.get("/api/v1/scenes/sentinel/condition-overlay").headers[
            "content-type"
        ] == "image/png"


def test_sentinel_mvp_does_not_ingest_cloud_rejected_scene(tmp_path: Path) -> None:
    source = tmp_path / "sentinel.tif"
    _write_sentinel_scene(source)
    output = tmp_path / "run"
    ground_store = tmp_path / "ground"

    result = run_sentinel_mvp(
        source,
        output_root=output,
        ground_store=ground_store,
        acquired_at="2026-07-28T12:00:00Z",
        reflectance_scale=10_000,
        cloud_backend=AllCloudBackend(),
        cloud_config=load_config(DEFAULT_CONFIG),
    )

    assert result["status"] == "PAYLOAD_STOPPED"
    assert result["completed_stages"] == ["payload"]
    assert result["ground"] is None
    assert not ground_store.exists()
