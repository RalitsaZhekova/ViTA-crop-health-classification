from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from prithvi_ground.api import create_app
from prithvi_ground.pipeline_runs import (
    PipelineLaunch,
    PipelineLaunchError,
    PipelineRunManager,
)


class StubPipelineManager:
    def __init__(self) -> None:
        self.run = None

    def capability(self):
        return {
            "available": True,
            "reason": None,
            "sensors": [
                "sentinel-2",
                "sentinel-2-live",
                "balkan-1",
                "balkan-1-raw",
            ],
        }

    def current(self):
        return self.run

    def start(self, launch):
        self.run = {
            "run_id": "web-sentinel-test",
            "scene_id": "web-sentinel-test",
            "sensor": launch.sensor,
            "region_id": launch.region_id,
            "status": "queued",
        }
        return self.run


def test_web_application_exposes_client_dashboard_and_safe_run_api(tmp_path: Path) -> None:
    manager = StubPipelineManager()
    client = TestClient(create_app(tmp_path / "store", pipeline_manager=manager))

    page = client.get("/")
    assert page.status_code == 200
    assert "Satellite image &amp; crop health" in page.text
    assert "Crop condition history" in page.text
    assert "Run crop analysis" in page.text
    assert 'id="theme-toggle"' in page.text
    assert 'src="/static/theme-init.js"' in page.text
    assert 'href="/static/styles.css?v=live-area-1"' in page.text
    assert 'src="/static/app.js?v=live-area-1"' in page.text
    assert 'src="/static/vendor/leaflet/leaflet.js?v=1.9.4"' in page.text
    assert 'id="region-trigger" class="region-trigger"' in page.text
    assert 'id="region-menu" class="region-menu hidden" role="listbox"' in page.text
    assert 'class="history-card history-panel card"' in page.text
    assert 'class="area-card health-breakdown card"' in page.text
    assert 'id="baseline-card" class="baseline-card signal-building"' in page.text
    assert "Selected condition" in page.text
    assert page.text.index('class="viewer-card card"') < page.text.index(
        'class="history-card history-panel card"'
    ) < page.text.index('class="area-card health-breakdown card"')
    assert page.text.index('class="trajectory-heading"') < page.text.index(
        'class="history-insights"'
    )
    assert '<option value="balkan-1-raw">Balkan-1 raw</option>' in page.text
    assert '<option value="sentinel-2-live">Sentinel-2' in page.text
    assert 'id="live-area-map"' in page.text
    assert 'id="live-start-date"' in page.text
    assert "Reuse a region name to append history" in page.text
    assert (
        '<link rel="icon" type="image/png" '
        'href="/static/ViTA_satellite_icon.png?v=2">'
    ) in page.text
    assert 'class="brand-logo brand-logo-light" src="/static/ViTA_logo_green.png"' in page.text
    assert 'class="brand-logo brand-logo-dark" src="/static/ViTA_logo_white.png"' in page.text

    green_logo = client.get("/static/ViTA_logo_green.png")
    assert green_logo.status_code == 200
    assert green_logo.headers["content-type"] == "image/png"

    logo = client.get("/static/ViTA_logo_white.png")
    assert logo.status_code == 200
    assert logo.headers["content-type"] == "image/png"

    icon = client.get("/static/ViTA_satellite_icon.png")
    assert icon.status_code == 200
    assert icon.headers["content-type"] == "image/png"

    styles = client.get("/static/styles.css?v=live-area-1")
    assert styles.status_code == 200
    assert 'html[data-theme="dark"] .region-menu' in styles.text
    assert "background: #0c1a14" in styles.text
    assert 'html[data-theme="dark"] .region-option.selected' in styles.text
    assert ".live-area-map" in styles.text
    assert 'html[data-theme="dark"] .live-area-map .leaflet-tile-pane' in styles.text

    app_script = client.get("/static/app.js?v=live-area-1")
    assert app_script.status_code == 200
    assert "isBaselineVigorScore" in app_script.text
    assert "first eligible" in app_script.text
    assert "Crop-condition trajectory from earlier to later observations" in app_script.text
    assert "renderSummaryResult(result, selected)" in app_script.text
    assert "point.date.valueOf() < selected.date.valueOf()" in app_script.text
    assert "Strong improvement" in app_script.text
    assert "Early review signal" in app_script.text
    assert "setRegionMenuOpen" in app_script.text
    assert 'event.key === "ArrowDown"' in app_script.text
    assert 'sensor === "balkan-1-raw"' in app_script.text
    assert "Raw detector imagery" in app_script.text
    assert "Pixel ${Math.round(point.x)}" in app_script.text
    assert "Use 3370, 3408, or 3458" in app_script.text
    assert "balkan1/preprocessed/3370_L1ORT.tif" in app_script.text
    assert "Leave blank for Flevoland" in app_script.text
    assert "initializeLiveAreaMap" in app_script.text
    assert 'payload.bbox_wgs84 = state.liveAreaBounds' in app_script.text

    leaflet = client.get("/static/vendor/leaflet/leaflet.js?v=1.9.4")
    assert leaflet.status_code == 200
    assert "Leaflet 1.9.4" in leaflet.text

    response = client.post(
        "/api/v1/pipeline-runs",
        json={
            "sensor": "sentinel-2",
            "region_id": "north-field",
            "input_path": "sentinel2",
            "image": "scene.tif",
        },
    )
    assert response.status_code == 202
    assert response.json()["run"]["region_id"] == "north-field"
    assert client.get("/api/v1/pipeline-runs/current").json()["run"]["status"] == "queued"


def test_web_run_api_rejects_cross_origin_launch(tmp_path: Path) -> None:
    client = TestClient(
        create_app(tmp_path / "store", pipeline_manager=StubPipelineManager())
    )

    response = client.post(
        "/api/v1/pipeline-runs",
        json={"sensor": "balkan-1", "region_id": "field-1"},
        headers={"Origin": "https://untrusted.example", "Host": "dashboard.local"},
    )

    assert response.status_code == 403


def test_web_run_api_reports_invalid_launch_as_unprocessable(tmp_path: Path) -> None:
    client = TestClient(
        create_app(tmp_path / "store", pipeline_manager=StubPipelineManager())
    )

    response = client.post(
        "/api/v1/pipeline-runs",
        json={"sensor": "sentinel-2", "region_id": "../field"},
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        {"sensor": "unknown", "region_id": "field-1"},
        {"sensor": "sentinel-2", "region_id": "../field"},
        {"sensor": "sentinel-2", "region_id": "field-1", "input_path": "../secret"},
        {"sensor": "sentinel-2", "region_id": "field-1", "image": "folder/scene.tif"},
        {"sensor": "balkan-1", "region_id": "field-1", "image": "scene.tif"},
        {"sensor": "balkan-1-raw", "region_id": "field-1", "input_path": "raw/3408"},
        {"sensor": "balkan-1-raw", "region_id": "field-1", "image": "scene.tif"},
        {"sensor": "sentinel-2-live", "region_id": "field-1"},
        {
            "sensor": "sentinel-2-live",
            "region_id": "field-1",
            "input_path": "sentinel2",
            "bbox_wgs84": [5.43, 52.50, 5.49, 52.54],
            "start_date": "2026-06-01",
            "end_date": "2026-07-01",
        },
        {
            "sensor": "sentinel-2-live",
            "region_id": "field-1",
            "bbox_wgs84": [5.0, 52.0, 6.0, 53.0],
            "start_date": "2026-06-01",
            "end_date": "2026-07-01",
        },
        {
            "sensor": "sentinel-2-live",
            "region_id": "field-1",
            "bbox_wgs84": [5.43, 52.50, 5.49, 52.54],
            "start_date": "2026-01-01",
            "end_date": "2026-07-01",
        },
        {
            "sensor": "sentinel-2",
            "region_id": "field-1",
            "bbox_wgs84": [5.43, 52.50, 5.49, 52.54],
            "start_date": "2026-06-01",
            "end_date": "2026-07-01",
        },
    ],
)
def test_pipeline_launch_rejects_unsafe_or_incompatible_input(payload) -> None:
    with pytest.raises(PipelineLaunchError):
        PipelineLaunch.from_payload(payload)


def test_pipeline_manager_invokes_existing_entry_point_without_a_shell(
    tmp_path: Path,
    monkeypatch,
) -> None:
    script = tmp_path / "vita.ps1"
    script.write_text("# test entry point\n", encoding="utf-8")
    manager = PipelineRunManager(tmp_path)
    monkeypatch.setattr(manager, "_powershell_executable", lambda: "powershell.exe")
    finished = threading.Event()
    captured = {}

    def fake_run(arguments, **kwargs):
        captured["arguments"] = arguments
        captured["kwargs"] = kwargs
        finished.set()
        return SimpleNamespace(
            returncode=0,
            stdout="PAYLOAD TOTAL                       1.2345 s\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    started = manager.start(
        PipelineLaunch(
            sensor="sentinel-2",
            region_id="north-field",
            input_path="sentinel2",
            image="scene.tif",
        )
    )
    assert finished.wait(timeout=2)
    completed = manager.current()

    assert started["status"] in {"queued", "running"}
    assert completed is not None
    assert completed["status"] == "succeeded"
    assert completed["payload_seconds"] == 1.2345
    assert captured["kwargs"]["cwd"] == tmp_path.resolve()
    assert "-File" in captured["arguments"]
    assert str(script.resolve()) in captured["arguments"]
    assert captured["arguments"][-4:] == [
        "-InputPath",
        "sentinel2",
        "-Image",
        "scene.tif",
    ]


def test_raw_pipeline_launch_uses_separate_command_when_jetson_is_configured(
    tmp_path: Path,
    monkeypatch,
) -> None:
    script = tmp_path / "vita.ps1"
    script.write_text("# test entry point\n", encoding="utf-8")
    manager = PipelineRunManager(tmp_path)
    monkeypatch.setattr(manager, "_powershell_executable", lambda: "powershell.exe")
    monkeypatch.setenv("VITA_RAW_SSH_TARGET", "payload@jetson.local")

    launch = PipelineLaunch.from_payload(
        {
            "sensor": "balkan-1-raw",
            "region_id": "raw-field",
            "input_path": "3408",
        }
    )
    command = manager._command("web-raw-test", launch)

    assert "balkan-1-raw" in manager.capability()["sensors"]
    assert command[command.index(str(script.resolve())) + 1] == "raw"
    assert command[-2:] == ["-InputPath", "3408"]


def test_live_sentinel_launch_uses_only_bounded_area_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    script = tmp_path / "vita.ps1"
    script.write_text("# test entry point\n", encoding="utf-8")
    manager = PipelineRunManager(tmp_path)
    monkeypatch.setattr(manager, "_powershell_executable", lambda: "powershell.exe")
    monkeypatch.setenv("VITA_JETSON_SSH_TARGET", "payload@jetson.local")

    launch = PipelineLaunch.from_payload(
        {
            "sensor": "sentinel-2-live",
            "region_id": "selected-farm",
            "bbox_wgs84": [5.43, 52.50, 5.49, 52.54],
            "start_date": "2026-06-01",
            "end_date": "2026-07-01",
        }
    )
    command = manager._command("web-live-test", launch)

    assert "sentinel-2-live" in manager.capability()["sensors"]
    assert command[command.index(str(script.resolve())) + 1] == "earth-engine"
    assert "-InputPath" not in command
    assert "-Image" not in command
    assert command[-6:] == [
        "-BboxWgs84",
        "5.43000000,52.50000000,5.49000000,52.54000000",
        "-StartDate",
        "2026-06-01",
        "-EndDate",
        "2026-07-01",
    ]
