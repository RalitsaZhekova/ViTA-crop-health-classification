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
        return {"available": True, "reason": None, "sensors": ["sentinel-2", "balkan-1"]}

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
    assert 'href="/static/styles.css?v=region-listbox-1"' in page.text
    assert 'src="/static/app.js?v=region-listbox-1"' in page.text
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

    styles = client.get("/static/styles.css?v=region-listbox-1")
    assert styles.status_code == 200
    assert 'html[data-theme="dark"] .region-menu' in styles.text
    assert "background: #0c1a14" in styles.text
    assert 'html[data-theme="dark"] .region-option.selected' in styles.text

    app_script = client.get("/static/app.js?v=region-listbox-1")
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
