from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import google.auth
import pytest
from prithvi_payload import pc_service


def test_pc_runtime_routes_stored_sentinel_through_operational_runtime(
    monkeypatch,
) -> None:
    runtime = pc_service.PCPayloadRuntime.__new__(pc_service.PCPayloadRuntime)
    captured = {}

    def fake_run(_runtime, request):
        captured["request"] = request
        return {"status": "DOWNLINK_READY", "sensor": request.sensor}

    monkeypatch.setattr(pc_service.PayloadRuntime, "run", fake_run)
    response = runtime.run(
        pc_service.PCJobRequest(
            sensor="sentinel-2",
            input="sentinel2",
            image="scene.tif",
            region_id="stored-field",
            job_id="pc-stored-test",
        )
    )

    assert response == {"status": "DOWNLINK_READY", "sensor": "sentinel-2"}
    assert isinstance(captured["request"], pc_service.JobRequest)
    assert captured["request"].image == "scene.tif"


@pytest.mark.parametrize(
    ("sensor", "expected_method"),
    [
        ("balkan-1-raw", "raw"),
        ("sentinel-2-live", "live"),
    ],
)
def test_pc_runtime_routes_extended_workflows_through_warm_raw_methods(
    sensor: str,
    expected_method: str,
    monkeypatch,
) -> None:
    runtime = pc_service.PCPayloadRuntime.__new__(pc_service.PCPayloadRuntime)
    captured = {}

    def fake_raw(request):
        captured["method"] = "raw"
        captured["request"] = request
        return {"status": "DOWNLINK_READY"}

    def fake_live(_runtime, request):
        captured["method"] = "live"
        captured["request"] = request
        return {
            "status": "DOWNLINK_READY",
            "pipeline_timing_seconds": {
                "total_acquisition_seconds": 6.5,
                "orchestration_seconds": 6.7,
            },
        }

    monkeypatch.setattr(runtime, "_run_raw_cached", fake_raw)
    monkeypatch.setattr(pc_service.RawPayloadRuntime, "_run_live_sentinel", fake_live)
    fields = {
        "sensor": sensor,
        "input": "3408" if sensor == "balkan-1-raw" else "earth-engine",
        "region_id": "selected-field",
        "job_id": "pc-extended-test",
    }
    if sensor == "sentinel-2-live":
        fields.update(
            {
                "bbox_wgs84": (5.43, 52.50, 5.49, 52.54),
                "start_date": date(2026, 6, 1),
                "end_date": date(2026, 7, 1),
            }
        )

    response = runtime.run(pc_service.PCJobRequest(**fields))

    assert response["status"] == "DOWNLINK_READY"
    assert captured["method"] == expected_method
    assert isinstance(captured["request"], pc_service.RawJobRequest)
    if sensor == "sentinel-2-live":
        assert response["pipeline_timing_seconds"]["orchestration_seconds"] == pytest.approx(
            0.2
        )


def test_pc_request_rejects_live_selection_without_dates() -> None:
    with pytest.raises(ValueError, match="requires an area and date range"):
        pc_service.PCJobRequest(
            sensor="sentinel-2-live",
            input="earth-engine",
            region_id="selected-field",
            job_id="pc-live-test",
        )


def test_pc_earth_engine_provider_scopes_service_account_credentials(
    tmp_path: Path,
    monkeypatch,
) -> None:
    credentials_path = tmp_path / "earth-engine.json"
    credentials_path.write_text("{}", encoding="utf-8")
    scoped_credential = object()
    captured = {}

    class Credential:
        requires_scopes = True

        def with_scopes(self, scopes):
            captured["scopes"] = scopes
            return scoped_credential

    def initialize(**kwargs):
        captured["initialize"] = kwargs

    scopes = ["earth-engine-scope", "cloud-platform-scope"]
    monkeypatch.setenv("VITA_EE_PROJECT", "vita-local-test")
    monkeypatch.setenv("VITA_EE_CREDENTIALS", str(credentials_path))
    monkeypatch.setattr(
        google.auth,
        "load_credentials_from_file",
        lambda _path: (Credential(), None),
    )
    monkeypatch.setitem(
        sys.modules,
        "ee",
        SimpleNamespace(Initialize=initialize, oauth=SimpleNamespace(SCOPES=scopes)),
    )

    pc_service.PCEarthEngineAcquisitionProvider().initialize()

    assert captured["scopes"] == scopes
    assert captured["initialize"] == {
        "credentials": scoped_credential,
        "project": "vita-local-test",
    }


def test_pc_raw_cache_is_content_addressed_and_validated(tmp_path: Path) -> None:
    inputs = {}
    for name in (
        "raw_path",
        "metadata_path",
        "radiometric_diagnostics_path",
        "position_path",
        "attitude_path",
        "parent_calibration_path",
    ):
        path = tmp_path / f"{name}.input"
        path.write_text(name, encoding="utf-8")
        inputs[name] = path
    cache_key, identity = pc_service._raw_cache_identity(inputs)
    artifacts_root = tmp_path / "artifacts"
    artifacts_root.mkdir()
    aligned = artifacts_root / "aligned.tif"
    alignment_report = artifacts_root / "aligned.alignment.json"
    proxy = artifacts_root / "proxy.tif"
    proxy_report = artifacts_root / "proxy.raw_proxy.json"
    proxy_calibration = artifacts_root / "proxy.crop_calibration.json"
    for path in (aligned, alignment_report, proxy, proxy_report, proxy_calibration):
        path.write_text(path.name, encoding="utf-8")
    cache_directory = tmp_path / "cache" / cache_key

    pc_service._publish_raw_cache(
        cache_directory,
        cache_key=cache_key,
        identity=identity,
        response={
            "artifacts": {
                "alignment": str(aligned),
                "alignment_report": str(alignment_report),
                "raw_proxy": str(proxy),
                "raw_proxy_report": str(proxy_report),
            }
        },
    )

    cached = pc_service._validated_raw_cache(cache_directory, cache_key=cache_key)
    assert cached is not None
    assert set(cached) == set(pc_service.PC_RAW_CACHE_ASSETS)
    cached["proxy"].write_text("corrupt", encoding="utf-8")
    assert pc_service._validated_raw_cache(cache_directory, cache_key=cache_key) is None
