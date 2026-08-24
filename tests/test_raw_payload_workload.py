from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from prithvi_payload import raw_payload_workload


def _input_files(root: Path) -> dict[str, Path]:
    paths = {
        "raw_path": root / "3408_Raw.tif",
        "metadata_path": root / "3408_L0R_manifest.json",
        "radiometric_diagnostics_path": root / "3408_L1A_reference_validation.json",
        "position_path": root / "position.csv",
        "attitude_path": root / "attitude.csv",
        "parent_calibration_path": root / "3408_L1ORT.crop_calibration.json",
    }
    for path in paths.values():
        path.write_text("input", encoding="utf-8")
    return paths


def test_isolated_raw_job_creates_normal_downlink_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _input_files(tmp_path)
    observed: dict[str, object] = {}

    def fake_align(source: Path, output: Path, **kwargs):
        observed["alignment"] = kwargs
        output.write_bytes(b"aligned")
        report = {
            "source": {"sha256": "a" * 64},
            "execution": {
                "resolved_device": "cuda",
                "cuda_batched_phase_correlation": True,
                "cuda_fused_four_band_warp": True,
            },
        }
        output.with_suffix(".alignment.json").write_text(
            json.dumps(report),
            encoding="utf-8",
        )
        return report

    def fake_proxy(aligned: Path, output: Path, **kwargs):
        observed["proxy"] = kwargs
        output.write_bytes(b"proxy")
        output.with_suffix(".raw_proxy.json").write_text("{}", encoding="utf-8")
        output.with_name(f"{output.stem}.crop_calibration.json").write_text(
            json.dumps({"acquired_at": "2026-06-16T12:00:00+00:00"}),
            encoding="utf-8",
        )
        return {
            "execution": {
                "reconstruction_backend": "gdal-average-band-parallel",
                "band_workers": 4,
                "cpu_thread_budget": 8,
            }
        }

    fake_cloud = SimpleNamespace(backend=object(), config={})
    fake_crop = object()

    def fake_models():
        return fake_cloud, fake_crop, {
            "cloud_backend": "tensorrt",
            "crop_backend": "tensorrt",
        }

    def fake_pipeline(source: Path, **kwargs):
        observed["pipeline_source"] = source
        observed["pipeline"] = kwargs
        bundle = kwargs["output_root"] / "downlink"
        bundle.mkdir()
        for name in raw_payload_workload.DOWNLINK_FILES:
            (bundle / name).write_bytes(name.encode("ascii"))
        return {
            "status": "DOWNLINK_READY",
            "scene_id": kwargs["scene_id"],
            "summary": {"crop": {"status": "ready"}},
        }

    monkeypatch.setattr(raw_payload_workload, "align_balkan_geotiff", fake_align)
    monkeypatch.setattr(raw_payload_workload, "build_raw_model_proxy", fake_proxy)
    monkeypatch.setattr(
        raw_payload_workload,
        "_load_warm_accelerated_models",
        fake_models,
    )
    monkeypatch.setattr(raw_payload_workload, "run_scene", fake_pipeline)
    monkeypatch.setattr(raw_payload_workload, "_synchronize_cuda", lambda: None)

    output_root = tmp_path / "isolated-runtime" / "runs"
    response = raw_payload_workload.run_raw_payload_job(
        **inputs,
        output_root=output_root,
        job_id="raw-3408-test",
        region_id="balkan-raw-3408",
    )

    assert response["status"] == "DOWNLINK_READY"
    assert response["bundle_relative"] == "runs/raw-3408-test/downlink"
    assert response["acceleration"]["alignment_device"] == "cuda"
    assert response["acceleration"]["cloud_backend"] == "tensorrt"
    assert response["acceleration"]["crop_backend"] == "tensorrt"
    assert response["acceleration"]["raw_model_reconstruction_band_workers"] == 4
    assert observed["alignment"]["config"].device == "cuda"
    assert observed["alignment"]["config"].build_overviews is False
    assert observed["alignment"]["require_georeferencing"] is False
    assert observed["pipeline"]["allow_experimental_raw_proxy"] is True
    assert observed["pipeline"]["reflectance_scale"] == 1.0
    assert observed["pipeline"]["stop_after"] == "downlink"
    assert observed["pipeline"]["acquisition_metadata"] == {
        "provider": "raw_local",
        "raw_payload_job": "raw-3408-test",
        "raw_source_sha256": "a" * 64,
        "alignment_report": "3408_Raw_aligned.alignment.json",
        "raw_proxy_report": "3408_Raw_model_proxy.raw_proxy.json",
    }
    assert (output_root / "raw-3408-test" / "downlink" / "scene.json").is_file()
    status = json.loads(
        (output_root / "raw-3408-test" / "raw-job.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "DOWNLINK_READY"

    with pytest.raises(FileExistsError, match="use a new job ID"):
        raw_payload_workload.run_raw_payload_job(
            **inputs,
            output_root=output_root,
            job_id="raw-3408-test",
            region_id="balkan-raw-3408",
        )


def test_isolated_raw_job_reuses_validated_preprocessed_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _input_files(tmp_path)
    prepared_root = tmp_path / "prepared"
    prepared_root.mkdir()
    prepared = {
        "aligned": prepared_root / "aligned.tif",
        "alignment_report": prepared_root / "alignment.json",
        "proxy": prepared_root / "proxy.tif",
        "proxy_report": prepared_root / "proxy.json",
        "proxy_calibration": prepared_root / "proxy.crop_calibration.json",
    }
    prepared["aligned"].write_bytes(b"aligned")
    prepared["alignment_report"].write_text(
        json.dumps(
            {
                "source": {"sha256": "a" * 64},
                "execution": {
                    "resolved_device": "cuda",
                    "cuda_batched_phase_correlation": True,
                    "cuda_fused_four_band_warp": True,
                },
            }
        ),
        encoding="utf-8",
    )
    prepared["proxy"].write_bytes(b"proxy")
    prepared["proxy_report"].write_text(
        json.dumps(
            {
                "execution": {
                    "reconstruction_backend": "gdal-average-band-parallel",
                    "band_workers": 2,
                    "cpu_thread_budget": 8,
                }
            }
        ),
        encoding="utf-8",
    )
    prepared["proxy_calibration"].write_text(
        json.dumps({"acquired_at": "2026-06-16T12:00:00+00:00"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        raw_payload_workload,
        "align_balkan_geotiff",
        lambda *_args, **_kwargs: pytest.fail("alignment must be reused"),
    )
    monkeypatch.setattr(
        raw_payload_workload,
        "build_raw_model_proxy",
        lambda *_args, **_kwargs: pytest.fail("proxy must be reused"),
    )
    monkeypatch.setattr(raw_payload_workload, "_synchronize_cuda", lambda: None)

    def fake_pipeline(_source: Path, **kwargs):
        bundle = kwargs["output_root"] / "downlink"
        bundle.mkdir()
        for name in raw_payload_workload.DOWNLINK_FILES:
            (bundle / name).write_bytes(name.encode("ascii"))
        return {"status": "DOWNLINK_READY", "scene_id": kwargs["scene_id"], "summary": {}}

    monkeypatch.setattr(raw_payload_workload, "run_scene", fake_pipeline)
    response = raw_payload_workload.run_raw_payload_job(
        **inputs,
        output_root=tmp_path / "runs",
        job_id="raw-cache-hit",
        region_id="raw-cache-field",
        preloaded_cloud=SimpleNamespace(backend=object(), config={}),
        preloaded_crop=object(),
        preloaded_acceleration={"cloud_backend": "pytorch", "crop_backend": "pytorch"},
        preprocessed_paths=prepared,
    )

    assert response["status"] == "DOWNLINK_READY"
    assert response["acceleration"]["raw_preprocess_cache_hit"] is True
    assert response["timing_seconds"]["cuda_alignment_seconds"] == 0.0
    assert response["timing_seconds"]["raw_model_reconstruction_seconds"] == 0.0
    assert response["acceleration"]["raw_model_reconstruction_band_workers"] == 2


def test_cuda_alignment_contract_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="required CUDA execution contract"):
        raw_payload_workload._assert_cuda_alignment(
            {
                "execution": {
                    "resolved_device": "cpu",
                    "cuda_batched_phase_correlation": False,
                    "cuda_fused_four_band_warp": False,
                }
            }
        )


def test_raw_compose_cannot_replace_operational_payload() -> None:
    project = Path(__file__).resolve().parents[1]
    raw_compose = yaml.safe_load(
        (project / "deploy" / "compose.payload.raw.yaml").read_text(encoding="utf-8")
    )
    operational_text = (project / "deploy" / "compose.payload.yaml").read_text(
        encoding="utf-8"
    )
    raw_text = (project / "deploy" / "compose.payload.raw.yaml").read_text(
        encoding="utf-8"
    )
    runner = (project / "deploy" / "payload" / "run-raw.sh").read_text(
        encoding="utf-8"
    )

    assert raw_compose["name"] == "vita-payload-raw"
    service = raw_compose["services"]["raw-payload"]
    assert "ports" not in service
    assert service["restart"] == "no"
    assert any("VITA_RAW_OUTPUT_HOST" in volume for volume in service["volumes"])
    assert any(
        "VITA_RAW_ENGINE_CACHE_HOST" in volume and volume.endswith(":/engine-cache:ro")
        for volume in service["volumes"]
    )
    assert any(
        "VITA_RAW_DATA_HOST" in volume and volume.endswith(":/data:ro")
        for volume in service["volumes"]
    )
    assert "../runtime/payload:/runtime" not in raw_text
    assert "vita-payload:1.0.0" in raw_text
    assert "runtime/raw-payload" not in operational_text
    assert "$OPERATIONAL_ROOT/runtime/worktrees/raw-band-jetson-isolated" in runner
    assert "$OPERATIONAL_ROOT/data/raw-inputs" in runner
    assert "$OPERATIONAL_ROOT/runtime/raw-payload" in runner
    assert "docker compose down" not in runner
    assert "docker compose stop" not in runner


def test_warm_raw_service_is_separate_and_uses_read_only_operational_assets() -> None:
    project = Path(__file__).resolve().parents[1]
    compose = yaml.safe_load(
        (project / "deploy" / "compose.payload.raw-service.yaml").read_text(
            encoding="utf-8"
        )
    )
    service = compose["services"]["raw-payload"]

    assert compose["name"] == "vita-payload-raw-service"
    assert service["restart"] == "unless-stopped"
    assert service["build"]["args"]["VITA_INSTALL_EARTH_ENGINE"] == "1"
    assert service["ports"] == ["127.0.0.1:${VITA_RAW_PAYLOAD_PORT:-8091}:8091"]
    assert any(volume.endswith(":/data:ro") for volume in service["volumes"])
    assert any(volume.endswith(":/operational-data:ro") for volume in service["volumes"])
    assert any(volume.endswith(":/models:ro") for volume in service["volumes"])
    assert any(volume.endswith(":/engine-cache:ro") for volume in service["volumes"])
    assert any(volume.endswith(":/earth-engine-auth:ro") for volume in service["volumes"])
    assert service["environment"]["VITA_CROP_BACKEND"] == "tensorrt"
    assert service["environment"]["VITA_CLOUD_BACKEND"] == "tensorrt"
    assert service["environment"]["VITA_EE_CREDENTIALS"] == (
        "/earth-engine-auth/credentials.json"
    )


def test_raw_engine_builder_isolated_from_operational_service() -> None:
    project = Path(__file__).resolve().parents[1]
    builder = (project / "deploy" / "payload" / "build-raw-service-engines.sh")
    builder_text = builder.read_text(encoding="utf-8")
    build_override = yaml.safe_load(
        (project / "deploy" / "compose.payload.raw-engine-build.yaml").read_text(
            encoding="utf-8"
        )
    )
    deploy_text = (project / "deploy" / "payload" / "deploy-raw-service.sh").read_text(
        encoding="utf-8"
    )

    assert builder.is_file()
    assert (
        'RAW_ENGINE_CACHE="${VITA_RAW_ENGINE_CACHE_HOST:-'
        '$OPERATIONAL_ROOT/runtime/raw-engines}"' in builder_text
    )
    assert 'cp -a "$STABLE_ENGINE_CACHE" "$resolved_raw_cache"' in builder_text
    assert (
        "VITA_BALKAN_CLOUD_PROFILE_INPUTS="
        "balkan1/preprocessed/3458_L1ORT.tif" in builder_text
    )
    assert "VITA_RAW_CLOUD_FIXED_PATCH_SIZE=" in builder_text
    assert 'BUILD_COMPOSE_FILE="$PROJECT_ROOT/deploy/' in builder_text
    assert '--file "$BUILD_COMPOSE_FILE" run' in builder_text
    assert "stop raw-payload" in builder_text
    assert "stop payload" not in builder_text
    assert "docker compose down" not in builder_text
    assert "stable TensorRT manifest changed" in builder_text
    assert "stable payload container stopped" in builder_text
    assert "runtime/raw-engines/tensorrt/direct/accepted.json" in deploy_text
    assert build_override["services"]["raw-payload"]["volumes"] == [
        "${VITA_RAW_ENGINE_CACHE_HOST:?set VITA_RAW_ENGINE_CACHE_HOST}:/engine-cache"
    ]
