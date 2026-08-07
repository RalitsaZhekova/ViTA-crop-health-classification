"""Manual, stage-gated payload pipeline entry point."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from cloud_detection.backend import CloudBackend
from cloud_detection.config import load_config

from prithvi_payload.balkan_crop_calibration import ADAPTER_MODE
from prithvi_payload.cloud_classifier import DEFAULT_CLOUD_CONFIG, load_cloud_model
from prithvi_payload.cloud_executor import execute_cloud_stage
from prithvi_payload.cloud_stage import build_cloud_stage_plan
from prithvi_payload.crop_stage import (
    DEFAULT_MAX_CLOUD_PERCENTAGE,
    build_crop_stage_plan,
)
from prithvi_payload.downlink import DEFAULT_GRID_SIZE, DEFAULT_MAX_IMAGE_DIMENSION
from prithvi_payload.runtime_config import environment_flag
from prithvi_payload.scene_intake import inspect_scene

_CPU_OVERLAP_POOL = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="payload-overlap",
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finish(output_root: Path, result: dict[str, Any]) -> dict[str, Any]:
    summary_path = output_root / "result.json"
    result["artifacts"]["run_summary"] = str(summary_path.resolve())
    _write_json(summary_path, result)
    return result


def run_scene(
    input_path: str | Path,
    *,
    sensor: str,
    output_root: str | Path,
    acquired_at: str | None = None,
    scene_id: str | None = None,
    band_order: Sequence[str] | None = None,
    crop_calibration_path: str | Path | None = None,
    reflectance_scale: float | None = None,
    stop_after: str = "cloud",
    max_crop_cloud_percentage: float = DEFAULT_MAX_CLOUD_PERCENTAGE,
    region_id: str | None = None,
    condition_tile_size: int = 512,
    downlink_max_image_dimension: int = DEFAULT_MAX_IMAGE_DIMENSION,
    downlink_grid_size: int = DEFAULT_GRID_SIZE,
    overwrite: bool = False,
    cloud_config_path: str | Path = DEFAULT_CLOUD_CONFIG,
    cloud_backend: CloudBackend | None = None,
    cloud_config: dict[str, Any] | None = None,
    crop_model: Any | None = None,
    acquisition_metadata: dict[str, Any] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run only the explicitly selected stages for one preprocessed scene."""
    if stop_after not in {"intake", "cloud", "crop", "condition", "downlink"}:
        raise ValueError("stop_after must be intake, cloud, crop, condition or downlink")
    if condition_tile_size <= 0:
        raise ValueError("condition_tile_size must be positive")
    if downlink_max_image_dimension <= 0 or downlink_grid_size <= 0:
        raise ValueError("Downlink image and grid dimensions must be positive")
    compact_payload = stop_after == "downlink" and environment_flag(
        "VITA_COMPACT_PAYLOAD_PIPELINE", False
    )
    output_root = Path(output_root)
    intake_started = time.perf_counter()
    intake = inspect_scene(
        input_path,
        sensor=sensor,
        acquired_at=acquired_at,
        scene_id=scene_id,
        band_order=band_order,
        crop_calibration_path=crop_calibration_path,
    )
    resolved_scene_id = intake["scene_id"]
    intake_seconds = time.perf_counter() - intake_started
    analysis_grid_seconds = 0.0
    if (
        sensor == "balkan-1"
        and stop_after != "intake"
        and intake["readiness"]["intake"] == "READY"
    ):
        from prithvi_payload.balkan_analysis import materialize_balkan_analysis_grid

        if progress_callback is not None:
            progress_callback("preparing_analysis_grid")
        analysis = materialize_balkan_analysis_grid(
            intake,
            output_root=output_root,
            overwrite=overwrite,
        )
        intake["analysis"] = analysis
        analysis_grid_seconds = float(analysis["runtime"]["seconds"])
    intake_path = output_root / "metadata" / f"{resolved_scene_id}_intake.json"
    _write_json(intake_path, intake)
    artifacts = {"intake": str(intake_path.resolve())}
    if isinstance(intake.get("analysis"), dict):
        artifacts["analysis_grid"] = intake["analysis"]["source_path"]
    result: dict[str, Any] = {
        "schema_version": "0.1-draft",
        "scene_id": resolved_scene_id,
        "sensor": sensor,
        "requested_stop_after": stop_after,
        "completed_stages": ["intake"],
        "status": "INTAKE_READY",
        "artifacts": artifacts,
        "summary": {
            "intake": {
                "readiness": intake["readiness"]["intake"],
                "band_count": intake["raster"]["band_count"],
                "width": intake["raster"]["width"],
                "height": intake["raster"]["height"],
            }
        },
        "stage_metadata": {"intake": intake},
        "timing": {
            "intake_seconds": intake_seconds,
            "shared_analysis_grid_seconds": analysis_grid_seconds,
        },
        "warnings": list(intake["warnings"]),
        "errors": list(intake["errors"]),
    }
    if intake["readiness"]["intake"] != "READY":
        result["status"] = "REJECTED_AT_INTAKE"
        return _finish(output_root, result)
    if stop_after == "intake":
        return _finish(output_root, result)

    cloud_plan_started = time.perf_counter()
    plan = build_cloud_stage_plan(
        intake,
        reflectance_scale=reflectance_scale,
    )
    plan_path = output_root / "metadata" / f"{resolved_scene_id}_cloud_plan.json"
    _write_json(plan_path, plan)
    result["artifacts"]["cloud_plan"] = str(plan_path.resolve())
    result["stage_metadata"]["cloud_plan"] = plan
    result["timing"]["cloud_plan_seconds"] = time.perf_counter() - cloud_plan_started
    result["warnings"].extend(plan["warnings"])
    result["errors"].extend(plan["errors"])
    if plan["readiness"] != "READY":
        result["status"] = "BLOCKED_AT_CLOUD_PLAN"
        return _finish(output_root, result)

    prepared_rgb_future: Future[dict[str, Any]] | None = None
    if compact_payload:
        from prithvi_payload.downlink import prepare_rgb_preview

        analysis = intake.get("analysis")
        use_shared_analysis = sensor == "balkan-1" and isinstance(analysis, dict)
        rgb_source = (
            analysis["source_path"] if use_shared_analysis else intake["source_path"]
        )
        rgb_mapping = (
            analysis.get("logical_band_mapping", {})
            if use_shared_analysis
            else intake.get("logical_band_mapping", {})
        )
        crop_route = intake.get("model_band_routes", {}).get(
            "crop_classification", {}
        )
        spectral_adapter = crop_route.get("spectral_adapter")
        calibrated_balkan = (
            sensor == "balkan-1"
            and isinstance(spectral_adapter, dict)
            and spectral_adapter.get("mode") == ADAPTER_MODE
        )
        rgb_scale = 10_000.0 if calibrated_balkan else float(
            plan["input"]["reflectance_scale"]
        )
        prepared_rgb_future = _CPU_OVERLAP_POOL.submit(
            prepare_rgb_preview,
            rgb_source,
            mapping=rgb_mapping,
            spectral_adapter=(
                spectral_adapter if isinstance(spectral_adapter, dict) else None
            ),
            original_source_path=Path(intake["source_path"]),
            sensor=sensor,
            reflectance_scale=rgb_scale,
            maximum_dimension=downlink_max_image_dimension,
        )

    if cloud_backend is None:
        runtime = load_cloud_model(cloud_config_path)
        cloud_backend = runtime.backend
        cloud_config = runtime.config
    elif cloud_config is None:
        cloud_config = load_config(cloud_config_path)
    cloud_metadata = execute_cloud_stage(
        plan,
        output_root=output_root,
        backend=cloud_backend,
        config=cloud_config,
        persist_rasters=not compact_payload,
    )
    cloud_products = cloud_metadata.pop("_products", None)
    result["completed_stages"].append("cloud")
    result["status"] = "CLOUD_COMPLETE"
    result["cloud_decision"] = cloud_metadata["decision"]
    result["artifacts"]["cloud"] = cloud_metadata["output_files"]
    percentages = cloud_metadata["class_percentages"]
    result["summary"]["cloud"] = {
        "decision": cloud_metadata["decision"],
        "clear_percentage": percentages["clear"],
        "thick_cloud_percentage": percentages["thick_cloud"],
        "thin_cloud_percentage": percentages["thin_cloud"],
        "cloud_shadow_percentage": percentages["cloud_shadow"],
        "total_cloud_percentage": cloud_metadata["cloud_percentage"],
        "usable_percentage": cloud_metadata["usable_percentage"],
        "unusable_percentage": cloud_metadata["unusable_percentage"],
        "invalid_percentage": cloud_metadata["invalid_percentage"],
        "runtime_seconds": cloud_metadata["runtime"]["seconds"],
    }
    result["stage_metadata"]["cloud"] = cloud_metadata
    if stop_after == "cloud":
        return _finish(output_root, result)
    _finish(output_root, result)
    return continue_scene_from_cloud(
        output_root / "result.json",
        stop_after=stop_after,
        max_cloud_percentage=max_crop_cloud_percentage,
        region_id=region_id,
        condition_tile_size=condition_tile_size,
        downlink_max_image_dimension=downlink_max_image_dimension,
        downlink_grid_size=downlink_grid_size,
        overwrite=overwrite,
        crop_model=crop_model,
        acquisition_metadata=acquisition_metadata,
        progress_callback=progress_callback,
        cloud_products=cloud_products,
        prepared_rgb_future=prepared_rgb_future,
    )


def continue_scene_from_cloud(
    payload_result_path: str | Path,
    *,
    stop_after: str = "downlink",
    max_cloud_percentage: float = DEFAULT_MAX_CLOUD_PERCENTAGE,
    region_id: str | None = None,
    condition_tile_size: int = 512,
    downlink_max_image_dimension: int = DEFAULT_MAX_IMAGE_DIMENSION,
    downlink_grid_size: int = DEFAULT_GRID_SIZE,
    overwrite: bool = False,
    crop_model: Any | None = None,
    acquisition_metadata: dict[str, Any] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    cloud_products: dict[str, Any] | None = None,
    prepared_rgb_future: Future[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Continue from one completed cloud stage without executing cloud inference again."""
    if stop_after not in {"crop", "condition", "downlink"}:
        raise ValueError("cloud continuation stop_after must be crop, condition or downlink")
    result_path = Path(payload_result_path).resolve()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "CLOUD_COMPLETE" or result.get("completed_stages") != [
        "intake",
        "cloud",
    ]:
        raise ValueError("Cloud continuation requires an uncontinued CLOUD_COMPLETE result")
    output_root = result_path.parent
    stage_metadata = result.get("stage_metadata", {})
    timing = result.get("timing")
    if not isinstance(timing, dict):
        raise ValueError("Cloud continuation timing metadata is invalid")
    intake = stage_metadata.get("intake")
    plan = stage_metadata.get("cloud_plan")
    cloud_metadata = stage_metadata.get("cloud")
    if not all(isinstance(value, dict) for value in (intake, plan, cloud_metadata)):
        raise ValueError("Cloud continuation metadata is incomplete")
    resolved_scene_id = str(result["scene_id"])
    if acquisition_metadata is not None:
        result["stage_metadata"]["acquisition"] = acquisition_metadata
        result["summary"]["acquisition"] = {
            "provider": acquisition_metadata.get("provider"),
            "collection": acquisition_metadata.get("collection"),
            "provider_scene_id": acquisition_metadata.get("provider_scene_id"),
            "metadata_cloud_percentage": acquisition_metadata.get(
                "earth_engine_metadata_cloud_percentage"
            ),
            "candidate_rank": acquisition_metadata.get("candidate_rank"),
            "candidate_attempt_count": acquisition_metadata.get("candidate_attempt_count"),
        }
    if progress_callback is not None:
        progress_callback("validating_input")

    crop_plan_started = time.perf_counter()
    crop_plan = build_crop_stage_plan(
        intake,
        plan,
        cloud_metadata,
        max_cloud_percentage=max_cloud_percentage,
        unusable_mask_available=cloud_products is not None,
    )
    crop_plan_path = output_root / "metadata" / f"{resolved_scene_id}_crop_plan.json"
    _write_json(crop_plan_path, crop_plan)
    result["artifacts"]["crop_plan"] = str(crop_plan_path.resolve())
    result["stage_metadata"]["crop_plan"] = crop_plan
    timing["crop_plan_seconds"] = time.perf_counter() - crop_plan_started
    result["warnings"].extend(crop_plan["warnings"])
    result["errors"].extend(crop_plan["errors"])
    if crop_plan["readiness"] == "SKIPPED_CLOUD_GATE":
        result["status"] = "CROP_SKIPPED_CLOUD_GATE"
        result["crop_decision"] = "SKIP_CLOUD_COVERAGE"
        result["summary"]["crop"] = {
            "decision": "SKIP_CLOUD_COVERAGE",
            "cloud_percentage": crop_plan["gate"]["observed_percentage"],
            "maximum_cloud_percentage": crop_plan["gate"]["maximum_percentage"],
        }
        return _finish(output_root, result)
    if crop_plan["readiness"] != "READY":
        result["status"] = "BLOCKED_AT_CROP_PLAN"
        return _finish(output_root, result)

    # Keep the 100M crop model and TerraTorch out of intake/cloud-only runs.
    from prithvi_payload.crop_executor import execute_crop_stage

    if progress_callback is not None:
        progress_callback("running_crop")
    compact_payload = stop_after == "downlink" and environment_flag(
        "VITA_COMPACT_PAYLOAD_PIPELINE", False
    )
    crop_metadata = execute_crop_stage(
        crop_plan,
        output_root=output_root,
        model=crop_model,
        persist_rasters=not compact_payload,
        cloud_products=cloud_products,
    )
    crop_products = crop_metadata.pop("_products", None)
    result["completed_stages"].append("crop")
    result["status"] = "CROP_COMPLETE"
    result["crop_decision"] = "CLASSIFIED"
    result["artifacts"]["crop"] = crop_metadata["output_files"]
    result["summary"]["crop"] = {
        "decision": "CLASSIFIED",
        "crop_percentage_usable": crop_metadata["crop_percentage_usable"],
        "usable_percentage": crop_metadata["usable_percentage"],
        "mean_crop_probability_usable": crop_metadata["mean_crop_probability_usable"],
        "mean_confidence_usable": crop_metadata["mean_confidence_usable"],
        "runtime_seconds": crop_metadata["runtime"]["seconds"],
        "device": crop_metadata["runtime"]["device"],
        "inference_backend": crop_metadata["runtime"].get("inference_backend", "pytorch"),
        "tensorrt_engine_count": crop_metadata["runtime"].get("tensorrt_engine_count", 0),
    }
    result["stage_metadata"]["crop"] = crop_metadata
    if stop_after == "crop":
        return _finish(output_root, result)

    # Write the completed crop result before the condition stage validates and
    # consumes the canonical payload contract.
    _finish(output_root, result)
    from prithvi_payload.condition_stage import run_payload_condition

    if progress_callback is not None:
        progress_callback("running_condition")
    condition_root = output_root / "condition_analysis"
    condition_report = run_payload_condition(
        output_root / "result.json",
        output_root=condition_root,
        region_id=region_id,
        tile_size=condition_tile_size,
        overwrite=overwrite,
        crop_products=crop_products,
        persist_rasters=not compact_payload,
    )
    condition_products = condition_report.pop("_products", None)
    condition_report_path = condition_root / "crop_condition_report.json"
    result["completed_stages"].append("condition")
    result["status"] = "CONDITION_COMPLETE"
    result["artifacts"]["condition"] = {
        "report": str(condition_report_path.resolve()),
    }
    result["summary"]["condition"] = {
        "status": condition_report["status"],
        "label": condition_report["condition"]["label"],
        "score": condition_report["condition"]["condition_score"],
        "evidence_quality_label": condition_report["condition"]["evidence_quality_label"],
        "analysis_percentage": condition_report["quality"]["analysis_percentage"],
        "runtime_seconds": condition_report["runtime"]["seconds"],
    }
    result["stage_metadata"]["condition"] = condition_report
    if stop_after == "condition":
        return _finish(output_root, result)

    # The packager consumes the finalized condition result and emits the only
    # three files intended for routine transmission to the ground application.
    _finish(output_root, result)
    from prithvi_payload.downlink import build_downlink_bundle

    if progress_callback is not None:
        progress_callback("packaging")
    downlink_root = output_root / "downlink"
    packaging_started = time.perf_counter()
    downlink = build_downlink_bundle(
        output_root / "result.json",
        output_root=downlink_root,
        max_image_dimension=downlink_max_image_dimension,
        grid_size=downlink_grid_size,
        overwrite=overwrite,
        products=condition_products,
        prepared_rgb=(
            prepared_rgb_future.result()
            if prepared_rgb_future is not None
            else None
        ),
    )
    packaging_detail = downlink.pop("_runtime", {})
    packaging_seconds = time.perf_counter() - packaging_started
    downlink_files = {
        "metadata": downlink_root / "scene.json",
        "rgb_preview": downlink_root / downlink["assets"]["rgb_preview"]["href"],
        "condition_overlay": downlink_root / downlink["assets"]["condition_overlay"]["href"],
    }
    source_bytes = Path(intake["source_path"]).stat().st_size
    package_bytes = int(downlink["package"]["total_bytes"])
    result["completed_stages"].append("downlink")
    result["status"] = "DOWNLINK_READY"
    result["artifacts"]["downlink"] = {
        name: str(path.resolve()) for name, path in downlink_files.items()
    }
    result["summary"]["downlink"] = {
        "file_count": downlink["package"]["file_count"],
        "total_bytes": package_bytes,
        "source_scene_bytes": source_bytes,
        "size_fraction_of_source": package_bytes / source_bytes if source_bytes else None,
        "runtime_seconds": packaging_seconds,
    }
    result["stage_metadata"]["downlink"] = {
        "schema_version": downlink["schema_version"],
        "product_type": downlink["product_type"],
        "algorithm_version": downlink["algorithm_version"],
        "package": downlink["package"],
        "runtime": {"seconds": packaging_seconds, **packaging_detail},
    }
    return _finish(output_root, result)
