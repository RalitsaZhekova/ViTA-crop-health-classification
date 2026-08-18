from __future__ import annotations

import logging
from pathlib import Path

import pytest
from vita_integration.cli import (
    _configure_runtime_logging,
    _execution_scene_id,
    _ExpectedProviderTiffFilter,
    _local_pipeline_timings,
    _render_timing_report,
    _resolve_sentinel_source,
    parser,
    run_balkan,
    run_sentinel,
)


def test_cli_exposes_exactly_the_two_mvp_commands() -> None:
    command_parser = parser()
    subparsers = next(
        action
        for action in command_parser._actions
        if hasattr(action, "choices") and action.choices
    )
    assert set(subparsers.choices) == {"sentinel", "balkan"}


def test_sentinel_arguments_are_one_complete_command() -> None:
    args = parser().parse_args(
        [
            "sentinel",
            "data/sentinel2",
            "--image",
            "S2_20260712T170851_T14TPL_cloudy.tif",
            "--region-id",
            "nebraska-crops",
        ]
    )
    assert args.handler is run_sentinel
    assert args.input == Path("data/sentinel2")
    assert args.image == "S2_20260712T170851_T14TPL_cloudy.tif"


def test_sentinel_folder_requires_an_explicit_choice_when_ambiguous(tmp_path: Path) -> None:
    first = tmp_path / "first.tif"
    second = tmp_path / "second.tif"
    first.touch()
    second.touch()

    with pytest.raises(ValueError, match="pass --image"):
        _resolve_sentinel_source(tmp_path, None)
    assert _resolve_sentinel_source(tmp_path, second.name) == second.resolve()


def test_default_scene_ids_are_unique_but_explicit_ids_are_stable(tmp_path: Path) -> None:
    source = tmp_path / "scene.tif"

    first = _execution_scene_id(source, None)
    second = _execution_scene_id(source, None)

    assert first != second
    assert first.startswith("scene_20")
    assert _execution_scene_id(source, "fixed-id") == "fixed-id"


def test_runtime_logging_filters_only_known_provider_noise() -> None:
    warning_filter = _ExpectedProviderTiffFilter()
    known = logging.LogRecord(
        "rasterio._env",
        logging.WARNING,
        "test.py",
        1,
        "TIFFReadDirectory: ExtraSamples doesn't match SamplesPerPixel",
        (),
        None,
    )
    exact_known = logging.LogRecord(
        "rasterio._env",
        logging.WARNING,
        "test.py",
        1,
        "Sum of Photometric type-related color channels and ExtraSamples doesn't match",
        (),
        None,
    )
    legacy_deflate = logging.LogRecord(
        "rasterio._env",
        logging.WARNING,
        "test.py",
        1,
        "Creating TIFF with legacy Deflate codec identifier",
        (),
        None,
    )
    unexpected = logging.LogRecord(
        "rasterio._env",
        logging.WARNING,
        "test.py",
        1,
        "Corrupt TIFF directory",
        (),
        None,
    )

    assert warning_filter.filter(known)
    assert not warning_filter.filter(exact_known)
    assert not warning_filter.filter(legacy_deflate)
    assert warning_filter.filter(unexpected)
    _configure_runtime_logging()
    assert logging.getLogger("torch.utils.flop_counter").level == logging.ERROR


def test_balkan_command_fails_before_models_when_source_is_missing(tmp_path: Path) -> None:
    args = parser().parse_args(
        [
            "balkan",
            str(tmp_path / "missing.tif"),
            "--region-id",
            "balkan-test",
        ]
    )
    assert args.handler is run_balkan
    with pytest.raises(ValueError, match="does not exist"):
        run_balkan(args)


def test_balkan_command_exposes_local_alignment_handoff() -> None:
    args = parser().parse_args(
        [
            "balkan",
            "data/balkan1/preprocessed/3408_L1ORT.tif",
            "--region-id",
            "balkan-test",
            "--align-bands-to",
            "runtime/aligned/3408_L1ORT_aligned.tif",
            "--alignment-band-start-axis",
            "column",
            "--alignment-device",
            "cuda",
            "--alignment-warp-tile-size",
            "1024",
            "--alignment-compression",
            "deflate",
            "--alignment-skip-overviews",
            "--experimental-raw-proxy",
        ]
    )

    assert args.align_bands_to == Path("runtime/aligned/3408_L1ORT_aligned.tif")
    assert args.alignment_metadata is None
    assert args.alignment_band_order == ("BLUE", "GREEN", "RED", "NIR", "PAN")
    assert args.alignment_band_start_axis == "column"
    assert args.alignment_device == "cuda"
    assert args.alignment_warp_tile_size == 1024
    assert args.alignment_compression == "deflate"
    assert args.alignment_skip_overviews is True
    assert args.experimental_raw_proxy is True


def test_local_timing_extraction_uses_stage_runtime_contracts() -> None:
    result = {
        "timing": {
            "intake_seconds": 0.1,
            "cloud_plan_seconds": 0.2,
            "crop_plan_seconds": 0.3,
        },
        "stage_metadata": {
            "cloud": {
                "runtime": {
                    "seconds": 4.0,
                    "analysis_grid_preparation_seconds": 1.0,
                    "inference_seconds": 2.0,
                    "mask_processing_seconds": 0.5,
                    "mask_reprojection_seconds": 0.25,
                }
            },
            "crop": {"runtime": {"seconds": 5.0, "inference_seconds": 3.0}},
            "condition": {
                "runtime": {
                    "seconds": 6.0,
                    "first_pass_seconds": 2.0,
                    "robust_statistics_seconds": 1.0,
                    "spatial_pass_seconds": 1.5,
                    "preview_seconds": 0.5,
                }
            },
            "downlink": {"runtime": {"seconds": 0.75}},
        },
    }

    timing = _local_pipeline_timings(result)

    assert timing["intake_seconds"] == 0.1
    assert timing["cloud_inference_seconds"] == 2.0
    assert timing["crop_inference_seconds"] == 3.0
    assert timing["condition_spatial_pass_seconds"] == 1.5
    assert timing["downlink_packaging_seconds"] == 0.75


def test_terminal_timing_report_is_readable_and_warns_about_nested_totals() -> None:
    report = _render_timing_report(
        {
            "pipeline_total_seconds": 12.34567,
            "cloud_inference_seconds": 1.25,
            "end_to_end_seconds": 15.0,
        }
    )

    assert "Top-level rows sum to END TO END" in report
    assert "Pipeline total" in report and "12.3457" in report
    assert "Cloud inference" in report and "1.2500" in report
    assert "END TO END" in report and "15.0000" in report
