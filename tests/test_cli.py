from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from vita_integration.cli import (
    _balkan_pipeline_timings,
    _bbox,
    _render_timing_report,
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
            "--bbox=-96.70,40.65,-96.64,40.71",
            "--start",
            "2026-06-01",
            "--end",
            "2026-07-31",
            "--region-id",
            "nebraska-crops",
        ]
    )
    assert args.handler is run_sentinel
    assert args.bbox == (-96.70, 40.65, -96.64, 40.71)


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


@pytest.mark.parametrize("value", ["1,2,3", "west,south,east,north"])
def test_bbox_rejects_malformed_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _bbox(value)


def test_balkan_timing_extraction_uses_stage_runtime_contracts() -> None:
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

    timing = _balkan_pipeline_timings(result)

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
