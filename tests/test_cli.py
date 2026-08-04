from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from vita_integration.cli import _bbox, parser, run_balkan, run_sentinel


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
