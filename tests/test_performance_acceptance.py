from __future__ import annotations

import json

import pytest
from prithvi_payload.performance_acceptance import (
    _remove_acceptance_output,
    _requests,
    _write_report_atomic,
)


def test_performance_acceptance_builds_two_requests_per_sensor(monkeypatch) -> None:
    monkeypatch.setenv("VITA_DEMO_SENTINEL_IMAGES", "sentinel/a.tif,sentinel/b.tif")
    monkeypatch.setenv("VITA_BALKAN_PREPARE_INPUTS", "balkan/a.tif,balkan/b.tif")

    requests = _requests("stamp")

    assert [request["sensor"] for request in requests] == [
        "sentinel-2",
        "sentinel-2",
        "balkan-1",
        "balkan-1",
    ]
    assert requests[0]["input"] == "sentinel"
    assert requests[0]["image"] == "a.tif"


def test_performance_acceptance_removes_only_its_exact_run(tmp_path, monkeypatch) -> None:
    acceptance = tmp_path / "accept-b1-stamp-r1"
    retained = tmp_path / "operator-job"
    acceptance.mkdir()
    retained.mkdir()
    (acceptance / "result.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("VITA_OUTPUT_ROOT", str(tmp_path))

    _remove_acceptance_output(acceptance.name)

    assert not acceptance.exists()
    assert retained.is_dir()


def test_performance_acceptance_persists_release_evidence_atomically(
    tmp_path,
    monkeypatch,
) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setenv("VITA_OUTPUT_ROOT", str(runs))
    report = {"status": "UNDER_TWO_SECONDS", "scenes": []}

    path = _write_report_atomic(report)

    assert path == tmp_path / "performance-acceptance.json"
    assert json.loads(path.read_text(encoding="utf-8")) == report
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("job_id", ["operator-job", "accept-../operator-job"])
def test_performance_acceptance_rejects_unsafe_cleanup_names(
    tmp_path,
    monkeypatch,
    job_id,
) -> None:
    retained = tmp_path / "operator-job"
    retained.mkdir()
    monkeypatch.setenv("VITA_OUTPUT_ROOT", str(tmp_path))

    with pytest.raises(ValueError, match="Refusing"):
        _remove_acceptance_output(job_id)

    assert retained.is_dir()
