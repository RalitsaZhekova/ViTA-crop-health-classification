from __future__ import annotations

from prithvi_payload.performance_acceptance import _requests


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
