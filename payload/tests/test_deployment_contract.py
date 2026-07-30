from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parents[2]


def test_production_compose_uses_runtime_secret_and_loopback_binding() -> None:
    compose = (REPOSITORY_ROOT / "compose.payload.production.yaml").read_text(
        encoding="utf-8"
    )

    assert "EE_PROJECT_ID: vita-503208" in compose
    assert "GOOGLE_APPLICATION_CREDENTIALS: /run/secrets/earth_engine_credentials" in compose
    assert 'EE_MAX_CANDIDATES: "50"' in compose
    assert 'EE_MAX_SCENE_ATTEMPTS: "5"' in compose
    assert 'CUDA_REQUIRED: "1"' in compose
    assert '"127.0.0.1:8081:8081"' in compose
    assert "file: /etc/vita/secrets/earth-engine.json" in compose
    assert "earth_engine_credentials" in compose


def test_payload_image_requires_target_qualified_base_and_never_copies_credentials() -> None:
    dockerfile = (REPOSITORY_ROOT / "deployment/payload/Dockerfile").read_text(
        encoding="utf-8"
    )
    dockerignore = (REPOSITORY_ROOT / ".dockerignore").read_text(encoding="utf-8")
    entrypoint = (REPOSITORY_ROOT / "deployment/payload/entrypoint.sh").read_text(
        encoding="utf-8"
    )

    assert "ARG PAYLOAD_BASE_IMAGE" in dockerfile
    assert "FROM ${PAYLOAD_BASE_IMAGE}" in dockerfile
    assert "earth_engine_credentials" not in dockerfile
    assert "*credentials*.json" in dockerignore
    assert "*earth-engine*.json" in dockerignore
    assert "*earth_engine*.json" in dockerignore
    assert "ee.Authenticate" not in entrypoint
    assert 'exec python -m prithvi_payload.service --host 0.0.0.0 --port 8081' in entrypoint
