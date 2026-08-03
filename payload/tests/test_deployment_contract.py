from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[2]
MODEL_HASHES = {
    "c948977bffdaeb89ecf4f4d069db13c7ed81d4f3403eec9e257c45c235b1484e",
    "ab8f039866d6714b249f850779b9523b5f6afb55ee891077a71db0fdebc9b529",
}


def _read(relative: str) -> str:
    return (REPOSITORY_ROOT / relative).read_text(encoding="utf-8")


def test_secrets_environments_and_runtime_are_excluded() -> None:
    gitignore = _read(".gitignore")
    dockerignore = _read(".dockerignore")

    for rule in (
        "secrets/*",
        "!secrets/.gitkeep",
        "earth-engine.json",
        "*.service-account.json",
        ".env.payload",
        ".env.ground",
        "runtime/*",
        "!runtime/.gitkeep",
        "data/*",
        "testing/inputs/balkan1/*",
        "*.tif",
        "*.tiff",
    ):
        assert rule in gitignore
    for rule in (
        ".git",
        ".venv",
        ".env*",
        "secrets",
        "earth-engine.json",
        "*.service-account.json",
        "runtime",
        "data",
        "outputs",
        "testing/inputs",
        "testing/runs",
        "notebooks",
    ):
        assert rule in dockerignore
    assert "payload/models" not in dockerignore.splitlines()


def test_payload_image_has_fixed_boundary_and_no_credentials() -> None:
    dockerfile = _read("deployment/payload/Dockerfile")
    entrypoint = _read("deployment/payload/entrypoint.sh")

    assert "ARG BASE_IMAGE" in dockerfile
    assert "FROM ${BASE_IMAGE}" in dockerfile
    assert "COPY ground" not in dockerfile
    assert "COPY integration" not in dockerfile
    assert "COPY payload/models payload/models" in dockerfile
    assert "ARG APPLICATION_UID=1000" in dockerfile
    assert "ARG APPLICATION_GID=1000" in dockerfile
    assert "USER ${APPLICATION_UID}:${APPLICATION_GID}" in dockerfile
    assert "COPY secrets" not in dockerfile
    assert "earth-engine.json" not in dockerfile
    assert "GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/earth_engine_credentials" in dockerfile
    assert "prithvi_payload.container_preflight --models-only" in dockerfile
    assert "vita-stack-guard.py compare" in dockerfile
    assert "--constraint /tmp/vita-payload-constraints.txt" in dockerfile
    assert 'ARG VITA_TARGET_PLATFORM' in dockerfile
    for checksum in MODEL_HASHES:
        assert checksum in dockerfile

    assert "EE_PROJECT_ID is required" in entrypoint
    assert "GOOGLE_APPLICATION_CREDENTIALS is required" in entrypoint
    assert "readable regular file" in entrypoint
    assert "prithvi_payload.container_preflight" in entrypoint
    assert 'exec "$@"' in entrypoint
    assert "ee.Authenticate" not in entrypoint
    assert "cat " not in entrypoint


def test_ground_image_excludes_payload_science_and_secret() -> None:
    dockerfile = _read("deployment/ground/Dockerfile")

    assert "COPY ground/src ground/src" in dockerfile
    assert "COPY shared/src shared/src" in dockerfile
    assert "COPY payload" not in dockerfile
    assert "COPY integration" not in dockerfile
    assert "earth_engine" not in dockerfile.lower()
    assert "ARG APPLICATION_UID=1000" in dockerfile
    assert "USER ${APPLICATION_UID}:${APPLICATION_GID}" in dockerfile


def test_payload_composes_use_project_paths_secret_and_loopback_only() -> None:
    for filename in (
        "compose.payload.local.yaml",
        "compose.payload.jetson.yaml",
        "compose.payload.production.yaml",
    ):
        compose = _read(filename)
        assert '"127.0.0.1:8081:8081"' in compose
        assert "./runtime/payload/jobs:/data/jobs" in compose
        assert "./runtime/payload/outbox:/data/outbox" in compose
        assert "./runtime/payload/logs:/data/logs" in compose
        assert "file: ${EE_CREDENTIALS_FILE:-./secrets/earth-engine.json}" in compose
        assert "GOOGLE_APPLICATION_CREDENTIALS: /run/secrets/earth_engine_credentials" in compose
        assert "network_mode: host" not in compose
        assert "privileged:" not in compose
        assert "/etc/" not in compose
        assert "/srv/" not in compose
        assert "/opt/" not in compose

    jetson = _read("compose.payload.jetson.yaml")
    local = _read("compose.payload.local.yaml")
    assert "VITA_TARGET_PLATFORM: local-x86" in local
    assert "runtime: nvidia" in jetson
    assert "reservations:" not in jetson
    assert "VITA_TARGET_PLATFORM: jetson" in jetson
    assert 'VITA_MAX_CONCURRENT_JOBS: "1"' in jetson
    assert "models_warmed" in jetson
    assert "cuda_available" in jetson


def test_payload_container_constraints_prevent_future_numpy_and_opencv_drift() -> None:
    constraints = _read("deployment/payload/constraints.txt")
    stack_guard = _read("deployment/payload/stack_guard.py")

    assert "numpy>=2.2,<2.3" in constraints
    assert "opencv-python-headless==4.11.0.86" in constraints
    assert "opencv-python-headless==5" not in constraints
    assert 'PINNED_OPENCV_HEADLESS_VERSION = "4.11.0.86"' in stack_guard
    assert '"nvidia-nccl-"' in stack_guard
    for snapshot_field in (
        "cv2_importable",
        "cv2_version",
        "cv2_path",
        "opencv_wheels",
    ):
        assert snapshot_field in stack_guard


def test_ground_and_full_local_composes_keep_services_separate() -> None:
    ground = _read("compose.ground.yaml")
    full = _read("compose.full-local.yaml")

    assert '"127.0.0.1:8000:8000"' in ground
    assert "./runtime/ground:/data/ground" in ground
    assert "/api/v1/health" in ground
    assert "earth_engine" not in ground.lower()
    assert "payload/models" not in ground

    assert "  payload:" in full
    assert "  ground:" in full
    ground_section = full.split("  ground:", maxsplit=1)[1].split("\nsecrets:", maxsplit=1)[0]
    assert "earth_engine_credentials" not in ground_section
    assert "runtime/payload" not in ground_section
    assert "./runtime/ground:/data/ground" in ground_section


def test_environment_examples_contain_no_real_project_or_secret() -> None:
    payload = _read(".env.payload.example")
    ground = _read(".env.ground.example")

    assert "EE_PROJECT_ID=REPLACE_WITH_GOOGLE_CLOUD_PROJECT_ID" in payload
    assert "EE_CREDENTIALS_FILE=./secrets/earth-engine.json" in payload
    assert "VITA_PAYLOAD_IMAGE=vita-payload:0.1.0" in payload
    assert "VITA_PAYLOAD_BASE_IMAGE=REPLACE_WITH_COMPATIBLE_BASE_IMAGE" in payload
    assert "vita-503208" not in payload
    assert "EE_" not in ground
    assert "credential" not in ground.lower()


def test_shell_entrypoints_are_lf_only() -> None:
    for filename in ("deployment/payload/entrypoint.sh", "deployment/ground/entrypoint.sh"):
        assert b"\r\n" not in (REPOSITORY_ROOT / filename).read_bytes()
