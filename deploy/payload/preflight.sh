#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${VITA_PROJECT_ROOT:-/data/code/VITA}"
if [ -f "$PROJECT_ROOT/deploy/payload.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/deploy/payload.env"
    set +a
fi
BASE_IMAGE="${VITA_PAYLOAD_BASE_IMAGE:-nvcr.io/nvidia/pytorch:25.01-py3-igpu}"
EXPECTED_BASE_IMAGE_ID="${VITA_EXPECTED_BASE_IMAGE_ID:-sha256:912d7d7591fc0957092ef202b747b78ba883ca50e5e71781aa8505d9f6a12f6b}"
MIN_FREE_BYTES="${VITA_MIN_FREE_BYTES:-10737418240}"

cd "$PROJECT_ROOT"
test "$(uname -m)" = "aarch64" || { echo "ERROR: payload host must be aarch64" >&2; exit 1; }
command -v docker >/dev/null || { echo "ERROR: docker is not installed" >&2; exit 1; }
docker compose version
docker_record="$(docker info --format 'Docker={{.ServerVersion}} Runtimes={{json .Runtimes}} Root={{.DockerRootDir}}')"
echo "$docker_record"
case "$docker_record" in
    *'"nvidia"'*) ;;
    *) echo "ERROR: Docker has no NVIDIA runtime" >&2; exit 1 ;;
esac
test -f /etc/nv_tegra_release || { echo "ERROR: /etc/nv_tegra_release is missing" >&2; exit 1; }
head -n 1 /etc/nv_tegra_release
dpkg-query --show --showformat='JetPack=${Version}\n' nvidia-jetpack 2>/dev/null || true
device_model="$(tr -d '\0' </proc/device-tree/model 2>/dev/null || true)"
echo "Device=${device_model:-unknown}"
case "$device_model" in
    *Orin*) ;;
    *) echo "ERROR: payload host is not an NVIDIA Jetson Orin" >&2; exit 1 ;;
esac
memory_kib="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
test "$memory_kib" -ge 57671680 || {
    echo "ERROR: expected the 64 GB Orin; MemTotal is ${memory_kib} KiB" >&2
    exit 1
}

project_free_bytes="$(df --output=avail -B1 "$PROJECT_ROOT" | tail -n 1 | tr -d ' ')"
test "$project_free_bytes" -ge "$MIN_FREE_BYTES" || {
    echo "ERROR: fewer than $MIN_FREE_BYTES bytes are free below $PROJECT_ROOT" >&2
    exit 1
}
echo "Project filesystem free bytes=$project_free_bytes"
docker_root="$(docker info --format '{{.DockerRootDir}}')"
docker_free_bytes="$(df --output=avail -B1 "$docker_root" | tail -n 1 | tr -d ' ')"
test "$docker_free_bytes" -ge "$MIN_FREE_BYTES" || {
    echo "ERROR: fewer than $MIN_FREE_BYTES bytes are free below $docker_root" >&2
    exit 1
}
echo "Docker filesystem free bytes=$docker_free_bytes"

docker image inspect "$BASE_IMAGE" >/dev/null || {
    echo "ERROR: expected installed base image not found: $BASE_IMAGE" >&2
    echo "Set VITA_PAYLOAD_BASE_IMAGE to the exact local NVIDIA image tag." >&2
    exit 1
}
base_architecture="$(docker image inspect --format '{{.Architecture}}' "$BASE_IMAGE")"
test "$base_architecture" = "arm64" || {
    echo "ERROR: payload base image architecture is $base_architecture, not arm64" >&2
    exit 1
}
base_image_id="$(docker image inspect --format '{{.Id}}' "$BASE_IMAGE")"
test "$base_image_id" = "$EXPECTED_BASE_IMAGE_ID" || {
    echo "ERROR: payload base image ID is $base_image_id" >&2
    echo "Expected the validated image ID $EXPECTED_BASE_IMAGE_ID" >&2
    exit 1
}
docker image inspect --format 'Base image={{index .RepoTags 0}} architecture={{.Architecture}} image-id={{.Id}} repo-digests={{json .RepoDigests}}' "$BASE_IMAGE"

test -f deploy/payload/demo-assets.sha256 || {
    echo "ERROR: payload demo asset manifest is missing" >&2
    exit 1
}
asset_count="$(grep -Ec '^[0-9a-f]{64}  data/' deploy/payload/demo-assets.sha256)"
test "$asset_count" -eq 6 || {
    echo "ERROR: payload demo manifest must list exactly six files" >&2
    exit 1
}
sha256sum --check --strict deploy/payload/demo-assets.sha256

test -f deploy/payload/model-assets.sha256 || {
    echo "ERROR: payload model asset manifest is missing" >&2
    exit 1
}
model_count="$(grep -Ec '^[0-9a-f]{64}  payload/models/' deploy/payload/model-assets.sha256)"
test "$model_count" -eq 3 || {
    echo "ERROR: payload model manifest must list exactly three files" >&2
    exit 1
}
sha256sum --check --strict deploy/payload/model-assets.sha256

docker run --rm --runtime nvidia --entrypoint python "$BASE_IMAGE" -c \
  'import numpy, tensorrt, torch; assert torch.cuda.is_available(); assert numpy.__version__ == "1.26.4", numpy.__version__; assert torch.__version__.startswith("2.6.0a0+ecf3bae40a"), torch.__version__; assert torch.version.cuda == "12.8", torch.version.cuda; assert tensorrt.__version__.startswith("10.8."), tensorrt.__version__; print({"numpy": numpy.__version__, "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "tensorrt": tensorrt.__version__})'

echo "Checking the temporary-container package sources used by the Docker build."
docker run --rm \
    --entrypoint bash \
    --env DEBIAN_FRONTEND=noninteractive \
    --volume "$PROJECT_ROOT/deploy/requirements-payload.txt:/tmp/requirements-payload.txt:ro" \
    "$BASE_IMAGE" -lc \
  'set -e; apt-get update >/dev/null; apt-get install --yes --no-install-recommends ca-certificates curl gdal-bin libgdal-dev tini >/dev/null; test -x /usr/bin/gdal-config; echo "Disposable GDAL=$(gdal-config --version)"; : > /tmp/vita-constraint.txt; if [ -f /etc/pip/constraint.txt ]; then grep -viE "^[[:space:]]*numpy([[:space:]<>=!~]|$)" /etc/pip/constraint.txt > /tmp/vita-constraint.txt || true; fi; printf "numpy==1.26.4\n" >> /tmp/vita-constraint.txt; python -m pip install --dry-run --constraint /tmp/vita-constraint.txt --requirement /tmp/requirements-payload.txt >/dev/null'

docker compose -f deploy/compose.payload.yaml config --quiet

echo "Payload preflight passed."
