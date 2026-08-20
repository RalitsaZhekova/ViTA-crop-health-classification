#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OPERATIONAL_ROOT="${VITA_OPERATIONAL_PROJECT_ROOT:-/data/code/VITA}"
COMPOSE_FILE="$PROJECT_ROOT/deploy/compose.payload.raw-service.yaml"
BUILD_COMPOSE_FILE="$PROJECT_ROOT/deploy/compose.payload.raw-engine-build.yaml"
BASE_IMAGE="${VITA_RAW_PAYLOAD_BASE_IMAGE:-vita-payload:1.0.0}"
STABLE_ENGINE_CACHE="$OPERATIONAL_ROOT/runtime/engines"
RAW_ENGINE_CACHE="${VITA_RAW_ENGINE_CACHE_HOST:-$OPERATIONAL_ROOT/runtime/raw-engines}"

export VITA_RAW_HOST_UID="${VITA_RAW_HOST_UID:-$(id -u)}"
export VITA_RAW_HOST_GID="${VITA_RAW_HOST_GID:-$(id -g)}"
export VITA_RAW_PAYLOAD_IMAGE="${VITA_RAW_PAYLOAD_IMAGE:-vita-payload:raw-band}"
export VITA_RAW_PAYLOAD_BASE_IMAGE="$BASE_IMAGE"
export VITA_RAW_PAYLOAD_PORT="${VITA_RAW_PAYLOAD_PORT:-8091}"
export VITA_RAW_DATA_HOST="${VITA_RAW_DATA_HOST:-$OPERATIONAL_ROOT/data/raw-inputs}"
export VITA_PROCESSED_DATA_HOST="${VITA_PROCESSED_DATA_HOST:-$OPERATIONAL_ROOT/data}"
export VITA_RAW_MODEL_ASSETS_HOST="${VITA_RAW_MODEL_ASSETS_HOST:-$OPERATIONAL_ROOT/payload/models}"
export VITA_RAW_ENGINE_CACHE_HOST="$RAW_ENGINE_CACHE"
export VITA_RAW_OUTPUT_HOST="${VITA_RAW_OUTPUT_HOST:-$OPERATIONAL_ROOT/runtime/raw-payload}"

test -f "$COMPOSE_FILE" || { echo "ERROR: missing $COMPOSE_FILE" >&2; exit 1; }
test -f "$BUILD_COMPOSE_FILE" || { echo "ERROR: missing $BUILD_COMPOSE_FILE" >&2; exit 1; }
test -f "$STABLE_ENGINE_CACHE/tensorrt/direct/accepted.json" || {
    echo "ERROR: stable accepted TensorRT manifest is missing" >&2
    exit 1
}
test -f "$VITA_PROCESSED_DATA_HOST/balkan1/preprocessed/3458_L1ORT.tif" || {
    echo "ERROR: 3458 processed qualification scene is missing" >&2
    exit 1
}

runtime_root="$(realpath -m "$OPERATIONAL_ROOT/runtime")"
resolved_raw_cache="$(realpath -m "$RAW_ENGINE_CACHE")"
case "$resolved_raw_cache/" in
    "$runtime_root"/*) ;;
    *) echo "ERROR: isolated engine cache must remain below $runtime_root" >&2; exit 1 ;;
esac
if [ ! -e "$resolved_raw_cache" ]; then
    echo "Creating an isolated engine cache from the accepted stable artifacts."
    cp -a "$STABLE_ENGINE_CACHE" "$resolved_raw_cache"
elif [ ! -f "$resolved_raw_cache/tensorrt/direct/accepted.json" ]; then
    echo "ERROR: existing isolated engine cache has no accepted manifest" >&2
    exit 1
fi
export VITA_RAW_ENGINE_CACHE_HOST="$resolved_raw_cache"

stable_manifest_hash="$(sha256sum "$STABLE_ENGINE_CACHE/tensorrt/direct/accepted.json" | awk '{print $1}')"
stable_container_id="$(docker ps --filter label=com.docker.compose.project=vita-payload \
    --filter label=com.docker.compose.service=payload --format '{{.ID}}' | head -n 1)"
test -n "$stable_container_id" || {
    echo "ERROR: the stable payload service is not running" >&2
    exit 1
}
echo "Stable payload remains running during the isolated build: $stable_container_id"

cd "$PROJECT_ROOT"
docker compose --project-name vita-payload-raw-service --file "$COMPOSE_FILE" build raw-payload
docker compose --project-name vita-payload-raw-service --file "$COMPOSE_FILE" stop raw-payload

restart_raw_service=true
restore_raw_service() {
    if [ "$restart_raw_service" = true ]; then
        echo "Restoring the raw service from the last accepted isolated manifest."
        docker compose --project-name vita-payload-raw-service --file "$COMPOSE_FILE" up \
            --detach --no-deps raw-payload || true
    fi
}
trap restore_raw_service EXIT

echo "Building only missing plans and validating all five cloud qualification scenes."
docker compose --project-name vita-payload-raw-service --file "$COMPOSE_FILE" \
    --file "$BUILD_COMPOSE_FILE" run \
    --rm --no-deps \
    -e VITA_INPUT_ROOT=/operational-data \
    -e VITA_OUTPUT_ROOT=/runtime/runs \
    -e VITA_BALKAN_PREPARE_INPUTS=balkan1/preprocessed/3370_L1ORT.tif,balkan1/preprocessed/3408_L1ORT.tif \
    -e VITA_BALKAN_CLOUD_PROFILE_INPUTS=balkan1/preprocessed/3458_L1ORT.tif \
    -e VITA_DEMO_SENTINEL_IMAGES=sentinel2/S2_20260610T091331_T35TLG_bulgaria-thrace.tif,sentinel2/S2_20260115T135659_T21LXF_brazil-mato-grosso.tif \
    -e VITA_CLOUD_BATCH_SIZE=4 \
    -e VITA_RAW_CLOUD_FIXED_PATCH_SIZE= \
    --entrypoint python raw-payload -m prithvi_payload.tensorrt_builder

test "$(sha256sum "$STABLE_ENGINE_CACHE/tensorrt/direct/accepted.json" | awk '{print $1}')" = "$stable_manifest_hash" || {
    echo "ERROR: stable TensorRT manifest changed during isolated build" >&2
    exit 1
}
test "$(docker ps --filter id="$stable_container_id" --format '{{.ID}}')" = "$stable_container_id" || {
    echo "ERROR: stable payload container stopped during isolated build" >&2
    exit 1
}

restart_raw_service=false
trap - EXIT
echo "Starting the raw service with its isolated accepted engine set."
VITA_RAW_ENGINE_CACHE_HOST="$resolved_raw_cache" \
    bash "$PROJECT_ROOT/deploy/payload/deploy-raw-service.sh"
