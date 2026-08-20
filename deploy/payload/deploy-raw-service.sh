#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OPERATIONAL_ROOT="${VITA_OPERATIONAL_PROJECT_ROOT:-/data/code/VITA}"
COMPOSE_FILE="$PROJECT_ROOT/deploy/compose.payload.raw-service.yaml"
BASE_IMAGE="${VITA_RAW_PAYLOAD_BASE_IMAGE:-vita-payload:1.0.0}"

export VITA_RAW_HOST_UID="${VITA_RAW_HOST_UID:-$(id -u)}"
export VITA_RAW_HOST_GID="${VITA_RAW_HOST_GID:-$(id -g)}"
export VITA_RAW_PAYLOAD_IMAGE="${VITA_RAW_PAYLOAD_IMAGE:-vita-payload:raw-band}"
export VITA_RAW_PAYLOAD_BASE_IMAGE="$BASE_IMAGE"
export VITA_RAW_PAYLOAD_PORT="${VITA_RAW_PAYLOAD_PORT:-8091}"
export VITA_RAW_DATA_HOST="${VITA_RAW_DATA_HOST:-$OPERATIONAL_ROOT/data/raw-inputs}"
export VITA_PROCESSED_DATA_HOST="${VITA_PROCESSED_DATA_HOST:-$OPERATIONAL_ROOT/data}"
export VITA_RAW_MODEL_ASSETS_HOST="${VITA_RAW_MODEL_ASSETS_HOST:-$OPERATIONAL_ROOT/payload/models}"
if [ -z "${VITA_RAW_ENGINE_CACHE_HOST:-}" ]; then
    if [ -f "$OPERATIONAL_ROOT/runtime/raw-engines/tensorrt/direct/accepted.json" ]; then
        VITA_RAW_ENGINE_CACHE_HOST="$OPERATIONAL_ROOT/runtime/raw-engines"
    else
        VITA_RAW_ENGINE_CACHE_HOST="$OPERATIONAL_ROOT/runtime/engines"
    fi
fi
export VITA_RAW_ENGINE_CACHE_HOST
export VITA_RAW_OUTPUT_HOST="${VITA_RAW_OUTPUT_HOST:-$OPERATIONAL_ROOT/runtime/raw-payload}"

test -f "$COMPOSE_FILE" || { echo "ERROR: missing $COMPOSE_FILE" >&2; exit 1; }
docker image inspect "$BASE_IMAGE" >/dev/null || {
    echo "ERROR: operational parent image is missing: $BASE_IMAGE" >&2
    exit 1
}
test -d "$VITA_RAW_DATA_HOST" || { echo "ERROR: missing raw data: $VITA_RAW_DATA_HOST" >&2; exit 1; }
test -d "$VITA_PROCESSED_DATA_HOST/balkan1/preprocessed" || {
    echo "ERROR: missing processed Balkan data: $VITA_PROCESSED_DATA_HOST/balkan1/preprocessed" >&2
    exit 1
}
test -d "$VITA_RAW_MODEL_ASSETS_HOST" || { echo "ERROR: missing models: $VITA_RAW_MODEL_ASSETS_HOST" >&2; exit 1; }
test -f "$VITA_RAW_ENGINE_CACHE_HOST/tensorrt/direct/accepted.json" || {
    echo "ERROR: accepted TensorRT manifest is missing" >&2
    exit 1
}

raw_root="$(realpath -m "$OPERATIONAL_ROOT/runtime/raw-payload")"
resolved_output="$(realpath -m "$VITA_RAW_OUTPUT_HOST")"
case "$resolved_output/" in
    "$raw_root"/|"$raw_root"/*) ;;
    *) echo "ERROR: raw output must remain below $raw_root" >&2; exit 1 ;;
esac
export VITA_RAW_OUTPUT_HOST="$resolved_output"
mkdir -p "$VITA_RAW_OUTPUT_HOST/runs" "$VITA_RAW_OUTPUT_HOST/home"
test -w "$VITA_RAW_OUTPUT_HOST" || {
    echo "ERROR: raw output is not writable by uid $(id -u): $VITA_RAW_OUTPUT_HOST" >&2
    exit 1
}

operational_id="$(docker ps --filter label=com.docker.compose.project=vita-payload \
    --filter label=com.docker.compose.service=payload --format '{{.ID}}' | head -n 1)"
if [ -n "$operational_id" ]; then
    echo "Operational payload remains running: $operational_id"
else
    echo "WARNING: no running operational vita-payload container was detected." >&2
fi
operational_image_id="$(docker image inspect --format '{{.Id}}' "$BASE_IMAGE")"

cd "$PROJECT_ROOT"
echo "Building the isolated warm raw service from immutable parent $BASE_IMAGE."
docker compose --project-name vita-payload-raw-service --file "$COMPOSE_FILE" build raw-payload
test "$(docker image inspect --format '{{.Id}}' "$BASE_IMAGE")" = "$operational_image_id" || {
    echo "ERROR: operational image identity changed unexpectedly" >&2
    exit 1
}
docker compose --project-name vita-payload-raw-service --file "$COMPOSE_FILE" up \
    --detach --no-deps raw-payload

echo "Waiting for CUDA alignment and TensorRT model warmup on port $VITA_RAW_PAYLOAD_PORT."
for attempt in $(seq 1 180); do
    health="$(docker compose --project-name vita-payload-raw-service --file "$COMPOSE_FILE" \
        ps --format json raw-payload 2>/dev/null || true)"
    if echo "$health" | grep -q '"Health":"healthy"'; then
        echo "Warm raw payload service is ready on Jetson loopback port $VITA_RAW_PAYLOAD_PORT."
        docker compose --project-name vita-payload-raw-service --file "$COMPOSE_FILE" ps
        exit 0
    fi
    sleep 1
done

docker compose --project-name vita-payload-raw-service --file "$COMPOSE_FILE" logs --tail 120 raw-payload
echo "ERROR: warm raw payload service did not become healthy" >&2
exit 1
