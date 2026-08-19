#!/usr/bin/env bash
set -euo pipefail

OPERATIONAL_ROOT="${VITA_OPERATIONAL_PROJECT_ROOT:-/data/code/VITA}"
PROJECT_ROOT="${VITA_RAW_PROJECT_ROOT:-$OPERATIONAL_ROOT/runtime/worktrees/raw-band-jetson-isolated}"
COMPOSE_FILE="$PROJECT_ROOT/deploy/compose.payload.raw.yaml"
RAW_IMAGE="${VITA_RAW_PAYLOAD_IMAGE:-vita-payload:raw-band}"
BASE_IMAGE="${VITA_RAW_PAYLOAD_BASE_IMAGE:-vita-payload:1.0.0}"

usage() {
    cat <<'EOF'
Usage: run-raw.sh SCENE_ID JOB_ID [REGION_ID]

The standard scene layout is resolved below /data:
  balkan1/raw/SCENE_ID/SCENE_ID_Raw.tif
  balkan1/raw/SCENE_ID/{position,attitude}.csv
  balkan1/derived/l1a/SCENE_ID_L0R_manifest.json
  balkan1/derived/l1a/SCENE_ID_L1A_reference_validation.json
  balkan1/preprocessed/SCENE_ID_L1ORT.crop_calibration.json

Override any source with VITA_RAW_INPUT, VITA_RAW_METADATA,
VITA_RAW_RADIOMETRIC_DIAGNOSTICS, VITA_RAW_POSITION, VITA_RAW_ATTITUDE, or
VITA_RAW_PARENT_CALIBRATION. Outputs are isolated below runtime/raw-payload.
EOF
}

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    usage >&2
    exit 2
fi
SCENE_ID="$1"
JOB_ID="$2"
REGION_ID="${3:-balkan-raw-$SCENE_ID}"
for identifier in "$SCENE_ID" "$JOB_ID" "$REGION_ID"; do
    case "$identifier" in
        ""|*[!A-Za-z0-9._-]*)
            echo "ERROR: scene, job and region IDs may contain only letters, digits, dot, underscore or dash" >&2
            exit 2
            ;;
    esac
done

cd "$PROJECT_ROOT"
test -f "$COMPOSE_FILE" || { echo "ERROR: missing $COMPOSE_FILE" >&2; exit 1; }
docker image inspect "$BASE_IMAGE" >/dev/null || {
    echo "ERROR: operational parent image is missing: $BASE_IMAGE" >&2
    exit 1
}

export VITA_RAW_HOST_UID="${VITA_RAW_HOST_UID:-$(id -u)}"
export VITA_RAW_HOST_GID="${VITA_RAW_HOST_GID:-$(id -g)}"
export VITA_RAW_PAYLOAD_IMAGE="$RAW_IMAGE"
export VITA_RAW_PAYLOAD_BASE_IMAGE="$BASE_IMAGE"
export VITA_RAW_DATA_HOST="${VITA_RAW_DATA_HOST:-$OPERATIONAL_ROOT/data/raw-inputs}"
export VITA_RAW_MODEL_ASSETS_HOST="${VITA_RAW_MODEL_ASSETS_HOST:-$OPERATIONAL_ROOT/payload/models}"
export VITA_RAW_ENGINE_CACHE_HOST="${VITA_RAW_ENGINE_CACHE_HOST:-$OPERATIONAL_ROOT/runtime/engines}"
export VITA_RAW_OUTPUT_HOST="${VITA_RAW_OUTPUT_HOST:-$OPERATIONAL_ROOT/runtime/raw-payload}"
raw_output_root="$(realpath -m "$OPERATIONAL_ROOT/runtime/raw-payload")"
resolved_output="$(realpath -m "$VITA_RAW_OUTPUT_HOST")"
case "$resolved_output/" in
    "$raw_output_root"/|"$raw_output_root"/*) ;;
    *) echo "ERROR: raw output must remain below $raw_output_root" >&2; exit 1 ;;
esac
export VITA_RAW_OUTPUT_HOST="$resolved_output"

RAW_INPUT="${VITA_RAW_INPUT:-/data/balkan1/raw/$SCENE_ID/${SCENE_ID}_Raw.tif}"
RAW_METADATA="${VITA_RAW_METADATA:-/data/balkan1/derived/l1a/${SCENE_ID}_L0R_manifest.json}"
RAW_DIAGNOSTICS="${VITA_RAW_RADIOMETRIC_DIAGNOSTICS:-/data/balkan1/derived/l1a/${SCENE_ID}_L1A_reference_validation.json}"
RAW_POSITION="${VITA_RAW_POSITION:-/data/balkan1/raw/$SCENE_ID/position.csv}"
RAW_ATTITUDE="${VITA_RAW_ATTITUDE:-/data/balkan1/raw/$SCENE_ID/attitude.csv}"
RAW_CALIBRATION="${VITA_RAW_PARENT_CALIBRATION:-/data/balkan1/preprocessed/${SCENE_ID}_L1ORT.crop_calibration.json}"

mkdir -p "$VITA_RAW_OUTPUT_HOST/runs"
test -w "$VITA_RAW_OUTPUT_HOST/runs" || {
    echo "ERROR: $VITA_RAW_OUTPUT_HOST/runs is not writable by uid $(id -u)" >&2
    exit 1
}
test -d "$VITA_RAW_DATA_HOST" || { echo "ERROR: raw data root is missing: $VITA_RAW_DATA_HOST" >&2; exit 1; }
test -d "$VITA_RAW_MODEL_ASSETS_HOST" || { echo "ERROR: model root is missing: $VITA_RAW_MODEL_ASSETS_HOST" >&2; exit 1; }
test -f "$VITA_RAW_ENGINE_CACHE_HOST/tensorrt/direct/accepted.json" || {
    echo "ERROR: accepted TensorRT manifest is missing; operational cache was not modified" >&2
    exit 1
}
for container_path in \
    "$RAW_INPUT" "$RAW_METADATA" "$RAW_DIAGNOSTICS" \
    "$RAW_POSITION" "$RAW_ATTITUDE" "$RAW_CALIBRATION"; do
    case "$container_path" in
        /data/*) host_path="$VITA_RAW_DATA_HOST/${container_path#/data/}" ;;
        *) echo "ERROR: raw inputs must be absolute paths below /data: $container_path" >&2; exit 1 ;;
    esac
    test -f "$host_path" || { echo "ERROR: required raw input is missing: $host_path" >&2; exit 1; }
done

operational_id="$(docker ps --filter label=com.docker.compose.project=vita-payload \
    --filter label=com.docker.compose.service=payload --format '{{.ID}}' | head -n 1)"
operational_image_id="$(docker image inspect --format '{{.Id}}' "$BASE_IMAGE")"
if [ -n "$operational_id" ]; then
    operational_state="$(docker inspect --format '{{.State.Status}}' "$operational_id")"
    test "$operational_state" = "running" || {
        echo "ERROR: operational payload container is present but not running" >&2
        exit 1
    }
    echo "Operational payload remains running: $operational_id"
else
    echo "WARNING: no running operational vita-payload container was detected." >&2
fi

echo "Building isolated overlay $RAW_IMAGE from immutable parent $BASE_IMAGE."
docker compose --project-name vita-payload-raw --file "$COMPOSE_FILE" build raw-payload
test "$(docker image inspect --format '{{.Id}}' "$BASE_IMAGE")" = "$operational_image_id" || {
    echo "ERROR: operational image identity changed unexpectedly" >&2
    exit 1
}

lock_file="/tmp/vita-raw-payload.lock"
exec 9>"$lock_file"
flock --nonblock 9 || {
    echo "ERROR: another isolated raw payload job is already running" >&2
    exit 1
}

echo "Running isolated raw job $JOB_ID; operational image/container are not stopped or replaced."
docker compose --project-name vita-payload-raw --file "$COMPOSE_FILE" run --rm --no-deps \
    raw-payload \
    python -m prithvi_payload.raw_payload_workload \
    --raw "$RAW_INPUT" \
    --metadata "$RAW_METADATA" \
    --radiometric-diagnostics "$RAW_DIAGNOSTICS" \
    --position "$RAW_POSITION" \
    --attitude "$RAW_ATTITUDE" \
    --parent-calibration "$RAW_CALIBRATION" \
    --output-root /runtime/runs \
    --job-id "$JOB_ID" \
    --region-id "$REGION_ID" \
    --band-start-row-scale "${VITA_RAW_BAND_START_ROW_SCALE:-0.19}" \
    --band-start-axis "${VITA_RAW_BAND_START_AXIS:-column}" \
    --warp-tile-size "${VITA_RAW_WARP_TILE_SIZE:-2048}" \
    --compression "${VITA_RAW_ALIGNMENT_COMPRESSION:-zstd}" \
    --condition-tile-size "${VITA_CONDITION_TILE_SIZE:-4096}"

if [ -n "$operational_id" ]; then
    test "$(docker inspect --format '{{.State.Status}}' "$operational_id")" = "running" || {
        echo "ERROR: operational payload did not remain running" >&2
        exit 1
    }
fi
echo "Raw downlink bundle: $VITA_RAW_OUTPUT_HOST/runs/$JOB_ID/downlink"
