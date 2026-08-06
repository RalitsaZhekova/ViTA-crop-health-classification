#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${VITA_PROJECT_ROOT:-/data/code/VITA}"
cd "$PROJECT_ROOT"

if [ -f deploy/payload.env ]; then
    set -a
    # shellcheck disable=SC1091
    source deploy/payload.env
    set +a
fi

./deploy/payload/preflight.sh
mkdir -p runtime/payload/runs runtime/engines/torch-export runtime/engines/tensorrt

compose_args=(-f deploy/compose.payload.yaml)
if [ -f deploy/payload.env ]; then
    compose_args=(--env-file deploy/payload.env "${compose_args[@]}")
fi
if [ "${VITA_SKIP_BUILD:-0}" != "1" ]; then
    docker compose "${compose_args[@]}" build
fi
docker compose "${compose_args[@]}" up -d

container_id="$(docker compose "${compose_args[@]}" ps -q payload)"
test -n "$container_id" || { echo "ERROR: payload container was not created" >&2; exit 1; }

for attempt in $(seq 1 360); do
    status="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$container_id")"
    if [ "$status" = "healthy" ]; then
        curl --fail --silent http://127.0.0.1:${VITA_PAYLOAD_PORT:-8090}/healthz
        echo
        echo "Payload service is ready on Jetson loopback."
        exit 0
    fi
    if [ "$status" = "unhealthy" ] || [ "$status" = "exited" ]; then
        docker compose "${compose_args[@]}" logs --tail 200 payload
        exit 1
    fi
    sleep 5
done

docker compose "${compose_args[@]}" logs --tail 200 payload
echo "ERROR: payload did not become healthy within 30 minutes" >&2
exit 1
