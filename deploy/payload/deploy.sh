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
echo "Validating CUDA and Torch-TensorRT inside the final image through the NVIDIA runtime."
docker compose "${compose_args[@]}" run --rm --no-deps \
    --entrypoint python payload -c \
    'import tensorrt, torch, torch_tensorrt; assert torch.cuda.is_available(); assert torch.__version__.startswith("2.6.0a0+ecf3bae40a"), torch.__version__; assert torch.version.cuda == "12.8", torch.version.cuda; assert tensorrt.__version__.startswith("10.8."), tensorrt.__version__; assert torch_tensorrt.__version__.startswith("2.6.0a0"), torch_tensorrt.__version__; print({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda, "tensorrt": tensorrt.__version__, "torch_tensorrt": torch_tensorrt.__version__})'
echo "Validating the two Sentinel and two Balkan payload inputs inside the final image."
docker compose "${compose_args[@]}" run --rm --no-deps \
    --entrypoint python payload -m prithvi_payload.deployment_check
docker compose "${compose_args[@]}" up -d

container_id="$(docker compose "${compose_args[@]}" ps -q payload)"
test -n "$container_id" || { echo "ERROR: payload container was not created" >&2; exit 1; }

startup_timeout="${VITA_PAYLOAD_STARTUP_TIMEOUT_SECONDS:-5400}"
test "$startup_timeout" -ge 60 || {
    echo "ERROR: VITA_PAYLOAD_STARTUP_TIMEOUT_SECONDS must be at least 60" >&2
    exit 1
}
attempts=$((startup_timeout / 5))
for attempt in $(seq 1 "$attempts"); do
    status="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$container_id")"
    if [ "$status" = "healthy" ]; then
        docker compose "${compose_args[@]}" exec -T payload \
            python -m prithvi_payload.deployment_acceptance \
            --url http://127.0.0.1:8090/healthz
        if [ "${VITA_SKIP_PERFORMANCE_ACCEPTANCE:-0}" != "1" ]; then
            docker compose "${compose_args[@]}" exec -T payload \
                python -m prithvi_payload.performance_acceptance \
                --url http://127.0.0.1:8090/v1/jobs
        fi
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
echo "ERROR: payload did not become healthy within ${startup_timeout} seconds" >&2
exit 1
