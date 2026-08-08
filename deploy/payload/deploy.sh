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

# Match the service identity to the owner of the payload-local bind mounts.
# The NVIDIA runtime supplies the required video/render device groups.
export VITA_PAYLOAD_UID="${VITA_PAYLOAD_UID:-$(id -u)}"
export VITA_PAYLOAD_GID="${VITA_PAYLOAD_GID:-$(id -g)}"

# These are release acceptance settings, not tuning knobs. Fail with a useful
# message when an older copied payload.env silently overrides the Compose image.
if [ "${VITA_CROP_TRT_PRECISION:-fp16}" != "fp16" ]; then
    echo "ERROR: deploy/payload.env must set VITA_CROP_TRT_PRECISION=fp16" >&2
    exit 1
fi
if [ "${VITA_CROP_TRT_MAX_MEAN_PROBABILITY_ERROR:-0.01}" != "0.01" ]; then
    echo "ERROR: deploy/payload.env must set VITA_CROP_TRT_MAX_MEAN_PROBABILITY_ERROR=0.01" >&2
    exit 1
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
echo "Validating the two VITA-owned writable bind mounts."
docker compose "${compose_args[@]}" run --rm --no-deps \
    --entrypoint python payload -c \
    'from pathlib import Path; roots=(Path("/runtime"), Path("/engine-cache")); probes=[root / ".vita-write-probe" for root in roots]; [probe.write_text("ok", encoding="utf-8") for probe in probes]; [probe.unlink() for probe in probes]; print({"writable_mounts": [str(root) for root in roots]})'
echo "Validating the two Sentinel and two Balkan payload inputs inside the final image."
docker compose "${compose_args[@]}" run --rm --no-deps \
    --entrypoint python payload -m prithvi_payload.deployment_check
docker compose "${compose_args[@]}" up -d

container_id="$(docker compose "${compose_args[@]}" ps -q payload)"
test -n "$container_id" || { echo "ERROR: payload container was not created" >&2; exit 1; }

fail_startup() {
    message="$1"
    docker compose "${compose_args[@]}" logs --tail 200 payload || true
    # This Compose project is explicitly named "vita-payload". Removing its
    # failed service container/network prevents an endless restart loop without
    # touching images, bind-mounted caches/data, or another team's project.
    docker compose "${compose_args[@]}" down || true
    echo "ERROR: $message" >&2
    exit 1
}

remove_rejected_crop_engine_caches() {
    cache_root="$(realpath -m "$PROJECT_ROOT/runtime/engines/tensorrt")"
    for stale_name in crop crop-fp16 crop-fp32 crop-fp32-no-tf32 crop-fp16-no-tf32; do
        stale_path="$(realpath -m "$cache_root/$stale_name")"
        case "$stale_path" in
            "$cache_root"/*) ;;
            *)
                echo "ERROR: rejected engine cache escaped VITA cache root" >&2
                exit 1
                ;;
        esac
        if [ -d "$stale_path" ]; then
            rm -rf -- "$stale_path"
            echo "Removed rejected VITA crop engine cache: $stale_path"
        fi
    done
}

startup_timeout="${VITA_PAYLOAD_STARTUP_TIMEOUT_SECONDS:-5400}"
test "$startup_timeout" -ge 60 || {
    echo "ERROR: VITA_PAYLOAD_STARTUP_TIMEOUT_SECONDS must be at least 60" >&2
    exit 1
}
attempts=$((startup_timeout / 5))
for attempt in $(seq 1 "$attempts"); do
    status="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$container_id")"
    container_state="$(docker inspect --format '{{.State.Status}}' "$container_id")"
    restart_count="$(docker inspect --format '{{.RestartCount}}' "$container_id")"
    if [ "$status" = "healthy" ]; then
        docker compose "${compose_args[@]}" exec -T payload \
            python -m prithvi_payload.deployment_acceptance \
            --url http://127.0.0.1:8090/healthz
        if [ "${VITA_SKIP_PERFORMANCE_ACCEPTANCE:-0}" != "1" ]; then
            docker compose "${compose_args[@]}" exec -T payload \
                python -m prithvi_payload.performance_acceptance \
                --url http://127.0.0.1:8090/v1/jobs
            remove_rejected_crop_engine_caches
        fi
        echo "Payload service is ready on Jetson loopback."
        exit 0
    fi
    if [ "$status" = "unhealthy" ] || [ "$container_state" = "exited" ] \
        || [ "$container_state" = "dead" ] || [ "$container_state" = "restarting" ] \
        || [ "$restart_count" -gt 0 ]; then
        fail_startup "payload failed during startup"
    fi
    sleep 5
done

fail_startup "payload did not become healthy within ${startup_timeout} seconds"
