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

# Keep the two model backends in one qualified mode. Mixed deployments make the
# health and scientific acceptance record ambiguous.
crop_backend="${VITA_CROP_BACKEND:-pytorch}"
cloud_backend="${VITA_CLOUD_BACKEND:-pytorch}"
if [ "$crop_backend" != "$cloud_backend" ]; then
    echo "ERROR: crop and cloud backends must both be pytorch or both be tensorrt" >&2
    exit 1
fi
case "$crop_backend" in
    pytorch|tensorrt) ;;
    *) echo "ERROR: VITA_CROP_BACKEND must be pytorch or tensorrt" >&2; exit 1 ;;
esac
if [ "$crop_backend" = "tensorrt" ]; then
    case "${VITA_CROP_TRT_PRECISION:-mixed-fp16}" in
        fp32|mixed-fp16) ;;
        *) echo "ERROR: direct crop TensorRT precision must be fp32 or mixed-fp16" >&2; exit 1 ;;
    esac
    if [ "${VITA_CLOUD_INFERENCE_DTYPE:-fp16}" != "${VITA_CLOUD_TRT_PRECISION:-fp16}" ]; then
        echo "ERROR: cloud inference dtype must match VITA_CLOUD_TRT_PRECISION" >&2
        exit 1
    fi
fi
if [ "${VITA_TRT_CUDAGRAPHS:-0}" != "0" ]; then
    echo "ERROR: deploy/payload.env must set VITA_TRT_CUDAGRAPHS=0" >&2
    exit 1
fi

remove_rejected_vita_tensorrt_caches() {
    cache_root="$(realpath -m "$PROJECT_ROOT/runtime/engines/tensorrt")"
    expected_root="$(realpath -m "$PROJECT_ROOT/runtime/engines")"
    case "$cache_root" in
        "$expected_root"/*) ;;
        *)
            echo "ERROR: VITA TensorRT cache escaped the runtime engine root" >&2
            exit 1
            ;;
    esac
    for stale_name in \
        cloud crop crop-fp16 crop-fp32 crop-fp32-no-tf32 crop-fp16-no-tf32 \
        crop-artifacts; do
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
            echo "Removed rejected VITA TensorRT cache: $stale_path"
        fi
    done
    for stale_name in cloud-timing-cache.bin timing-cache.bin; do
        stale_path="$(realpath -m "$cache_root/$stale_name")"
        case "$stale_path" in
            "$cache_root"/*) ;;
            *)
                echo "ERROR: rejected timing cache escaped VITA cache root" >&2
                exit 1
                ;;
        esac
        if [ -f "$stale_path" ]; then
            rm -f -- "$stale_path"
            echo "Removed rejected VITA TensorRT timing cache: $stale_path"
        fi
    done
}

remove_stale_vita_acceptance_runs() {
    runs_root="$(realpath -m "$PROJECT_ROOT/runtime/payload/runs")"
    expected_root="$(realpath -m "$PROJECT_ROOT/runtime/payload")"
    case "$runs_root" in
        "$expected_root"/*) ;;
        *)
            echo "ERROR: VITA acceptance runs escaped the payload runtime root" >&2
            exit 1
            ;;
    esac
    while IFS= read -r -d '' stale_path; do
        resolved_path="$(realpath -m "$stale_path")"
        case "$resolved_path" in
            "$runs_root"/accept-*) ;;
            *)
                echo "ERROR: acceptance output escaped the VITA runs root" >&2
                exit 1
                ;;
        esac
        rm -rf -- "$resolved_path"
        echo "Removed stale VITA acceptance output: $resolved_path"
    done < <(find "$runs_root" -mindepth 1 -maxdepth 1 -type d -name 'accept-*' -print0)
}

./deploy/payload/preflight.sh
mkdir -p runtime/payload/runs runtime/engines/torch-export runtime/engines/tensorrt
remove_rejected_vita_tensorrt_caches
remove_stale_vita_acceptance_runs

compose_args=(-f deploy/compose.payload.yaml)
if [ -f deploy/payload.env ]; then
    compose_args=(--env-file deploy/payload.env "${compose_args[@]}")
fi
if [ "${VITA_SKIP_BUILD:-0}" != "1" ]; then
    docker compose "${compose_args[@]}" build
fi
echo "Validating CUDA, ONNX export, and direct TensorRT inside the final image."
docker compose "${compose_args[@]}" run --rm --no-deps \
    --entrypoint python payload -c \
    'import onnx, onnxscript, tensorrt, torch; assert torch.cuda.is_available(); assert torch.__version__.startswith("2.6.0a0+ecf3bae40a"), torch.__version__; assert torch.version.cuda == "12.8", torch.version.cuda; assert onnx.__version__ == "1.17.0", onnx.__version__; assert onnxscript.__version__ == "0.1.0", onnxscript.__version__; print({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda, "tensorrt": tensorrt.__version__, "onnx": onnx.__version__, "onnxscript": onnxscript.__version__})'
echo "Validating the two VITA-owned writable bind mounts."
docker compose "${compose_args[@]}" run --rm --no-deps \
    --entrypoint python payload -c \
    'from pathlib import Path; roots=(Path("/runtime"), Path("/engine-cache")); probes=[root / ".vita-write-probe" for root in roots]; [probe.write_text("ok", encoding="utf-8") for probe in probes]; [probe.unlink() for probe in probes]; print({"writable_mounts": [str(root) for root in roots]})'
echo "Validating the two Sentinel and two Balkan payload inputs inside the final image."
docker compose "${compose_args[@]}" run --rm --no-deps \
    --entrypoint python payload -m prithvi_payload.deployment_check
if [ "$crop_backend" = "tensorrt" ]; then
    running_payload_id="$(docker compose "${compose_args[@]}" ps --status running -q payload)"
    if [ -n "$running_payload_id" ]; then
        echo "Stopping the existing VITA payload service before exclusive TensorRT tactic search."
        docker compose "${compose_args[@]}" stop payload
    fi
    echo "Building and accepting direct TensorRT plans offline before service startup."
    docker compose "${compose_args[@]}" run --rm --no-deps \
        --entrypoint python payload -m prithvi_payload.tensorrt_builder
fi
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
        if ! docker compose "${compose_args[@]}" exec -T payload \
            python -m prithvi_payload.deployment_acceptance \
            --url http://127.0.0.1:8090/healthz; then
            fail_startup "payload acceleration acceptance failed"
        fi
        if [ "${VITA_SKIP_PERFORMANCE_ACCEPTANCE:-0}" != "1" ]; then
            if ! docker compose "${compose_args[@]}" exec -T payload \
                python -m prithvi_payload.performance_acceptance \
                --url http://127.0.0.1:8090/v1/jobs; then
                fail_startup "payload two-second performance acceptance failed"
            fi
        else
            echo "WARNING: performance acceptance was explicitly skipped; deployment is not SLO-qualified." >&2
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
