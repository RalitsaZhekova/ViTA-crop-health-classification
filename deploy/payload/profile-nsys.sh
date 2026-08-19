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

export VITA_PAYLOAD_UID="${VITA_PAYLOAD_UID:-$(id -u)}"
export VITA_PAYLOAD_GID="${VITA_PAYLOAD_GID:-$(id -g)}"
payload_image="${VITA_PAYLOAD_IMAGE:-vita-payload:1.0.0}"
export VITA_NSYS_IMAGE="${VITA_NSYS_IMAGE:-vita-payload:nsight}"

nsys_host_bin="${VITA_NSYS_BIN:-}"
if [ -z "$nsys_host_bin" ] && [ -d /opt/nvidia/nsight-systems ]; then
    # CUDA toolkit launchers can exist without exposing the profiler payload
    # inside a container. Prefer the newest directly installed Nsight CLI.
    nsys_host_bin="$(
        find /opt/nvidia/nsight-systems \
            -mindepth 3 -maxdepth 3 -path '*/bin/nsys' -executable \
            -print 2>/dev/null | sort -V | tail -n 1
    )"
fi
if [ -z "$nsys_host_bin" ]; then
    nsys_host_bin="$(command -v nsys || true)"
fi
if [ -z "$nsys_host_bin" ]; then
    echo "ERROR: nsys is not installed on the Jetson host or is not on PATH." >&2
    exit 1
fi
nsys_host_bin="$(readlink -f "$nsys_host_bin")"
if [ ! -x "$nsys_host_bin" ]; then
    echo "ERROR: Nsight Systems CLI is not executable: $nsys_host_bin" >&2
    exit 1
fi
if [[ "$nsys_host_bin" == /opt/nvidia/nsight-systems/* ]]; then
    export VITA_NSYS_HOST_ROOT=/opt/nvidia/nsight-systems
else
    nsys_default_root="$(dirname "$(dirname "$nsys_host_bin")")"
    export VITA_NSYS_HOST_ROOT="${VITA_NSYS_HOST_ROOT:-$nsys_default_root}"
fi
case "$nsys_host_bin" in
    "$VITA_NSYS_HOST_ROOT"/*) ;;
    *)
        echo "ERROR: nsys must be below VITA_NSYS_HOST_ROOT ($VITA_NSYS_HOST_ROOT)." >&2
        exit 1
        ;;
esac
nsys_container_bin="$nsys_host_bin"

report_root="$PROJECT_ROOT/runtime/nsight"
mkdir -p "$report_root"
runs_root="$PROJECT_ROOT/runtime/payload/runs"
mkdir -p "$runs_root"
run_stamp="${VITA_NSYS_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
report_base="vita-four-scenes-${run_stamp}"
report_path="$report_root/$report_base.nsys-rep"
workload_report="/profiles/$report_base.workload.json"

if [ -e "$report_path" ] && [ "${VITA_NSYS_OVERWRITE:-0}" != "1" ]; then
    echo "ERROR: report already exists: $report_path" >&2
    echo "Set VITA_NSYS_OVERWRITE=1 or choose a different VITA_NSYS_RUN_ID." >&2
    exit 1
fi

echo "Nsight Systems CLI:"
"$nsys_host_bin" --version
echo "Building a small profiling overlay on the existing payload image: $payload_image"
docker build \
    --file deploy/Dockerfile.payload-overlay \
    --build-arg "VITA_PAYLOAD_OVERLAY_BASE_IMAGE=$payload_image" \
    --build-arg "VITA_PAYLOAD_UID=$VITA_PAYLOAD_UID" \
    --build-arg "VITA_PAYLOAD_GID=$VITA_PAYLOAD_GID" \
    --tag "$VITA_NSYS_IMAGE" \
    .

base_compose=(-f deploy/compose.payload.yaml)
profile_compose=(-f deploy/compose.payload.yaml -f deploy/compose.payload.nsys.yaml)
if [ -f deploy/payload.env ]; then
    base_compose=(--env-file deploy/payload.env "${base_compose[@]}")
    profile_compose=(--env-file deploy/payload.env "${profile_compose[@]}")
fi

service_was_running=0
if [ -n "$(docker compose "${base_compose[@]}" ps --status running -q payload)" ]; then
    service_was_running=1
fi
report_root_mode="$(stat -c '%a' "$report_root")"
runs_root_mode="$(stat -c '%a' "$runs_root")"
permissions_relaxed=0
restore_state() {
    if [ "$permissions_relaxed" -eq 1 ]; then
        chmod "$report_root_mode" "$report_root" || true
        chmod "$runs_root_mode" "$runs_root" || true
    fi
    if [ "$service_was_running" -eq 1 ]; then
        echo "Restoring the operational payload service."
        docker compose "${base_compose[@]}" start payload >/dev/null
    fi
}
trap restore_state EXIT

# The privileged profiler container can use a different host UID when Docker
# user namespaces are enabled. Open only the two disposable output directories
# for the capture and restore their exact original modes on exit.
permissions_relaxed=1
if ! chmod o+rwx "$report_root" "$runs_root"; then
    echo "ERROR: cannot make the Nsight output directories container-writable." >&2
    echo "Ensure the current user owns $report_root and $runs_root." >&2
    exit 1
fi

if [ "$service_was_running" -eq 1 ]; then
    echo "Stopping the operational payload service to avoid GPU contention during capture."
    docker compose "${base_compose[@]}" stop payload
fi

echo "Checking profiler access from the isolated profiling container."
docker compose "${profile_compose[@]}" run --rm --no-deps \
    --entrypoint "$nsys_container_bin" payload status --environment
docker compose "${profile_compose[@]}" run --rm --no-deps \
    --entrypoint /bin/bash payload -c '
        set -e
        profile_marker=/profiles/.vita-nsys-write-test-$$
        runtime_marker=/runtime/runs/.vita-nsys-write-test-$$
        : > "$profile_marker"
        mkdir "$runtime_marker"
        rm "$profile_marker"
        rmdir "$runtime_marker"
    '

profile_help="$($nsys_host_bin profile --help 2>&1)"
require_option() {
    if ! grep -q -- "$1" <<<"$profile_help"; then
        echo "ERROR: installed Nsight Systems does not support required option $1" >&2
        exit 1
    fi
}
for option in \
    --capture-range \
    --capture-range-end \
    --cpuctxsw \
    --sampling-trigger; do
    require_option "$option"
done

nsys_options=(
    profile
    --force-overwrite=true
    --output="/profiles/$report_base"
    --trace=cuda,nvtx,osrt,cudnn,cublas
    --capture-range=cudaProfilerApi
    --capture-range-end=stop
    --sample=process-tree
    --sampling-trigger=cuda
    --cpuctxsw=process-tree
    --show-output=true
)
if grep -q -- "--accelerator-trace" <<<"$profile_help"; then
    nsys_options+=(--accelerator-trace=tegra-accelerators)
fi
if grep -q -- "--syscall" <<<"$profile_help"; then
    nsys_options+=(--syscall=process-tree)
fi
if grep -q -- "--osrt-file-access" <<<"$profile_help"; then
    nsys_options+=(--osrt-file-access=true)
fi
if grep -q -- "--cuda-event-trace" <<<"$profile_help"; then
    nsys_options+=(--cuda-event-trace=false)
fi

echo "Capturing one warm pass over two Sentinel and two Balkan scenes."
docker compose "${profile_compose[@]}" run --rm --no-deps \
    --env "VITA_NSYS_WORKLOAD_REPORT=$workload_report" \
    --entrypoint "$nsys_container_bin" payload \
    "${nsys_options[@]}" \
    python -m prithvi_payload.nsight_workload --run-id "$run_stamp"

if [ ! -s "$report_path" ]; then
    echo "ERROR: Nsight Systems did not produce $report_path" >&2
    exit 1
fi

echo "Exporting CUDA, NVTX, OS-runtime, and kernel summaries."
stats_reports=(cuda_gpu_kern_sum cuda_api_sum nvtx_sum osrt_sum)
for report in "${stats_reports[@]}"; do
    "$nsys_host_bin" stats \
        --force-overwrite=true \
        --report "$report" \
        --format csv \
        --output "$report_root/${report_base}-${report}" \
        "$report_path"
done

echo "Nsight capture ready: $report_path"
echo "Open it on the workstation with an Nsight Systems GUI of the same or newer version."
