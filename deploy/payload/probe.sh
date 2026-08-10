#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="${VITA_PROJECT_ROOT:-/data/code/VITA}"
BASE_IMAGE="${VITA_PAYLOAD_BASE_IMAGE:-nvcr.io/nvidia/pytorch:25.01-py3-igpu}"
failures=0

section() {
    printf '\n[%s]\n' "$1"
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "MISSING: $1"
        failures=$((failures + 1))
        return 1
    fi
}

section host
echo "project_root=$PROJECT_ROOT"
uname -a
echo "architecture=$(uname -m)"
echo "device=$(tr -d '\0' </proc/device-tree/model 2>/dev/null || echo unknown)"
head -n 1 /etc/nv_tegra_release 2>/dev/null || {
    echo 'MISSING: /etc/nv_tegra_release'
    failures=$((failures + 1))
}
dpkg-query --show --showformat='nvidia-jetpack=${Version}\n' nvidia-jetpack 2>/dev/null || \
    echo 'nvidia-jetpack meta-package is not installed'
grep -E '^(MemTotal|SwapTotal):' /proc/meminfo
df -h "$PROJECT_ROOT" 2>/dev/null || true

section power
if command -v nvpmodel >/dev/null 2>&1; then
    nvpmodel -q 2>&1 || true
else
    echo 'MISSING: nvpmodel'
fi
if command -v jetson_clocks >/dev/null 2>&1; then
    jetson_clocks --show 2>&1 || true
else
    echo 'MISSING: jetson_clocks'
fi

section docker
if require_command docker; then
    docker version 2>&1 || failures=$((failures + 1))
    docker compose version 2>&1 || failures=$((failures + 1))
    docker info --format 'server={{.ServerVersion}} runtimes={{json .Runtimes}} root={{.DockerRootDir}}' \
        2>&1 || failures=$((failures + 1))
    docker image ls --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}' | \
        grep -E '(^|/)pytorch:|l4t|deepstream' || true
fi

section base_image
echo "requested=$BASE_IMAGE"
if docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
    docker image inspect --format \
        'architecture={{.Architecture}} id={{.Id}} repo_digests={{json .RepoDigests}}' \
        "$BASE_IMAGE"
    if ! docker run --rm --runtime nvidia --entrypoint python "$BASE_IMAGE" -c \
        'import json, numpy, platform, tensorrt, torch; print(json.dumps({"machine": platform.machine(), "python": platform.python_version(), "numpy": numpy.__version__, "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "tensorrt": tensorrt.__version__}, sort_keys=True)); assert torch.cuda.is_available()'; then
        failures=$((failures + 1))
    fi
else
    echo "MISSING: $BASE_IMAGE"
    failures=$((failures + 1))
fi

section build_network
if command -v curl >/dev/null 2>&1; then
    for url in https://pypi.org/simple/ https://ports.ubuntu.com/ubuntu-ports/; do
        if curl --head --fail --silent --show-error --max-time 10 "$url" >/dev/null; then
            echo "reachable=$url"
        else
            echo "WARNING: build endpoint is not reachable: $url"
        fi
    done
else
    echo 'WARNING: curl is missing; build-network reachability was not tested'
fi

section result
if [ "$failures" -ne 0 ]; then
    echo "probe_status=FAILED failures=$failures"
    exit 1
fi
echo 'probe_status=PASSED'
