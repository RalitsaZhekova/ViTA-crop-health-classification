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

cd "$PROJECT_ROOT"
test "$(uname -m)" = "aarch64" || { echo "ERROR: payload host must be aarch64" >&2; exit 1; }
command -v docker >/dev/null || { echo "ERROR: docker is not installed" >&2; exit 1; }
docker compose version
docker info --format 'Docker={{.ServerVersion}} Runtimes={{json .Runtimes}}'
test -f /etc/nv_tegra_release || { echo "ERROR: /etc/nv_tegra_release is missing" >&2; exit 1; }
head -n 1 /etc/nv_tegra_release
dpkg-query --show --showformat='JetPack=${Version}\n' nvidia-jetpack 2>/dev/null || true

docker image inspect "$BASE_IMAGE" >/dev/null || {
    echo "ERROR: expected installed base image not found: $BASE_IMAGE" >&2
    echo "Set VITA_PAYLOAD_BASE_IMAGE to the exact local NVIDIA image tag." >&2
    exit 1
}
docker image inspect --format 'Base image={{index .RepoTags 0}} architecture={{.Architecture}} digest={{.Id}}' "$BASE_IMAGE"

test -f payload/models/prithvi_crop_binary_single_frame_v1_weights.pt || {
    echo "ERROR: crop weights are missing under payload/models" >&2
    exit 1
}
test -f payload/models/omnicloudmask/PM_model_OCM_7.97_R_G_NIR_3_smp_edgenext_small.usi_in1k_PT_state.safetensors
test -f payload/models/omnicloudmask/PM_model_OCM_7.97_R_G_NIR_3_smp_regnety_004.pycls_in1k_PT_state.safetensors
printf '%s  %s\n' \
  c948977bffdaeb89ecf4f4d069db13c7ed81d4f3403eec9e257c45c235b1484e \
  payload/models/prithvi_crop_binary_single_frame_v1_weights.pt \
  d5fe67ad00f6fdb73eb8382ad925849e97fb82cedb46225344afff1a734a9c1d \
  payload/models/omnicloudmask/PM_model_OCM_7.97_R_G_NIR_3_smp_edgenext_small.usi_in1k_PT_state.safetensors \
  7f6e4202e17ee73efa4aba7abb5c34f4f90a9f7eb42480820714994dff2db660 \
  payload/models/omnicloudmask/PM_model_OCM_7.97_R_G_NIR_3_smp_regnety_004.pycls_in1k_PT_state.safetensors \
  | sha256sum --check --strict

docker run --rm --runtime nvidia --entrypoint python "$BASE_IMAGE" -c \
  'import tensorrt, torch, torch_tensorrt; assert torch.cuda.is_available(); print({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "tensorrt": tensorrt.__version__, "torch_tensorrt": torch_tensorrt.__version__})'

echo "Payload preflight passed."
