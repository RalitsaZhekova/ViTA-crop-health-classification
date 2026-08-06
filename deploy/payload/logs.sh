#!/usr/bin/env bash
set -euo pipefail
cd "${VITA_PROJECT_ROOT:-/data/code/VITA}"
compose_args=(-f deploy/compose.payload.yaml)
if [ -f deploy/payload.env ]; then
    compose_args=(--env-file deploy/payload.env "${compose_args[@]}")
fi
docker compose "${compose_args[@]}" logs --follow --tail 200 payload
