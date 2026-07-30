#!/bin/sh
set -eu

store="${VITA_GROUND_STORE:-/data/ground}"
mkdir -p "${store}"
if [ ! -d "${store}" ] || [ ! -w "${store}" ]; then
  echo "Ground runtime directory is not writable" >&2
  exit 2
fi

exec "$@"
