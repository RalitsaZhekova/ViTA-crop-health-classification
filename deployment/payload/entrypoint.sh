#!/bin/sh
set -eu

if [ -z "${EE_PROJECT_ID:-}" ]; then
  echo "EE_PROJECT_ID is required" >&2
  exit 2
fi

if [ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]; then
  echo "GOOGLE_APPLICATION_CREDENTIALS is required" >&2
  exit 2
fi

if [ ! -f "${GOOGLE_APPLICATION_CREDENTIALS}" ] || [ ! -r "${GOOGLE_APPLICATION_CREDENTIALS}" ]; then
  echo "Earth Engine runtime credential is not a readable regular file" >&2
  exit 2
fi

if [ "${VITA_MAX_CONCURRENT_JOBS:-1}" != "1" ]; then
  echo "VITA_MAX_CONCURRENT_JOBS must be 1" >&2
  exit 2
fi

for directory in "${VITA_JOBS_DIR:-/data/jobs}" "${VITA_OUTBOX_DIR:-/data/outbox}" "${VITA_LOGS_DIR:-/data/logs}"; do
  mkdir -p "${directory}"
  if [ ! -d "${directory}" ] || [ ! -w "${directory}" ]; then
    echo "A required payload runtime directory is not writable" >&2
    exit 2
  fi
done

python -m prithvi_payload.container_preflight
exec "$@"
