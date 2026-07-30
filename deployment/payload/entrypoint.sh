#!/bin/sh
set -eu

if [ "${EE_PROJECT_ID:-}" != "vita-503208" ]; then
  echo "EE_PROJECT_ID must be the fixed ViTA Earth Engine project" >&2
  exit 2
fi

if [ "${GOOGLE_APPLICATION_CREDENTIALS:-}" != "/run/secrets/earth_engine_credentials" ]; then
  echo "GOOGLE_APPLICATION_CREDENTIALS must reference the runtime secret mount" >&2
  exit 2
fi

if [ ! -r "${GOOGLE_APPLICATION_CREDENTIALS}" ]; then
  echo "Earth Engine runtime credential file is not readable" >&2
  exit 2
fi

exec python -m prithvi_payload.service --host 0.0.0.0 --port 8081
