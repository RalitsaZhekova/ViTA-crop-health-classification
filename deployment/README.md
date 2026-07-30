# ViTA split deployment

This deployment keeps the scientific boundary unchanged:

```text
ground mission CLI -> payload HTTP API -> Earth Engine and all science
                   <- scene.json + scene.webp + condition.png only
ground checksum/schema validation -> immutable catalog -> API/dashboard
```

The payload image contains the shared package, payload package, fixed Earth
Engine provider, CloudSEN runtime, Prithvi crop runtime and exactly the selected
checksum-pinned model artifacts. The ground image contains the shared and
ground packages plus the dashboard assets. It contains no payload package,
Earth Engine client, model weights or scientific execution code.

## Project-local layout

Generated data and credentials remain below the repository:

```text
secrets/earth-engine.json       # manually placed; ignored; never build context
runtime/payload/jobs/           # persistent payload job state and products
runtime/payload/outbox/         # reserved payload transfer staging
runtime/payload/logs/           # application/cache logs
runtime/ground/                 # immutable ground catalog and SQLite database
```

Create the directories and local configuration from PowerShell:

```powershell
New-Item -ItemType Directory -Force secrets | Out-Null
New-Item -ItemType Directory -Force runtime\payload\jobs | Out-Null
New-Item -ItemType Directory -Force runtime\payload\outbox | Out-Null
New-Item -ItemType Directory -Force runtime\payload\logs | Out-Null
New-Item -ItemType Directory -Force runtime\ground | Out-Null
Copy-Item .env.payload.example .env.payload
Copy-Item .env.ground.example .env.ground
```

The Earth Engine key must then be placed manually at
`secrets/earth-engine.json`. Do not print it or pass it as a build argument or
environment value.

The images run without root privileges. `VITA_CONTAINER_UID` and
`VITA_CONTAINER_GID` default to `1000`; on Linux, set them to the output of
`id -u` and `id -g` so the service can write the user-owned project runtime
directories without changing host ownership or permissions.

## Base-image rule

`deployment/payload/Dockerfile` is shared by local x86 CUDA and Jetson. The
selected base must already provide compatible Python, PyTorch, torchvision,
CUDA and target GPU support. The build records and compares protected package
versions and fails if dependency installation replaces PyTorch, torchvision,
OpenCV or CuPy.

For the detected local Windows AMD64/RTX 3060 platform, the initial build
candidate is:

```text
pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
```

It is not considered qualified until the image builds, CUDA is visible in the
container, both model checksums pass, both models warm, and the regression
mission matches `deployment/local_regression_reference.json`. The environment
example intentionally retains `REPLACE_WITH_COMPATIBLE_BASE_IMAGE` so an
unverified image is never selected silently.

For Jetson, do not reuse the x86 image or choose a tag from the host name. The
base must match the exact JetPack/L4T release, `aarch64`, CUDA ABI and an
NVIDIA-supported PyTorch build. Inspect the target first as documented below.

## Ignore and secret checks

Run without opening the credential:

```powershell
git check-ignore secrets/earth-engine.json
git check-ignore .env.payload
git check-ignore .env.ground
```

All three paths must be reported as ignored. `.dockerignore` excludes
`secrets`, `.env*`, runtime data, datasets, outputs, notebooks and test runs.
It deliberately does not exclude the two selected `.pt` model artifacts.

## Local payload validation

Set `VITA_PAYLOAD_BASE_IMAGE` and the project ID in `.env.payload`. The Docker
client can be kept repository-local too:

```powershell
New-Item -ItemType Directory -Force runtime\docker-config | Out-Null
$env:DOCKER_CONFIG = (Resolve-Path runtime\docker-config).Path

docker compose `
  --env-file .env.payload `
  -f compose.payload.local.yaml `
  config

docker compose `
  --env-file .env.payload `
  -f compose.payload.local.yaml `
  build
```

The payload build does not require or read the Earth Engine key. Verify that
the image contains no mounted key:

```powershell
docker run `
  --rm `
  --entrypoint python `
  vita-payload:0.1.0 `
  -c "from pathlib import Path; assert not Path('/run/secrets/earth_engine_credentials').exists(); print('Secret absent from image: PASS')"

docker history --no-trunc vita-payload:0.1.0
```

The entrypoint intentionally fails before startup when `EE_PROJECT_ID` is
missing, the secret is absent/not a readable regular file, CUDA is required but
unavailable, concurrency is not one, a runtime directory is unwritable, or a
model checksum differs. It prints only Python/PyTorch/CUDA/GPU/application
versions and the two public model hashes.

With the key mounted, run the existing safe smoke test and service:

```powershell
docker compose `
  --env-file .env.payload `
  -f compose.payload.local.yaml `
  run --rm payload `
  python -m prithvi_payload.ee_smoke

docker compose `
  --env-file .env.payload `
  -f compose.payload.local.yaml `
  up -d

Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8081/health"
```

The health result must report `status=ok`, `ready=true`, Earth Engine
initialized, CUDA available when required, both models loaded and
`models_warmed=true`.

## Local ground validation

```powershell
docker compose `
  --env-file .env.ground `
  -f compose.ground.yaml `
  config

docker compose `
  --env-file .env.ground `
  -f compose.ground.yaml `
  build

docker compose `
  --env-file .env.ground `
  -f compose.ground.yaml `
  up -d

Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/api/v1/health"
```

The dashboard remains at `http://127.0.0.1:8000/` and OpenAPI at
`http://127.0.0.1:8000/docs`.

## Full local split

`compose.full-local.yaml` starts separate payload and ground services. Only the
payload receives the Earth Engine secret and model/GPU configuration. Only the
ground service receives `runtime/ground`. Neither service can read the other's
persistent data.

```powershell
docker compose `
  --env-file .env.payload `
  -f compose.full-local.yaml `
  config

docker compose `
  --env-file .env.payload `
  -f compose.full-local.yaml `
  up -d --build
```

The host mission CLI remains the ground operator. It preserves its canonical
local archive at `--ground-store` and also submits the same already-verified
three-file bundle to the canonical ground API derived from `--dashboard-url`.
The ground service independently validates and installs the bundle into
`runtime\ground`; therefore the confirmed mission command remains unchanged
and the containerized dashboard sees the result without access to payload
storage or scientific intermediates.

## Image versioning and rollback

Use immutable tags such as `vita-payload:0.1.0` and `vita-ground:0.1.0`. Supply
`VITA_GIT_COMMIT` and `VITA_BUILD_DATE` at build time if desired; labels never
contain credentials or local paths. To roll back, change the Compose image tag
to a previously retained immutable version and run:

```text
docker compose up -d
```

No rebuild is required.

## Jetson: inspect before selecting an image

The following commands are for a human operator on the Jetson. They are not
run by this repository or by any deployment script:

```bash
uname -m
id -u
id -g
cat /etc/os-release
cat /etc/nv_tegra_release 2>/dev/null
dpkg-query --show nvidia-jetpack 2>/dev/null
docker version
docker compose version
docker info
```

Select `VITA_PAYLOAD_BASE_IMAGE` only after confirming all of:

1. architecture is `aarch64`;
2. the base explicitly supports the installed JetPack/L4T release;
3. its PyTorch and torchvision builds use the matching Jetson CUDA ABI;
4. Python satisfies the payload package range;
5. `runtime: nvidia` is reported as supported by the installed container runtime.

Set `VITA_CONTAINER_UID` and `VITA_CONTAINER_GID` in `.env.payload` to the
reported IDs when they differ from `1000`.

Do not use the generic x86 GPU reservation from
`compose.payload.local.yaml` on Jetson. The Jetson Compose uses the target's
NVIDIA runtime and is neither privileged nor host-networked.

Copy the complete Git checkout or a complete archive so hidden files such as
`.dockerignore` and `.env.payload.example` are retained. Do not use
`scp -r Project/*`, which omits hidden files. In the checkout, manually create:

```text
project/secrets/earth-engine.json
project/runtime/payload/jobs/
project/runtime/payload/outbox/
project/runtime/payload/logs/
project/.env.payload
```

After human review and manual secret placement, the operator commands are:

```bash
docker compose \
  --env-file .env.payload \
  -f compose.payload.jetson.yaml \
  build

docker compose \
  --env-file .env.payload \
  -f compose.payload.jetson.yaml \
  up -d

curl --fail http://127.0.0.1:8081/health
docker compose \
  --env-file .env.payload \
  -f compose.payload.jetson.yaml \
  logs --tail=200 payload
tegrastats
```

No command requires `sudo`. If the current user cannot access Docker, stop and
ask the system owner to provide access; do not change users, groups or daemon
settings.

## Ground-to-payload tunnel

The payload remains bound to Jetson loopback. From the ground computer, a human
operator may open:

```powershell
ssh `
  -N `
  -L 18081:127.0.0.1:8081 `
  vita-payload
```

Then change only the mission URL to
`--payload-url "http://127.0.0.1:18081"`. SSH is not implemented in or invoked
by the payload service. `integration/scripts/open_payload_tunnel.ps1` contains
the same password-free helper and is never run automatically.

## Benchmark contract

`python -m prithvi_payload.benchmark <job-directory>` reads a completed payload
job's internal timing record. Warm scientific timing excludes Earth Engine
network acquisition, SSH, container/Python startup, model load/warm-up, ground
ingestion and dashboard rendering. GPU stage timing synchronizes CUDA around
model calls. Benchmark output is internal and never adds a fourth downlink
artifact.
