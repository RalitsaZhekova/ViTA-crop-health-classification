# syntax=docker/dockerfile:1

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG APPLICATION_VERSION=0.1.0
ARG VCS_REF=unknown
ARG BUILD_DATE=unknown
ARG APPLICATION_UID=1000
ARG APPLICATION_GID=1000

LABEL org.opencontainers.image.title="ViTA payload service" \
      org.opencontainers.image.version="${APPLICATION_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      vita.crop-model.sha256="c948977bffdaeb89ecf4f4d069db13c7ed81d4f3403eec9e257c45c235b1484e" \
      vita.cloud-model.sha256="37205adce72fbbb65a3cfa8f47676c84ebf9b1555a27a3838d584072c954b22d"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/earth_engine_credentials \
    VITA_JOBS_DIR=/data/jobs \
    VITA_OUTBOX_DIR=/data/outbox \
    VITA_LOGS_DIR=/data/logs \
    PRITHVI_MODEL_DIR=/workspace/vita/payload/models \
    CLOUDSEN12_MODEL_DIR=/workspace/vita/payload/models/cloudsen12 \
    MPLCONFIGDIR=/data/logs/matplotlib

WORKDIR /workspace/vita

# Copy only the shared contract and payload runtime. Training, ground,
# integration, secrets and generated runtime data never enter this image.
COPY shared/pyproject.toml shared/pyproject.toml
COPY shared/src shared/src
COPY shared/schemas shared/schemas
COPY payload/pyproject.toml payload/pyproject.toml
COPY payload/src payload/src
COPY payload/cloud_detection/configs payload/cloud_detection/configs
COPY payload/models payload/models
COPY deployment/payload/entrypoint.sh /usr/local/bin/vita-payload-entrypoint
COPY deployment/payload/stack_guard.py /tmp/vita-stack-guard.py

# The selected base must already provide its target-qualified PyTorch/CUDA
# stack. Abort if dependency installation changes torch, torchvision, OpenCV or
# CuPy instead of silently replacing the base-image stack.
RUN python /tmp/vita-stack-guard.py snapshot /tmp/vita-stack-before.json \
    && python -m pip install -e /workspace/vita/shared -e /workspace/vita/payload \
    && python /tmp/vita-stack-guard.py compare /tmp/vita-stack-before.json \
    && python -m prithvi_payload.container_preflight --models-only \
    && chmod 0555 /usr/local/bin/vita-payload-entrypoint \
    && mkdir -p /data/jobs /data/outbox /data/logs/matplotlib \
    && chown -R ${APPLICATION_UID}:${APPLICATION_GID} /data /workspace/vita \
    && rm -f /tmp/vita-stack-guard.py /tmp/vita-stack-before.json

USER ${APPLICATION_UID}:${APPLICATION_GID}
EXPOSE 8081
ENTRYPOINT ["/usr/local/bin/vita-payload-entrypoint"]
CMD ["python", "-m", "prithvi_payload.service", "--host", "0.0.0.0", "--port", "8081"]
