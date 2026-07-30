# syntax=docker/dockerfile:1

ARG BASE_IMAGE=python:3.11-slim-bookworm
FROM ${BASE_IMAGE}

ARG APPLICATION_VERSION=0.1.0
ARG VCS_REF=unknown
ARG BUILD_DATE=unknown
ARG APPLICATION_UID=1000
ARG APPLICATION_GID=1000

LABEL org.opencontainers.image.title="ViTA ground service" \
      org.opencontainers.image.version="${APPLICATION_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    VITA_GROUND_STORE=/data/ground

WORKDIR /workspace/vita

# The ground image receives only shared contracts and the ground API/catalog.
COPY shared/pyproject.toml shared/pyproject.toml
COPY shared/src shared/src
COPY shared/schemas shared/schemas
COPY ground/pyproject.toml ground/pyproject.toml
COPY ground/src ground/src
COPY deployment/ground/entrypoint.sh /usr/local/bin/vita-ground-entrypoint

RUN python -m pip install -e /workspace/vita/shared -e /workspace/vita/ground \
    && chmod 0555 /usr/local/bin/vita-ground-entrypoint \
    && mkdir -p /data/ground \
    && chown -R ${APPLICATION_UID}:${APPLICATION_GID} /data/ground /workspace/vita

USER ${APPLICATION_UID}:${APPLICATION_GID}
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/vita-ground-entrypoint"]
CMD ["python", "-m", "prithvi_ground.api", "--store", "/data/ground", "--host", "0.0.0.0", "--port", "8000"]
