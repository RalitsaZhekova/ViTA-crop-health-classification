# The deployment owner must select the target-qualified x86 CUDA or Jetson/L4T
# image. There is deliberately no architecture guess or silent CPU base.
ARG PAYLOAD_BASE_IMAGE
FROM ${PAYLOAD_BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VITA_JOB_ROOT=/data/jobs \
    PRITHVI_MODEL_DIR=/opt/vita/payload/models \
    CLOUDSEN12_MODEL_DIR=/opt/vita/payload/models/cloudsen12

WORKDIR /opt/vita
COPY shared /opt/vita/shared
COPY payload /opt/vita/payload
COPY deployment/payload/entrypoint.sh /opt/vita/entrypoint.sh

RUN python -m pip install --no-cache-dir /opt/vita/shared /opt/vita/payload \
    && chmod 0555 /opt/vita/entrypoint.sh \
    && mkdir -p /data/jobs \
    && chown -R 10001:10001 /data/jobs /opt/vita

USER 10001:10001
EXPOSE 8081
ENTRYPOINT ["/opt/vita/entrypoint.sh"]
