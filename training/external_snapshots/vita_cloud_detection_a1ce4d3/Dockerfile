FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[model]"

COPY configs ./configs
COPY scripts ./scripts
COPY models ./models

RUN test -f models/pretrained/dtacs4bands.pt || \
    (echo "ERROR: download the verified checkpoint before docker build" && exit 1)

ENTRYPOINT ["python", "scripts/run_inference.py", "--config", "configs/cloud_detector.yaml"]
