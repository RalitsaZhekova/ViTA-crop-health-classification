PYTHON ?= python
CONFIG ?= configs/prithvi_4band_head_only.yaml
CKPT ?=
BATCH_SIZE ?= 8
NUM_WORKERS ?= 4

.PHONY: install download download-validation validate check-config preflight smoke train test lint unit verify-components

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

download:
	$(PYTHON) -m prithvi_crop.download_data --root data/multi_temporal_crop --delete-archives

download-validation:
	$(PYTHON) -m prithvi_crop.download_data --root data/multi_temporal_crop --validation-only --delete-archives

validate:
	$(PYTHON) -m prithvi_crop.validate_data --root data/multi_temporal_crop --device cuda

check-config:
	$(PYTHON) -m prithvi_crop.check_config --config $(CONFIG)

preflight:
	$(PYTHON) -m prithvi_crop.preflight --config $(CONFIG)

smoke: check-config
	$(PYTHON) -m prithvi_crop.smoke --config $(CONFIG)

train: check-config
	$(PYTHON) -m prithvi_crop.train --config $(CONFIG) --batch-size $(BATCH_SIZE) --num-workers $(NUM_WORKERS)

test: check-config
	$(PYTHON) -m prithvi_crop.evaluate --config $(CONFIG) --checkpoint "$(CKPT)" --num-workers $(NUM_WORKERS)

lint:
	$(PYTHON) -m ruff check src tests scripts shared/src payload/src ground/src

unit:
	$(PYTHON) -m pytest

verify-components:
	$(PYTHON) scripts/verify_component_extraction.py --workspace .
