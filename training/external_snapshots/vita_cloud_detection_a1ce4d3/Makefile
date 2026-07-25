.PHONY: install test integration weights run

install:
	python -m pip install -e ".[model,dev]"

test:
	pytest -q -m "not integration"

integration:
	pytest -q -m integration

weights:
	python scripts/download_weights.py

run:
	python scripts/run_inference.py --input data/raw/cloud_scene_01.tif --output outputs
