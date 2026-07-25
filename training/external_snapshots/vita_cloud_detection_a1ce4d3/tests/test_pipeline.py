import rasterio

from cloud_detection.backend import TestBackend
from cloud_detection.pipeline import CloudDetectionPipeline


def test_end_to_end(config_path, synthetic_tif, tmp_path):
    pipeline = CloudDetectionPipeline.from_yaml(config_path, backend=TestBackend())
    result = pipeline.predict_file(synthetic_tif, tmp_path / "outputs")

    assert result.decision in {"PROCESS", "PROCESS_CLEAR_AREAS", "REJECT"}
    assert 0 <= result.cloud_percentage <= 100
    assert 0 <= result.unusable_percentage <= 100
    assert abs(result.usable_percentage + result.unusable_percentage - 100) < 1e-6
    assert result.score_kind == "synthetic_probability"

    with rasterio.open(result.output_files["semantic_mask"]) as dataset:
        assert dataset.count == 1
        assert dataset.crs.to_string() == "EPSG:32632"
