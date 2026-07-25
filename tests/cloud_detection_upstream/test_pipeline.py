import numpy as np
import rasterio
from cloud_detection.backend import BackendPrediction, TestBackend
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


class FourClassBackend:
    """Deterministic backend that exercises every semantic class."""

    def predict(self, tile: np.ndarray) -> BackendPrediction:
        _, height, width = tile.shape
        semantic = np.zeros((height, width), dtype=np.uint8)
        semantic[: height // 2, width // 2 :] = 1
        semantic[height // 2 :, : width // 2] = 2
        semantic[height // 2 :, width // 2 :] = 3
        scores = np.eye(4, dtype=np.float32)[semantic].transpose(2, 0, 1)
        return BackendPrediction(scores=scores, score_kind="deterministic_one_hot")


def test_all_four_cloud_classes_and_mask_rules(config_path):
    pipeline = CloudDetectionPipeline.from_yaml(
        config_path,
        backend=FourClassBackend(),
    )
    image = np.ones((4, 64, 64), dtype=np.float32)

    result = pipeline.predict_array(image)

    assert np.unique(result.semantic_mask).tolist() == [0, 1, 2, 3]
    assert result.thick_cloud_percentage == 25.0
    assert result.thin_cloud_percentage == 25.0
    assert result.shadow_percentage == 25.0
    assert result.cloud_percentage == 50.0
    assert result.unusable_percentage == 75.0
    assert result.usable_percentage == 25.0
    assert result.decision == "PROCESS_CLEAR_AREAS"
