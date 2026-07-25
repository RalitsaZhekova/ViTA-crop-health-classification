from cloud_detection.config import load_config


def test_valid_config(config_path):
    config = load_config(config_path)
    assert config["input"]["band_names"] == ["B08", "B04", "B03", "B02"]
