import pytest
import yaml
from cloud_detection.config import ConfigurationError, load_config


def test_valid_config(config_path):
    config = load_config(config_path)
    assert config["input"]["band_names"] == ["B08", "B04", "B03", "B02"]


def test_rejects_changed_cloud_class_mapping(config_path):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["classes"]["thin_cloud"] = 3
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Class mapping"):
        load_config(config_path)
