import json
import sys
from pathlib import Path

import numpy as np
import yaml

SHARED_SOURCE = Path("shared/src").resolve()
GROUND_SOURCE = Path("ground/src").resolve()
PAYLOAD_SOURCE = Path("payload/src").resolve()
for source in (SHARED_SOURCE, GROUND_SOURCE, PAYLOAD_SOURCE):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from prithvi_ground.health import build_analysis_mask  # noqa: E402
from prithvi_payload.inference import _model_state  # noqa: E402
from prithvi_shared import (  # noqa: E402
    CLASS_NAMES,
    CROP_CLASSIFICATION_THRESHOLD,
    HEALTH_ANALYSIS_CROP_THRESHOLD,
    MODEL_BANDS,
    SELECTED_CHECKPOINT_SHA256,
)


def test_shared_contract_matches_selected_model() -> None:
    manifest = yaml.safe_load(
        Path("payload/models/selected_model.yaml").read_text(encoding="utf-8")
    )
    architecture = yaml.safe_load(
        Path("payload/models/architecture.yaml").read_text(encoding="utf-8")
    )

    assert manifest["sha256"] == SELECTED_CHECKPOINT_SHA256
    assert architecture["model_args"]["backbone_bands"] == list(MODEL_BANDS)
    assert architecture["model_args"]["num_classes"] == len(CLASS_NAMES)
    assert CROP_CLASSIFICATION_THRESHOLD == 0.615
    assert HEALTH_ANALYSIS_CROP_THRESHOLD == 0.76


def test_ground_health_uses_shared_conservative_threshold() -> None:
    mask = build_analysis_mask(
        np.asarray([[2, 3]]),
        np.zeros((1, 2), dtype=bool),
        crop_probability=np.asarray([[0.75, 0.80]], dtype=np.float32),
    )

    assert mask.tolist() == [[False, True]]


def test_payload_checkpoint_filter_excludes_training_state() -> None:
    import torch

    checkpoint = {
        "state_dict": {
            "model.encoder.weight": torch.ones(1),
            "train_metrics.value": torch.zeros(1),
        }
    }

    assert torch.equal(_model_state(checkpoint)["encoder.weight"], torch.ones(1))


def test_payload_accepts_direct_weights_only_state() -> None:
    import torch

    state = {"encoder.weight": torch.ones(1)}

    assert _model_state(state) is state


def test_shared_json_schemas_are_parseable() -> None:
    for path in Path("shared/schemas").glob("*.json"):
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert schema["$schema"].endswith("2020-12/schema")
        assert schema["type"] == "object"


def test_payload_contains_no_training_or_dataset_artifacts() -> None:
    forbidden = [
        path
        for path in Path("payload").rglob("*")
        if path.is_file()
        and (
            path.suffix.lower() in {".npy", ".tif", ".tgz", ".zip"}
            or path.name in {"train.py", "events.out.tfevents"}
        )
    ]

    assert forbidden == []


def test_training_snapshot_preserves_original_source_bytes() -> None:
    manifest = yaml.safe_load(
        Path("training/EXTRACTION_MANIFEST.yaml").read_text(encoding="utf-8")
    )
    for name in manifest["source_modules"]:
        assert (Path("src/prithvi_crop") / name).read_bytes() == (
            Path("training/source_snapshot/prithvi_crop") / name
        ).read_bytes()
