"""Small set of contracts shared by the two MVPs."""

from prithvi_shared.bands import (
    INPUT_HEIGHT,
    INPUT_WIDTH,
    MODEL_BANDS,
    NORMALIZATION_MEANS,
    NORMALIZATION_STDS,
    TIME_STEPS,
)
from prithvi_shared.calibration import (
    CROP_CLASSIFICATION_THRESHOLD,
    HEALTH_ANALYSIS_CROP_THRESHOLD,
    SELECTED_CHECKPOINT_NAME,
    SELECTED_CHECKPOINT_SHA256,
)
from prithvi_shared.condition import (
    ConditionAssessment,
    ConditionConfig,
    ConditionScoreLayers,
    SpatialConditionLayers,
    build_condition_assessment,
    calculate_condition_score_layers,
    calculate_spatial_condition_layers,
)
from prithvi_shared.health import (
    HealthLayers,
    build_analysis_mask,
    calculate_health_layers,
)

__all__ = [
    "ConditionAssessment",
    "ConditionConfig",
    "ConditionScoreLayers",
    "CROP_CLASSIFICATION_THRESHOLD",
    "HEALTH_ANALYSIS_CROP_THRESHOLD",
    "HealthLayers",
    "INPUT_HEIGHT",
    "INPUT_WIDTH",
    "MODEL_BANDS",
    "NORMALIZATION_MEANS",
    "NORMALIZATION_STDS",
    "SELECTED_CHECKPOINT_NAME",
    "SELECTED_CHECKPOINT_SHA256",
    "SpatialConditionLayers",
    "TIME_STEPS",
    "build_analysis_mask",
    "build_condition_assessment",
    "calculate_condition_score_layers",
    "calculate_health_layers",
    "calculate_spatial_condition_layers",
]
