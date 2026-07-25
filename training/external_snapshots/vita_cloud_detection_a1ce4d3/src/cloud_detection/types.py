from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class CloudDetectionResult:
    semantic_mask: np.ndarray
    unusable_mask: np.ndarray
    class_scores: np.ndarray
    score_kind: str
    thick_cloud_percentage: float
    thin_cloud_percentage: float
    cloud_percentage: float
    shadow_percentage: float
    invalid_percentage: float
    usable_percentage: float
    unusable_percentage: float
    decision: str
    runtime_seconds: float
    output_files: dict[str, str] | None = None

    def metadata(self) -> dict:
        data = asdict(self)
        data.pop("semantic_mask")
        data.pop("unusable_mask")
        data.pop("class_scores")
        return data
