"""Internal acquisition records; never routine downlink assets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CandidateMetadata:
    system_index: str
    acquired_at: str
    acquired_at_millis: int
    metadata_cloud_percent: float | None
    product_id: str | None
    source_metadata: dict[str, Any]
    candidate_rank: int = 0

    def safe_record(self) -> dict[str, Any]:
        return {
            "provider_scene_id": self.system_index,
            "product_id": self.product_id,
            "acquired_at": self.acquired_at,
            "metadata_cloud_percent": self.metadata_cloud_percent,
            "candidate_rank": self.candidate_rank,
            "source_metadata": self.source_metadata,
        }


@dataclass(frozen=True)
class AcquiredScene:
    local_tiff_path: Path
    provider: str
    collection: str
    provider_scene_id: str
    product_id: str | None
    acquired_at: str
    metadata_cloud_percent: float | None
    requested_bbox_wgs84: tuple[float, float, float, float]
    output_crs: str
    output_transform: tuple[float, float, float, float, float, float]
    width: int
    height: int
    band_names: tuple[str, ...]
    reflectance_scale: float
    source_metadata: dict[str, Any]
    sha256: str
    byte_size: int
    candidate_rank: int
    timing: dict[str, float] = field(default_factory=dict)

    def safe_provenance(
        self,
        *,
        selection_policy: str,
        target_cloud_range: dict[str, float] | None,
        candidate_attempt_count: int,
    ) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "collection": self.collection,
            "provider_scene_id": self.provider_scene_id,
            "product_id": self.product_id,
            "acquired_at": self.acquired_at,
            "requested_bbox_wgs84": list(self.requested_bbox_wgs84),
            "source_crs": self.output_crs,
            "source_transform": list(self.output_transform),
            "source_scale": self.reflectance_scale,
            "source_sha256": self.sha256,
            "source_bytes": self.byte_size,
            "selection_policy": selection_policy,
            "target_cloud_range": target_cloud_range,
            "earth_engine_metadata_cloud_percentage": self.metadata_cloud_percent,
            "candidate_rank": self.candidate_rank,
            "candidate_attempt_count": candidate_attempt_count,
            "resampling_policy": "earth_engine_default_nearest",
        }
