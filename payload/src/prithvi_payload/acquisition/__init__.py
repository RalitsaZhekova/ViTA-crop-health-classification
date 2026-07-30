"""Payload-side remote scene acquisition boundaries."""

from prithvi_payload.acquisition.base import AcquisitionProvider
from prithvi_payload.acquisition.errors import AcquisitionError
from prithvi_payload.acquisition.models import AcquiredScene, CandidateMetadata

__all__ = [
    "AcquiredScene",
    "AcquisitionError",
    "AcquisitionProvider",
    "CandidateMetadata",
]
