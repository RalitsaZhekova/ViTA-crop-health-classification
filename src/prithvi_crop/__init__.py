"""Utilities for the Prithvi four-band crop-segmentation starter."""

from __future__ import annotations

import os
from pathlib import Path


def _set_workspace_cache_defaults() -> None:
    """Keep large/generated caches in the ignored outputs tree by default."""
    cache_root = Path("outputs/cache").resolve()
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))


_set_workspace_cache_defaults()

__version__ = "0.1.0"
