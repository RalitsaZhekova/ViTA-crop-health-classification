"""Safe, non-interactive Earth Engine credential and collection smoke check."""

from __future__ import annotations

import os

from prithvi_payload.acquisition.earth_engine import (
    EARTH_ENGINE_COLLECTION,
    EARTH_ENGINE_PROJECT_ID,
    initialize_earth_engine,
)


def main() -> None:
    project_id = os.environ.get("EE_PROJECT_ID", EARTH_ENGINE_PROJECT_ID)
    try:
        initialize_earth_engine(project_id)
        import ee

        collection = ee.ImageCollection(EARTH_ENGINE_COLLECTION)
        collection.limit(1).size().getInfo()
    except Exception:
        print("Earth Engine smoke check: FAILED (runtime credentials or access unavailable)")
        raise SystemExit(2) from None
    print(f"Earth Engine smoke check: OK project={project_id} collection={EARTH_ENGINE_COLLECTION}")


if __name__ == "__main__":
    main()
