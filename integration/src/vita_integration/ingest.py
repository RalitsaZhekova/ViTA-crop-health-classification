"""Validate and ingest one three-file payload downlink on the ground."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prithvi_ground.catalog import SceneCatalog


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        prog="vita-ingest",
        description="Checksum-validate and ingest one ViTA downlink bundle.",
    )
    command.add_argument("bundle", type=Path)
    command.add_argument("--store", type=Path, default=Path("runtime/ground"))
    return command


def main() -> None:
    args = parser().parse_args()
    try:
        scene, created = SceneCatalog(args.store).ingest(args.bundle)
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "INGEST_FAILED", "error": str(error)}))
        raise SystemExit(2) from None
    print(
        json.dumps(
            {
                "status": "INGESTED" if created else "ALREADY_PRESENT",
                "created": created,
                "scene": scene,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
