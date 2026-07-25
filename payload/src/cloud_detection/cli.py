from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import CloudDetectionPipeline

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "cloud_detection"
    / "configs"
    / "cloud_detector.yaml"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run pretrained CloudSEN12 cloud detection on a four-band GeoTIFF."
    )
    parser.add_argument("--input", required=True, help="Input Sentinel-2 L1C GeoTIFF.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Cloud-detector YAML configuration.",
    )
    parser.add_argument("--output", default="outputs", help="Output directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = CloudDetectionPipeline.from_yaml(args.config)
    result = pipeline.predict_file(args.input, args.output)
    print(json.dumps(result.metadata(), indent=2))


if __name__ == "__main__":
    main()
