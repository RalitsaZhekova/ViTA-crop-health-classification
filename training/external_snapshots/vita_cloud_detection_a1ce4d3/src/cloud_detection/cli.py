from __future__ import annotations

import argparse
import json

from .pipeline import CloudDetectionPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run pretrained CloudSEN12 cloud detection on a four-band GeoTIFF."
    )
    parser.add_argument("--input", required=True, help="Input Sentinel-2 L1C GeoTIFF.")
    parser.add_argument(
        "--config",
        default="configs/cloud_detector.yaml",
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
