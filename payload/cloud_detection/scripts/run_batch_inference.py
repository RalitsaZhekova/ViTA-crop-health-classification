from __future__ import annotations

import argparse
import json
from pathlib import Path

from cloud_detection.pipeline import CloudDetectionPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument(
        "--config",
        default="payload/cloud_detection/configs/cloud_detector.yaml",
    )
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()

    input_files = sorted(Path(args.input_dir).glob("*.tif"))
    if not input_files:
        raise SystemExit(f"No GeoTIFF files found in {args.input_dir}.")

    pipeline = CloudDetectionPipeline.from_yaml(args.config)
    summary: list[dict] = []
    for input_path in input_files:
        result = pipeline.predict_file(input_path, args.output)
        summary.append({"file": str(input_path), **result.metadata()})
        print(
            input_path.name,
            result.decision,
            f"{result.unusable_percentage:.2f}% unusable",
        )

    output_path = Path(args.output) / "metadata" / "batch_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
