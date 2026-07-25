from __future__ import annotations

import argparse
import json
from pathlib import Path

from cloud_detection.benchmark import benchmark_file
from cloud_detection.pipeline import CloudDetectionPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--config",
        default="payload/cloud_detection/configs/cloud_detector.yaml",
    )
    parser.add_argument("--output", default="outputs/benchmarks")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    pipeline = CloudDetectionPipeline.from_yaml(args.config)
    result = benchmark_file(pipeline, args.input, args.output, args.runs)
    output_path = Path(args.output) / "benchmark.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
