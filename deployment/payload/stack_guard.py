"""Protect the base image's GPU stack while allowing one scoped x86 dependency."""

from __future__ import annotations

import argparse
import json
import re
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

TARGET_PLATFORMS = ("local-x86", "jetson")
STATIC_PROTECTED_DISTRIBUTIONS = (
    "torch",
    "torchvision",
    "torchaudio",
    "triton",
    "cupy",
)
PROTECTED_DISTRIBUTION_PREFIXES = (
    "cupy-cuda",
    "nvidia-cuda-",
    "nvidia-cudnn-",
    "nvidia-cublas-",
)
OPENCV_DISTRIBUTIONS = (
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
)
OBSERVED_DISTRIBUTIONS = ("numpy",)


def _canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _installed_versions() -> dict[str, str]:
    installed: dict[str, str] = {}
    for distribution in distributions():
        raw_name = distribution.metadata.get("Name")
        if raw_name:
            installed[_canonical_name(raw_name)] = distribution.version
    return installed


def _is_protected_distribution(name: str) -> bool:
    return name in STATIC_PROTECTED_DISTRIBUTIONS or name.startswith(
        PROTECTED_DISTRIBUTION_PREFIXES
    )


def snapshot() -> dict[str, Any]:
    installed = _installed_versions()
    selected_names = {
        *STATIC_PROTECTED_DISTRIBUTIONS,
        *OPENCV_DISTRIBUTIONS,
        *OBSERVED_DISTRIBUTIONS,
        *(name for name in installed if _is_protected_distribution(name)),
    }
    packages = {name: installed.get(name) for name in sorted(selected_names)}
    if packages["torch"] is None or packages["torchvision"] is None:
        raise RuntimeError(
            "The selected payload base image must provide both torch and torchvision"
        )

    import torch

    return {
        "schema_version": "2.0",
        "packages": packages,
        "runtime": {"torch_cuda_build": torch.version.cuda},
    }


def _package_versions(record: dict[str, Any]) -> dict[str, str | None]:
    packages = record.get("packages")
    if not isinstance(packages, dict):
        raise RuntimeError("Payload stack snapshot is invalid")
    return packages


def _opencv_wheels(packages: dict[str, str | None]) -> dict[str, str]:
    return {
        name: value
        for name in OPENCV_DISTRIBUTIONS
        if isinstance((value := packages.get(name)), str)
    }


def _validate_opencv(
    before: dict[str, str | None],
    after: dict[str, str | None],
    *,
    target_platform: str,
) -> tuple[dict[str, str], dict[str, str], bool]:
    before_wheels = _opencv_wheels(before)
    after_wheels = _opencv_wheels(after)
    if len(before_wheels) > 1 or len(after_wheels) > 1:
        raise RuntimeError(
            "Conflicting OpenCV Python wheel families are installed: "
            + ", ".join(sorted({*before_wheels, *after_wheels}))
        )

    if target_platform == "jetson":
        if before_wheels != after_wheels:
            raise RuntimeError(
                "Jetson dependency installation may not add, remove or replace an OpenCV wheel"
            )
        return before_wheels, after_wheels, False

    if before_wheels:
        if before_wheels != after_wheels:
            raise RuntimeError("Dependency installation replaced an existing OpenCV wheel")
        return before_wheels, after_wheels, False

    if not after_wheels:
        return before_wheels, after_wheels, False
    if set(after_wheels) != {"opencv-python-headless"}:
        raise RuntimeError(
            "Local x86 may add only opencv-python-headless when the base has no OpenCV wheel"
        )
    opencv_version = after_wheels["opencv-python-headless"]
    if re.fullmatch(r"4(?:\.\d+){2,3}", opencv_version) is None:
        raise RuntimeError("Local x86 requires a stable OpenCV 4.x headless wheel")
    return before_wheels, after_wheels, True


def compare_snapshots(
    before_record: dict[str, Any],
    after_record: dict[str, Any],
    *,
    target_platform: str,
) -> dict[str, Any]:
    if target_platform not in TARGET_PLATFORMS:
        raise RuntimeError("Unknown payload target platform")
    before = _package_versions(before_record)
    after = _package_versions(after_record)
    protected_names = sorted(
        name for name in {*before, *after} if _is_protected_distribution(name)
    )
    changed = [name for name in protected_names if before.get(name) != after.get(name)]
    if before_record.get("runtime") != after_record.get("runtime"):
        changed.append("torch-cuda-build")
    if changed:
        raise RuntimeError(
            "Payload dependency installation changed protected GPU packages: "
            + ", ".join(changed)
        )

    before_opencv, after_opencv, allowed_addition = _validate_opencv(
        before,
        after,
        target_platform=target_platform,
    )
    return {
        "status": "ok",
        "target_platform": target_platform,
        "protected_gpu_stack_unchanged": True,
        "torch": after.get("torch"),
        "torchvision": after.get("torchvision"),
        "torch_cuda_build": after_record.get("runtime", {}).get("torch_cuda_build"),
        "numpy_before": before.get("numpy"),
        "numpy_after": after.get("numpy"),
        "opencv_before": before_opencv,
        "opencv_after": after_opencv,
        "allowed_local_x86_opencv_addition": allowed_addition,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("snapshot", "compare"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--target-platform", choices=TARGET_PLATFORMS, required=True)
    args = parser.parse_args()
    current = snapshot()
    if args.action == "snapshot":
        args.path.write_text(json.dumps(current, sort_keys=True), encoding="utf-8")
        packages = _package_versions(current)
        print(
            json.dumps(
                {
                    "status": "snapshot",
                    "target_platform": args.target_platform,
                    "torch": packages.get("torch"),
                    "torchvision": packages.get("torchvision"),
                    "torch_cuda_build": current["runtime"]["torch_cuda_build"],
                    "numpy": packages.get("numpy"),
                    "opencv_wheels": _opencv_wheels(packages),
                },
                sort_keys=True,
            )
        )
        return
    before = json.loads(args.path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            compare_snapshots(before, current, target_platform=args.target_platform),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
