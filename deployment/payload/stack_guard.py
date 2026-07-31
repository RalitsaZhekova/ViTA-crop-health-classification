"""Protect the base image's GPU stack and any existing OpenCV provider."""

from __future__ import annotations

import argparse
import json
import os
import re
from importlib import import_module
from importlib.metadata import Distribution, distributions
from pathlib import Path
from typing import Any

TARGET_PLATFORMS = ("local-x86", "jetson")
PINNED_OPENCV_HEADLESS_VERSION = "4.11.0.86"
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
    "nvidia-nccl-",
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


def _installed_distributions() -> dict[str, Distribution]:
    installed: dict[str, Distribution] = {}
    for distribution in distributions():
        raw_name = distribution.metadata.get("Name")
        if raw_name:
            installed[_canonical_name(raw_name)] = distribution
    return installed


def _is_protected_distribution(name: str) -> bool:
    return name in STATIC_PROTECTED_DISTRIBUTIONS or name.startswith(
        PROTECTED_DISTRIBUTION_PREFIXES
    )


def _normalized_path(value: str | Path) -> str:
    return os.path.normcase(str(Path(value).resolve(strict=False)))


def _distribution_owns_path(distribution: Distribution, path: str) -> bool:
    target = _normalized_path(path)
    for relative_path in distribution.files or ():
        candidate = distribution.locate_file(relative_path)
        if _normalized_path(candidate) == target:
            return True
    return False


def _inspect_cv2(installed: dict[str, Distribution]) -> dict[str, Any]:
    try:
        cv2 = import_module("cv2")
    except ModuleNotFoundError as exc:
        if exc.name == "cv2":
            return {
                "cv2_importable": False,
                "cv2_version": None,
                "cv2_path": None,
                "cv2_wheel_owners": [],
            }
        raise RuntimeError(
            "An existing cv2 installation was found but could not import a dependency"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "An existing cv2 installation was found but could not be imported safely"
        ) from exc

    raw_path = getattr(cv2, "__file__", None)
    cv2_path = _normalized_path(raw_path) if isinstance(raw_path, str) else None
    raw_version = getattr(cv2, "__version__", None)
    cv2_version = raw_version if isinstance(raw_version, str) else None
    wheel_owners = [
        name
        for name in OPENCV_DISTRIBUTIONS
        if cv2_path is not None
        and (distribution := installed.get(name)) is not None
        and _distribution_owns_path(distribution, cv2_path)
    ]
    return {
        "cv2_importable": True,
        "cv2_version": cv2_version,
        "cv2_path": cv2_path,
        "cv2_wheel_owners": wheel_owners,
    }


def snapshot() -> dict[str, Any]:
    installed = _installed_distributions()
    installed_versions = {
        name: distribution.version for name, distribution in installed.items()
    }
    selected_names = {
        *STATIC_PROTECTED_DISTRIBUTIONS,
        *OPENCV_DISTRIBUTIONS,
        *OBSERVED_DISTRIBUTIONS,
        *(name for name in installed if _is_protected_distribution(name)),
    }
    packages = {name: installed_versions.get(name) for name in sorted(selected_names)}
    if packages["torch"] is None or packages["torchvision"] is None:
        raise RuntimeError(
            "The selected payload base image must provide both torch and torchvision"
        )

    import torch

    opencv_wheels = _opencv_wheels(packages)
    return {
        "schema_version": "3.0",
        "packages": packages,
        "runtime": {"torch_cuda_build": torch.version.cuda},
        "opencv_wheels": opencv_wheels,
        **_inspect_cv2(installed),
    }


def _package_versions(record: dict[str, Any]) -> dict[str, str | None]:
    packages = record.get("packages")
    if not isinstance(packages, dict) or not all(
        isinstance(name, str) and (value is None or isinstance(value, str))
        for name, value in packages.items()
    ):
        raise RuntimeError("Payload stack snapshot has invalid package metadata")
    return packages


def _opencv_wheels(packages: dict[str, str | None]) -> dict[str, str]:
    return {
        name: value
        for name in OPENCV_DISTRIBUTIONS
        if isinstance((value := packages.get(name)), str)
    }


def _opencv_state(record: dict[str, Any]) -> dict[str, Any]:
    packages = _package_versions(record)
    expected_wheels = _opencv_wheels(packages)
    opencv_wheels = record.get("opencv_wheels")
    if opencv_wheels != expected_wheels:
        raise RuntimeError("Payload stack snapshot has inconsistent OpenCV wheel metadata")

    cv2_importable = record.get("cv2_importable")
    cv2_version = record.get("cv2_version")
    cv2_path = record.get("cv2_path")
    cv2_wheel_owners = record.get("cv2_wheel_owners")
    if not isinstance(cv2_importable, bool):
        raise RuntimeError("Payload stack snapshot has invalid cv2 import metadata")
    if cv2_version is not None and not isinstance(cv2_version, str):
        raise RuntimeError("Payload stack snapshot has invalid cv2 version metadata")
    if cv2_path is not None and not isinstance(cv2_path, str):
        raise RuntimeError("Payload stack snapshot has invalid cv2 path metadata")
    if not isinstance(cv2_wheel_owners, list) or not all(
        isinstance(owner, str) and owner in OPENCV_DISTRIBUTIONS
        for owner in cv2_wheel_owners
    ):
        raise RuntimeError("Payload stack snapshot has invalid cv2 ownership metadata")
    if any(owner not in opencv_wheels for owner in cv2_wheel_owners):
        raise RuntimeError("Payload stack snapshot has inconsistent cv2 ownership metadata")
    if not cv2_importable and (
        cv2_version is not None or cv2_path is not None or cv2_wheel_owners
    ):
        raise RuntimeError("Non-importable cv2 may not have provider metadata")

    return {
        "opencv_wheels": opencv_wheels,
        "cv2_importable": cv2_importable,
        "cv2_version": cv2_version,
        "cv2_path": cv2_path,
        "cv2_wheel_owners": cv2_wheel_owners,
    }


def _cv2_signature(state: dict[str, Any]) -> tuple[Any, ...]:
    return (
        state["cv2_importable"],
        state["cv2_version"],
        state["cv2_path"],
        tuple(state["cv2_wheel_owners"]),
    )


def _validate_new_headless_wheel(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    target_platform: str,
) -> None:
    if before["cv2_importable"]:
        raise RuntimeError(
            f"{target_platform} may not add an OpenCV wheel when cv2 was already "
            "importable from a system, NVIDIA or other provider"
        )
    if set(after["opencv_wheels"]) != {"opencv-python-headless"}:
        raise RuntimeError(
            f"{target_platform} may add only opencv-python-headless when the base "
            "has no OpenCV provider"
        )
    opencv_version = after["opencv_wheels"]["opencv-python-headless"]
    if target_platform == "jetson":
        if opencv_version != PINNED_OPENCV_HEADLESS_VERSION:
            raise RuntimeError(
                "Jetson requires opencv-python-headless=="
                f"{PINNED_OPENCV_HEADLESS_VERSION}"
            )
    elif re.fullmatch(r"4(?:\.\d+){2,3}", opencv_version) is None:
        raise RuntimeError("Local x86 requires a stable OpenCV 4.x headless wheel")
    if not after["cv2_importable"]:
        raise RuntimeError("The installed OpenCV wheel must provide an importable cv2")
    if after["cv2_wheel_owners"] != ["opencv-python-headless"]:
        raise RuntimeError(
            "The installed cv2 path must be owned only by opencv-python-headless; "
            "a system or NVIDIA cv2 may not be shadowed"
        )


def _validate_opencv(
    before_record: dict[str, Any],
    after_record: dict[str, Any],
    *,
    target_platform: str,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    before = _opencv_state(before_record)
    after = _opencv_state(after_record)
    before_wheels = before["opencv_wheels"]
    after_wheels = after["opencv_wheels"]
    if len(before_wheels) > 1 or len(after_wheels) > 1:
        raise RuntimeError(
            "Conflicting OpenCV Python wheel families are installed: "
            + ", ".join(sorted({*before_wheels, *after_wheels}))
        )

    if before_wheels:
        if before_wheels != after_wheels:
            action = "removed" if not after_wheels else "replaced"
            raise RuntimeError(f"Dependency installation {action} an existing OpenCV wheel")
        if _cv2_signature(before) != _cv2_signature(after):
            raise RuntimeError("Dependency installation changed the existing cv2 provider")
        return before, after, False

    if not after_wheels:
        if _cv2_signature(before) != _cv2_signature(after):
            raise RuntimeError("Dependency installation changed the existing cv2 provider")
        return before, after, False

    _validate_new_headless_wheel(before, after, target_platform=target_platform)
    return before, after, True


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
        before_record,
        after_record,
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
        "opencv_before": before_opencv["opencv_wheels"],
        "opencv_after": after_opencv["opencv_wheels"],
        "cv2_before": {
            key: before_opencv[key]
            for key in ("cv2_importable", "cv2_version", "cv2_path")
        },
        "cv2_after": {
            key: after_opencv[key]
            for key in ("cv2_importable", "cv2_version", "cv2_path")
        },
        "allowed_local_x86_opencv_addition": (
            target_platform == "local-x86" and allowed_addition
        ),
        "allowed_jetson_opencv_addition": (
            target_platform == "jetson" and allowed_addition
        ),
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
                    "opencv_wheels": current["opencv_wheels"],
                    "cv2_importable": current["cv2_importable"],
                    "cv2_version": current["cv2_version"],
                    "cv2_path": current["cv2_path"],
                    "cv2_wheel_owners": current["cv2_wheel_owners"],
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
