from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

STACK_GUARD_PATH = Path(__file__).parents[2] / "deployment/payload/stack_guard.py"


def _load_stack_guard() -> ModuleType:
    spec = importlib.util.spec_from_file_location("vita_stack_guard", STACK_GUARD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stack_guard = _load_stack_guard()


def _record(**overrides: str | None) -> dict[str, object]:
    packages: dict[str, str | None] = {
        "torch": "2.5.1+cu124",
        "torchvision": "0.20.1+cu124",
        "torchaudio": "2.5.1+cu124",
        "triton": "3.1.0",
        "cupy": None,
        "nvidia-cuda-runtime-cu12": "12.4.127",
        "nvidia-cudnn-cu12": "9.1.0.70",
        "nvidia-cublas-cu12": "12.4.5.8",
        "numpy": "2.2.6",
        "opencv-python": None,
        "opencv-python-headless": None,
        "opencv-contrib-python": None,
        "opencv-contrib-python-headless": None,
    }
    packages.update(overrides)
    return {
        "schema_version": "2.0",
        "packages": packages,
        "runtime": {"torch_cuda_build": "12.4"},
    }


def test_local_x86_allows_first_stable_headless_opencv_wheel() -> None:
    after = _record(**{"opencv-python-headless": "4.11.0.86"})
    report = stack_guard.compare_snapshots(
        _record(), after, target_platform="local-x86"
    )

    assert report["allowed_local_x86_opencv_addition"] is True
    assert report["opencv_before"] == {}
    assert report["opencv_after"] == {"opencv-python-headless": "4.11.0.86"}


def test_local_x86_rejects_replacing_existing_opencv_version() -> None:
    before = _record(**{"opencv-python-headless": "4.10.0.84"})
    after = _record(**{"opencv-python-headless": "4.11.0.86"})

    with pytest.raises(RuntimeError, match="replaced an existing OpenCV"):
        stack_guard.compare_snapshots(before, after, target_platform="local-x86")


def test_local_x86_rejects_conflicting_opencv_wheels() -> None:
    after = _record(
        **{
            "opencv-python": "4.11.0.86",
            "opencv-python-headless": "4.11.0.86",
        }
    )

    with pytest.raises(RuntimeError, match="Conflicting OpenCV"):
        stack_guard.compare_snapshots(_record(), after, target_platform="local-x86")


def test_torch_version_change_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="torch"):
        stack_guard.compare_snapshots(
            _record(), _record(torch="2.6.0+cu124"), target_platform="local-x86"
        )


def test_torchvision_version_change_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="torchvision"):
        stack_guard.compare_snapshots(
            _record(),
            _record(torchvision="0.21.0+cu124"),
            target_platform="local-x86",
        )


def test_nvidia_cuda_package_change_is_rejected() -> None:
    changed = _record(**{"nvidia-cuda-runtime-cu12": "12.6.77"})
    with pytest.raises(RuntimeError, match="nvidia-cuda-runtime-cu12"):
        stack_guard.compare_snapshots(_record(), changed, target_platform="local-x86")


def test_jetson_rejects_new_generic_pypi_opencv() -> None:
    changed = _record(**{"opencv-python-headless": "4.11.0.86"})
    with pytest.raises(RuntimeError, match="Jetson"):
        stack_guard.compare_snapshots(_record(), changed, target_platform="jetson")


def test_constrained_payload_dependency_set_passes_local_x86() -> None:
    before = _record(numpy="2.1.2")
    after = _record(
        numpy="2.2.6",
        **{"opencv-python-headless": "4.11.0.86"},
    )

    report = stack_guard.compare_snapshots(
        before, after, target_platform="local-x86"
    )

    assert report["protected_gpu_stack_unchanged"] is True
    assert report["numpy_after"] == "2.2.6"
    assert report["torch"] == "2.5.1+cu124"
    assert report["torchvision"] == "0.20.1+cu124"


def test_real_protected_stack_drift_still_fails() -> None:
    changed = _record(triton="3.2.0")
    with pytest.raises(RuntimeError, match="triton"):
        stack_guard.compare_snapshots(_record(), changed, target_platform="local-x86")


def test_local_x86_rejects_future_major_opencv() -> None:
    changed = _record(**{"opencv-python-headless": "5.0.0.93"})
    with pytest.raises(RuntimeError, match="stable OpenCV 4.x"):
        stack_guard.compare_snapshots(_record(), changed, target_platform="local-x86")
