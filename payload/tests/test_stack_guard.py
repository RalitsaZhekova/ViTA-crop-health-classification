from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

STACK_GUARD_PATH = Path(__file__).parents[2] / "deployment/payload/stack_guard.py"
PINNED_OPENCV_VERSION = "4.11.0.86"
PIP_CV2_PATH = "/usr/local/lib/python3.12/site-packages/cv2/__init__.py"
SYSTEM_CV2_PATH = "/usr/lib/python3/dist-packages/cv2.cpython-312-aarch64-linux-gnu.so"


def _load_stack_guard() -> ModuleType:
    spec = importlib.util.spec_from_file_location("vita_stack_guard", STACK_GUARD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stack_guard = _load_stack_guard()


def _record(
    *,
    cv2_importable: bool = False,
    cv2_version: str | None = None,
    cv2_path: str | None = None,
    cv2_wheel_owners: list[str] | None = None,
    torch_cuda_build: str = "13.2",
    **overrides: str | None,
) -> dict[str, object]:
    packages: dict[str, str | None] = {
        "torch": "2.12.0a0+5aff3928d8.nv26.5.50603568",
        "torchvision": "0.27.0a0+6214cb20.nv26.5.50603568",
        "torchaudio": "2.12.0a0+5aff3928d8.nv26.5.50603568",
        "triton": "3.6.0",
        "cupy": None,
        "nvidia-cuda-runtime-cu13": "13.2.0",
        "nvidia-cudnn-cu13": "9.16.0",
        "nvidia-cublas-cu13": "13.2.0",
        "nvidia-nccl-cu13": "2.28.9",
        "numpy": "2.1.0",
        "opencv-python": None,
        "opencv-python-headless": None,
        "opencv-contrib-python": None,
        "opencv-contrib-python-headless": None,
    }
    packages.update(overrides)
    opencv_wheels = {
        name: version
        for name in stack_guard.OPENCV_DISTRIBUTIONS
        if isinstance((version := packages.get(name)), str)
    }
    if cv2_importable:
        cv2_version = cv2_version or "4.11.0"
        cv2_path = cv2_path or PIP_CV2_PATH
        if cv2_wheel_owners is None:
            cv2_wheel_owners = list(opencv_wheels)
    else:
        cv2_version = None
        cv2_path = None
        cv2_wheel_owners = []
    return {
        "schema_version": "3.0",
        "packages": packages,
        "runtime": {"torch_cuda_build": torch_cuda_build},
        "opencv_wheels": opencv_wheels,
        "cv2_importable": cv2_importable,
        "cv2_version": cv2_version,
        "cv2_path": cv2_path,
        "cv2_wheel_owners": cv2_wheel_owners,
    }


def _headless_after(**overrides: str | None) -> dict[str, object]:
    return _record(
        cv2_importable=True,
        numpy="2.2.6",
        **{"opencv-python-headless": PINNED_OPENCV_VERSION, **overrides},
    )


def test_cv2_snapshot_handles_an_absent_module_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _absent(name: str) -> ModuleType:
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(stack_guard, "import_module", _absent)

    assert stack_guard._inspect_cv2({}) == {
        "cv2_importable": False,
        "cv2_version": None,
        "cv2_path": None,
        "cv2_wheel_owners": [],
    }


def test_jetson_clean_base_allows_first_pinned_headless_wheel() -> None:
    report = stack_guard.compare_snapshots(
        _record(),
        _headless_after(),
        target_platform="jetson",
    )

    assert report["allowed_jetson_opencv_addition"] is True
    assert report["opencv_before"] == {}
    assert report["opencv_after"] == {
        "opencv-python-headless": PINNED_OPENCV_VERSION
    }
    assert report["cv2_before"]["cv2_importable"] is False
    assert report["cv2_after"]["cv2_path"] == PIP_CV2_PATH


def test_jetson_rejects_wheel_when_system_or_nvidia_cv2_was_importable() -> None:
    before = _record(
        cv2_importable=True,
        cv2_version="4.5.4",
        cv2_path=SYSTEM_CV2_PATH,
    )

    with pytest.raises(RuntimeError, match="cv2 was already importable"):
        stack_guard.compare_snapshots(
            before,
            _headless_after(),
            target_platform="jetson",
        )


def test_jetson_rejects_replacing_existing_opencv_wheel() -> None:
    before = _record(
        cv2_importable=True,
        **{"opencv-python-headless": "4.10.0.84"},
    )

    with pytest.raises(RuntimeError, match="replaced an existing OpenCV wheel"):
        stack_guard.compare_snapshots(
            before,
            _headless_after(),
            target_platform="jetson",
        )


def test_jetson_rejects_removing_existing_opencv_wheel() -> None:
    before = _record(
        cv2_importable=True,
        **{"opencv-python-headless": PINNED_OPENCV_VERSION},
    )

    with pytest.raises(RuntimeError, match="removed an existing OpenCV wheel"):
        stack_guard.compare_snapshots(before, _record(), target_platform="jetson")


def test_jetson_rejects_opencv_python() -> None:
    after = _record(
        cv2_importable=True,
        **{"opencv-python": PINNED_OPENCV_VERSION},
    )

    with pytest.raises(RuntimeError, match="only opencv-python-headless"):
        stack_guard.compare_snapshots(_record(), after, target_platform="jetson")


@pytest.mark.parametrize(
    "contrib_family",
    ("opencv-contrib-python", "opencv-contrib-python-headless"),
)
def test_jetson_rejects_contrib_opencv_wheels(contrib_family: str) -> None:
    after = _record(
        cv2_importable=True,
        **{contrib_family: PINNED_OPENCV_VERSION},
    )

    with pytest.raises(RuntimeError, match="only opencv-python-headless"):
        stack_guard.compare_snapshots(_record(), after, target_platform="jetson")


def test_jetson_rejects_multiple_opencv_wheel_families() -> None:
    after = _record(
        cv2_importable=True,
        **{
            "opencv-python": PINNED_OPENCV_VERSION,
            "opencv-python-headless": PINNED_OPENCV_VERSION,
        },
    )

    with pytest.raises(RuntimeError, match="Conflicting OpenCV"):
        stack_guard.compare_snapshots(_record(), after, target_platform="jetson")


@pytest.mark.parametrize("version", ("4.10.0.84", "5.0.0.93"))
def test_jetson_rejects_unpinned_or_different_headless_version(version: str) -> None:
    after = _record(
        cv2_importable=True,
        **{"opencv-python-headless": version},
    )

    with pytest.raises(RuntimeError, match="opencv-python-headless==4.11.0.86"):
        stack_guard.compare_snapshots(_record(), after, target_platform="jetson")


def test_jetson_rejects_wheel_that_shadows_non_wheel_cv2_path() -> None:
    after = _record(
        cv2_importable=True,
        cv2_path=SYSTEM_CV2_PATH,
        cv2_wheel_owners=[],
        **{"opencv-python-headless": PINNED_OPENCV_VERSION},
    )

    with pytest.raises(RuntimeError, match="system or NVIDIA cv2 may not be shadowed"):
        stack_guard.compare_snapshots(_record(), after, target_platform="jetson")


def test_local_x86_clean_base_opencv_behavior_still_passes() -> None:
    report = stack_guard.compare_snapshots(
        _record(),
        _headless_after(),
        target_platform="local-x86",
    )

    assert report["allowed_local_x86_opencv_addition"] is True
    assert report["protected_gpu_stack_unchanged"] is True


def test_torch_change_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="torch"):
        stack_guard.compare_snapshots(
            _record(),
            _record(torch="2.13.0"),
            target_platform="jetson",
        )


def test_torchvision_change_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="torchvision"):
        stack_guard.compare_snapshots(
            _record(),
            _record(torchvision="0.28.0"),
            target_platform="jetson",
        )


def test_triton_change_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="triton"):
        stack_guard.compare_snapshots(
            _record(),
            _record(triton="3.7.0"),
            target_platform="jetson",
        )


def test_torch_cuda_build_change_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="torch-cuda-build"):
        stack_guard.compare_snapshots(
            _record(),
            _record(torch_cuda_build="13.3"),
            target_platform="jetson",
        )


@pytest.mark.parametrize(
    ("package", "replacement"),
    (
        ("nvidia-cuda-runtime-cu13", "13.3.0"),
        ("nvidia-cudnn-cu13", "9.17.0"),
        ("nvidia-cublas-cu13", "13.3.0"),
        ("nvidia-nccl-cu13", "2.29.0"),
    ),
)
def test_protected_nvidia_package_changes_are_rejected(
    package: str,
    replacement: str,
) -> None:
    with pytest.raises(RuntimeError, match=package):
        stack_guard.compare_snapshots(
            _record(),
            _record(**{package: replacement}),
            target_platform="jetson",
        )


def test_real_jetson_before_and_after_fixture_passes() -> None:
    before = _record(numpy="2.1.0")
    after = _headless_after()

    report = stack_guard.compare_snapshots(before, after, target_platform="jetson")

    assert report["allowed_jetson_opencv_addition"] is True
    assert report["protected_gpu_stack_unchanged"] is True
    assert report["torch"] == "2.12.0a0+5aff3928d8.nv26.5.50603568"
    assert report["torchvision"] == "0.27.0a0+6214cb20.nv26.5.50603568"
    assert report["torch_cuda_build"] == "13.2"
    assert report["numpy_before"] == "2.1.0"
    assert report["numpy_after"] == "2.2.6"
