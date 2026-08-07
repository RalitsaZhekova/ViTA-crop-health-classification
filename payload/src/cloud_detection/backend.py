from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

OMNICLOUDMASK_MODEL_VERSION = 4.0
OMNICLOUDMASK_PACKAGE_VERSION = "1.7.1"
OMNICLOUDMASK_INTEGRATION_COMMIT = "a991819bf4462238b12dd0191e5dfaa35b69f39a"
OMNICLOUDMASK_UPSTREAM_COMMIT = "fbc6d3f5665eb3425fb2474cb3e6f574e2e71a1b"
OMNICLOUDMASK_MODEL_FILES = {
    "PM_model_OCM_7.97_R_G_NIR_3_smp_edgenext_small.usi_in1k_PT_state.safetensors": (
        "d5fe67ad00f6fdb73eb8382ad925849e97fb82cedb46225344afff1a734a9c1d"
    ),
    "PM_model_OCM_7.97_R_G_NIR_3_smp_regnety_004.pycls_in1k_PT_state.safetensors": (
        "7f6e4202e17ee73efa4aba7abb5c34f4f90a9f7eb42480820714994dff2db660"
    ),
}
OMNICLOUDMASK_ENSEMBLE_SHA256 = "ab8f039866d6714b249f850779b9523b5f6afb55ee891077a71db0fdebc9b529"
DEFAULT_WEIGHTS_FOLDER = Path(__file__).resolve().parents[2] / "models" / "omnicloudmask"


class BackendError(RuntimeError):
    """Raised when the pretrained model cannot be loaded or executed."""


@dataclass(frozen=True)
class BackendPrediction:
    scores: np.ndarray
    score_kind: str


class CloudBackend(Protocol):
    def predict(self, tile: np.ndarray) -> BackendPrediction:
        """Return four class scores shaped (4, H, W)."""


def checkpoint_sha256(path: Path) -> str:
    """Hash one checkpoint without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensemble_sha256(weights_folder: str | Path) -> str:
    """Verify every V4 component and return the stable ensemble fingerprint."""
    folder = Path(weights_folder)
    digest = hashlib.sha256()
    for filename, expected in sorted(OMNICLOUDMASK_MODEL_FILES.items()):
        path = folder / filename
        if not path.is_file():
            raise BackendError(
                f"Missing verified OmniCloudMask checkpoint {path}. Run "
                "payload/scripts/download_cloud_weights.py while online."
            )
        actual = checkpoint_sha256(path)
        if actual != expected:
            raise BackendError(
                f"OmniCloudMask checkpoint checksum mismatch for {filename}: "
                f"expected {expected}, got {actual}."
            )
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(actual))
    return digest.hexdigest()


class OmniCloudMaskBackend:
    """Persistent adapter for the dual-sensor OmniCloudMask V4 ensemble.

    The surrounding pipeline intentionally retains its established four-band
    ``[NIR, Red, Green, Blue]`` contract. OmniCloudMask receives only its native
    ``[Red, Green, NIR]`` channels. Blue remains available to previews and later
    stages and is never passed to the cloud ensemble.
    """

    def __init__(
        self,
        name: str = "omnicloudmask_v4",
        weights_folder: str | Path = DEFAULT_WEIGHTS_FOLDER,
        device: str = "auto",
        expected_sha256: str | None = OMNICLOUDMASK_ENSEMBLE_SHA256,
        inference_dtype: str = "fp32",
        patch_size: int = 1000,
        patch_overlap: int = 300,
        batch_size: int = 1,
    ) -> None:
        if name != "omnicloudmask_v4":
            raise BackendError("Only the reviewed OmniCloudMask V4 ensemble is supported.")
        if patch_size < 32 or patch_overlap < 0 or patch_overlap >= patch_size:
            raise BackendError("Invalid OmniCloudMask patch size or overlap.")
        if batch_size < 1:
            raise BackendError("OmniCloudMask batch size must be positive.")

        try:
            import torch
            from omnicloudmask import __version__ as package_version
            from omnicloudmask import predict_from_array
            from omnicloudmask.cloud_mask import collect_models
            from omnicloudmask.model_utils import get_torch_dtype
        except ImportError as exc:
            raise BackendError(
                "Install omnicloudmask==1.7.1 and torch before loading the real backend."
            ) from exc
        if package_version != OMNICLOUDMASK_PACKAGE_VERSION:
            raise BackendError(
                "Unexpected OmniCloudMask package version: "
                f"expected {OMNICLOUDMASK_PACKAGE_VERSION}, got {package_version}."
            )

        self.torch = torch
        self._predict_from_array = predict_from_array
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise BackendError("CUDA was requested but is unavailable.")
        self.inference_dtype = inference_dtype
        self._torch_dtype = get_torch_dtype(inference_dtype)
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.batch_size = batch_size
        self.weights_folder = Path(weights_folder)
        requested_backend = os.environ.get("VITA_CLOUD_BACKEND", "pytorch").strip().casefold()
        if requested_backend not in {"pytorch", "tensorrt"}:
            raise BackendError("VITA_CLOUD_BACKEND must be pytorch or tensorrt")
        self.execution_backend = requested_backend

        actual_ensemble_sha256 = ensemble_sha256(self.weights_folder)
        if expected_sha256 is not None and actual_ensemble_sha256 != expected_sha256:
            raise BackendError(
                "OmniCloudMask ensemble checksum mismatch: "
                f"expected {expected_sha256}, got {actual_ensemble_sha256}."
            )

        try:
            models = collect_models(
                custom_models=None,
                inference_device=self.device,
                inference_dtype=self._torch_dtype,
                source="hugging_face",
                destination_model_dir=self.weights_folder,
                model_version=OMNICLOUDMASK_MODEL_VERSION,
                compile_models=False,
                patch_size=self.patch_size,
                batch_size=self.batch_size,
            )
            if requested_backend == "tensorrt":
                from cloud_detection.tensorrt_backend import CloudTensorRTRouter

                self._tensorrt_router = CloudTensorRTRouter(
                    models,
                    device=self.device,
                    dtype=self._torch_dtype,
                )
                self.models = [self._tensorrt_router]
            else:
                self._tensorrt_router = None
                self.models = models
        except Exception as exc:
            raise BackendError(
                "Could not load the verified OmniCloudMask V4 ensemble. Run "
                "payload/scripts/download_cloud_weights.py while online and retry."
            ) from exc

    def _prepare_input(self, tile: np.ndarray) -> np.ndarray | None:
        if tile.ndim != 3 or tile.shape[0] != 4:
            raise BackendError(f"Expected an array shaped (4,H,W), got {tile.shape}.")
        if min(tile.shape[1:]) < 32:
            raise BackendError("OmniCloudMask tiles must be at least 32 by 32 pixels.")

        # Existing pipeline order is NIR, Red, Green, Blue. The upstream model
        # order is Red, Green, NIR. Invalid values are set to the upstream
        # baseline's zero nodata value before its dynamic patch normalization.
        image_rgn = tile[[1, 2, 0]].astype(np.float32, copy=True)
        finite = np.all(np.isfinite(image_rgn), axis=0)
        positive = np.all(image_rgn > np.finfo(np.float32).tiny, axis=0)
        valid = finite & positive
        if not np.any(valid):
            return None
        image_rgn[:, ~valid] = 0.0
        return image_rgn

    def _run_upstream(
        self,
        image_rgn: np.ndarray,
        *,
        export_confidence: bool,
    ) -> np.ndarray:
        try:
            # inference_mode removes autograd view/version bookkeeping in addition
            # to the no_grad guards used inside OmniCloudMask.
            with self.torch.inference_mode():
                prediction = self._predict_from_array(
                    image_rgn,
                    patch_size=min(self.patch_size, *image_rgn.shape[1:]),
                    patch_overlap=min(
                        self.patch_overlap,
                        max(0, min(self.patch_size, *image_rgn.shape[1:]) // 2),
                    ),
                    batch_size=self.batch_size,
                    inference_device=self.device,
                    mosaic_device=self.device,
                    inference_dtype=self.inference_dtype,
                    export_confidence=export_confidence,
                    softmax_output=export_confidence,
                    no_data_value=0.0,
                    apply_no_data_mask=False,
                    custom_models=self.models,
                    pred_classes=4,
                    model_version=OMNICLOUDMASK_MODEL_VERSION,
                )
        except Exception as exc:
            raise BackendError("OmniCloudMask V4 inference failed.") from exc
        return np.asarray(prediction)

    def predict(self, tile: np.ndarray) -> BackendPrediction:
        image_rgn = self._prepare_input(tile)
        if image_rgn is None:
            scores = np.zeros((4, *tile.shape[1:]), dtype=np.float32)
            scores[0] = 1.0
            return BackendPrediction(scores=scores, score_kind="softmax_confidence")

        scores = np.asarray(
            self._run_upstream(image_rgn, export_confidence=True),
            dtype=np.float32,
        )
        expected_shape = (4, *tile.shape[1:])
        if scores.shape != expected_shape or not np.isfinite(scores).all():
            raise BackendError(f"Unexpected OmniCloudMask output shape or values: {scores.shape}.")
        # OmniCloudMask clips each softmax channel to [0.001, 0.999], so its
        # exported channels sum to approximately 1.004. Re-normalize without
        # changing argmax classes to preserve this pipeline's score contract.
        scores /= np.maximum(scores.sum(axis=0, keepdims=True), 1e-8)
        return BackendPrediction(scores=scores, score_kind="softmax_confidence")

    def predict_semantic(self, tile: np.ndarray) -> np.ndarray:
        """Return only class IDs, avoiding a four-channel confidence transfer to CPU."""
        image_rgn = self._prepare_input(tile)
        if image_rgn is None:
            return np.zeros(tile.shape[1:], dtype=np.uint8)
        semantic = np.asarray(
            self._run_upstream(image_rgn, export_confidence=False),
            dtype=np.uint8,
        )
        expected_shape = (1, *tile.shape[1:])
        if semantic.shape != expected_shape or np.any(semantic > 3):
            raise BackendError(f"Unexpected OmniCloudMask semantic output: {semantic.shape}.")
        return semantic[0]

    def warmup(self, additional_patch_sizes: tuple[int, ...] = ()) -> list[dict[str, int]]:
        """Warm the CUDA convolution profiles used by the fixed MVP scenes."""
        profiles = {(1, self.patch_size)}
        for patch_size in additional_patch_sizes:
            if patch_size < 32 or patch_size > self.patch_size:
                raise BackendError(
                    f"Cloud warmup patch size must be in 32..{self.patch_size}, got {patch_size}."
                )
            profiles.add((1, patch_size))
            profiles.add((self.batch_size, patch_size))

        with self.torch.inference_mode():
            for batch_size, patch_size in sorted(profiles):
                sample = self.torch.zeros(
                    (batch_size, 3, patch_size, patch_size),
                    device=self.device,
                    dtype=self._torch_dtype,
                )
                for model in self.models:
                    model(sample)
                del sample
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        return [
            {"batch_size": batch_size, "patch_size": patch_size}
            for batch_size, patch_size in sorted(profiles)
        ]

    @property
    def tensorrt_engine_count(self) -> int:
        router = self._tensorrt_router
        return router.engine_count if router is not None else 0

    @property
    def tensorrt_profiles(self) -> list[dict[str, object]]:
        router = self._tensorrt_router
        return router.profiles if router is not None else []

    def freeze_tensorrt_profiles(self) -> None:
        if self._tensorrt_router is not None:
            self._tensorrt_router.freeze()
