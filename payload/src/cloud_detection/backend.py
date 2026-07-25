from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

DEFAULT_WEIGHTS_FOLDER = (
    Path(__file__).resolve().parents[2] / "models" / "cloudsen12"
)


class BackendError(RuntimeError):
    """Raised when the pretrained model cannot be loaded or executed."""


@dataclass(frozen=True)
class BackendPrediction:
    scores: np.ndarray
    score_kind: str


class CloudBackend(Protocol):
    def predict(self, tile: np.ndarray) -> BackendPrediction:
        """Return four class scores shaped (4, H, W)."""


class CloudSEN12Backend:
    """Adapter for the official CloudSEN12 ``dtacs4bands`` model.

    The upstream public ``predict`` API returns a discrete semantic mask. The distributed
    TorchScript model itself emits class logits, so this adapter first attempts to call the
    wrapped TorchScript module directly and converts logits to softmax scores. If upstream
    internals change, it safely falls back to the documented semantic API and emits one-hot
    scores. Softmax scores are model confidence values, not calibrated probabilities.
    """

    def __init__(
        self,
        name: str = "dtacs4bands",
        weights_folder: str | Path = DEFAULT_WEIGHTS_FOLDER,
        device: str = "auto",
        expected_sha256: str | None = None,
    ) -> None:
        if name != "dtacs4bands":
            raise BackendError("Only dtacs4bands is supported by this four-band pipeline.")

        try:
            import torch
            from cloudsen12_models import cloudsen12
        except ImportError as exc:
            raise BackendError(
                "Install cloudsen12_models and torch before loading the real backend."
            ) from exc

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise BackendError("CUDA was requested but is unavailable.")
        Path(weights_folder).mkdir(parents=True, exist_ok=True)
        model_path = Path(weights_folder) / f"{name}.pt"
        if expected_sha256 is not None:
            if not model_path.is_file():
                raise BackendError(
                    f"Missing verified checkpoint {model_path}. Run "
                    "payload/scripts/download_cloud_weights.py while online."
                )
            digest = hashlib.sha256()
            with model_path.open("rb") as file:
                for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            actual_sha256 = digest.hexdigest()
            if actual_sha256 != expected_sha256:
                raise BackendError(
                    "Cloud checkpoint checksum mismatch: "
                    f"expected {expected_sha256}, got {actual_sha256}."
                )

        try:
            self.model = cloudsen12.load_model_by_name(
                name=name,
                weights_folder=str(weights_folder),
                device=self.device,
            )
        except Exception as exc:
            raise BackendError(
                "Could not load the pretrained dtacs4bands weights. Run "
                "payload/scripts/download_cloud_weights.py while online and retry."
            ) from exc

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        values = logits.astype(np.float32, copy=False)
        values = values - values.max(axis=0, keepdims=True)
        exponentials = np.exp(values)
        return exponentials / np.maximum(exponentials.sum(axis=0, keepdims=True), 1e-8)

    def predict(self, tile: np.ndarray) -> BackendPrediction:
        if tile.ndim != 3 or tile.shape[0] != 4:
            raise BackendError(f"Expected an array shaped (4,H,W), got {tile.shape}.")
        tile = tile.astype(np.float32, copy=False)

        internal_model = getattr(self.model, "model", None)
        if internal_model is not None:
            tensor = self.torch.from_numpy(tile[None]).to(self.device)
            try:
                with self.torch.inference_mode():
                    logits = internal_model(tensor)[0]
                logits_numpy = logits.detach().cpu().numpy()
                if logits_numpy.ndim == 3 and logits_numpy.shape[0] == 4:
                    return BackendPrediction(
                        scores=self._softmax(logits_numpy),
                        score_kind="softmax_confidence",
                    )
            except Exception:
                pass

        try:
            semantic = np.asarray(self.model.predict(tile)).squeeze()
        except Exception as exc:
            raise BackendError("CloudSEN12 inference failed.") from exc

        if (
            semantic.ndim != 2
            or semantic.size == 0
            or semantic.min() < 0
            or semantic.max() > 3
        ):
            raise BackendError(f"Unexpected upstream output shape/range: {semantic.shape}.")

        one_hot = np.eye(4, dtype=np.float32)[semantic.astype(np.int64)].transpose(2, 0, 1)
        return BackendPrediction(scores=one_hot, score_kind="hard_one_hot")


class TestBackend:
    """Deterministic backend used only for local unit tests."""

    def predict(self, tile: np.ndarray) -> BackendPrediction:
        if tile.ndim != 3 or tile.shape[0] != 4:
            raise BackendError(f"Expected an array shaped (4,H,W), got {tile.shape}.")
        _, red, green, blue = tile
        brightness = (red + green + blue) / 3
        whiteness = 1 - np.clip(
            np.std(np.stack([red, green, blue]), axis=0) / 0.25,
            0,
            1,
        )
        cloud = np.clip((brightness - 0.35) * 2.5, 0, 1) * whiteness
        thin = np.clip((brightness - 0.20) * 1.5, 0, 0.6) * whiteness * (1 - cloud)
        shadow = np.clip((0.12 - brightness) * 3, 0, 0.7)
        clear = np.clip(1 - cloud - thin - shadow, 0, 1)
        scores = np.stack([clear, cloud, thin, shadow]).astype(np.float32)
        scores /= np.maximum(scores.sum(axis=0, keepdims=True), 1e-8)
        return BackendPrediction(scores=scores, score_kind="synthetic_probability")
