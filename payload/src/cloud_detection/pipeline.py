from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from .backend import CloudBackend, OmniCloudMaskBackend
from .config import load_config
from .io import read_geotiff, write_mask
from .postprocessing import postprocess
from .preprocessing import normalize_reflectance, strict_valid_mask
from .preview import save_preview
from .tiling import reconstruct, split_tiles
from .types import CloudDetectionResult


class CloudDetectionPipeline:
    def __init__(self, config: dict, backend: CloudBackend | None = None) -> None:
        self.cfg = config
        model_cfg = config["model"]
        weights_folder = os.environ.get(
            "OMNICLOUDMASK_MODEL_DIR",
            model_cfg["weights_folder"],
        )
        self.backend = backend or OmniCloudMaskBackend(
            name=model_cfg["name"],
            weights_folder=weights_folder,
            device=model_cfg.get("device", "auto"),
            expected_sha256=model_cfg.get("expected_sha256"),
            inference_dtype=model_cfg.get("inference_dtype", "fp32"),
            patch_size=int(model_cfg.get("patch_size", 1000)),
            patch_overlap=int(model_cfg.get("patch_overlap", 300)),
            batch_size=int(model_cfg.get("batch_size", 1)),
        )

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        backend: CloudBackend | None = None,
    ) -> CloudDetectionPipeline:
        config_path = Path(path).resolve()
        config = load_config(config_path)
        weights_folder = Path(config["model"]["weights_folder"])
        if not weights_folder.is_absolute():
            config["model"]["weights_folder"] = str((config_path.parent / weights_folder).resolve())
        return cls(config, backend)

    def predict_array(
        self,
        image: np.ndarray,
        invalid: np.ndarray | None = None,
    ) -> CloudDetectionResult:
        started = time.perf_counter()
        if image.ndim != 3 or image.shape[0] != 4:
            raise ValueError(f"Expected four-band C x H x W image, found {image.shape}.")
        if invalid is None:
            invalid = np.zeros(image.shape[1:], dtype=bool)

        tiling_cfg = self.cfg["tiling"]
        tiles, windows, original_shape, padded_shape = split_tiles(
            image,
            int(tiling_cfg["size"]),
            int(tiling_cfg["overlap"]),
            tiling_cfg["padding_mode"],
        )

        backend_predictions = [
            self.backend.predict(tile.astype(np.float32, copy=False)) for tile in tiles
        ]
        score_kinds = {prediction.score_kind for prediction in backend_predictions}
        score_kind = next(iter(score_kinds)) if len(score_kinds) == 1 else "mixed"
        class_scores = reconstruct(
            [prediction.scores for prediction in backend_predictions],
            windows,
            original_shape,
            padded_shape,
        )
        semantic = class_scores.argmax(axis=0).astype(np.uint8)
        unusable = postprocess(
            semantic,
            self.cfg["classes"],
            self.cfg["postprocessing"],
            invalid,
        )

        classes = self.cfg["classes"]
        thick = semantic == int(classes["thick_cloud"])
        thin = semantic == int(classes["thin_cloud"])
        shadow = semantic == int(classes["cloud_shadow"])
        total = semantic.size

        def percentage(mask: np.ndarray) -> float:
            return 100.0 * float(mask.sum()) / total

        thick_percentage = percentage(thick)
        thin_percentage = percentage(thin)
        cloud_percentage = thick_percentage + thin_percentage
        shadow_percentage = percentage(shadow)
        invalid_percentage = percentage(invalid)
        unusable_percentage = percentage(unusable == 1)
        usable_percentage = 100.0 - unusable_percentage

        decision_cfg = self.cfg["decision"]
        if unusable_percentage >= float(decision_cfg["reject_min_unusable_percentage"]):
            decision = "REJECT"
        elif unusable_percentage > float(decision_cfg["process_max_unusable_percentage"]):
            decision = "PROCESS_CLEAR_AREAS"
        else:
            decision = "PROCESS"

        return CloudDetectionResult(
            semantic_mask=semantic,
            unusable_mask=unusable,
            class_scores=class_scores,
            score_kind=score_kind,
            thick_cloud_percentage=thick_percentage,
            thin_cloud_percentage=thin_percentage,
            cloud_percentage=cloud_percentage,
            shadow_percentage=shadow_percentage,
            invalid_percentage=invalid_percentage,
            usable_percentage=usable_percentage,
            unusable_percentage=unusable_percentage,
            decision=decision,
            runtime_seconds=time.perf_counter() - started,
        )

    def predict_file(
        self,
        input_path: str | Path,
        output_root: str | Path,
    ) -> CloudDetectionResult:
        input_cfg = self.cfg["input"]
        array, profile = read_geotiff(
            input_path,
            input_cfg["band_names"],
            require_descriptions=bool(input_cfg.get("require_band_descriptions", False)),
        )
        image, invalid = normalize_reflectance(
            array,
            float(input_cfg["reflectance_scale"]),
            input_cfg.get("clip_min"),
            input_cfg.get("clip_max"),
            input_cfg.get("nodata_value"),
        )
        if bool(input_cfg.get("strict_positive_rgn", True)):
            invalid |= ~strict_valid_mask(image[[1, 2, 0]])
            image[:, invalid] = 0.0
        result = self.predict_array(image, invalid)

        root = Path(output_root)
        stem = Path(input_path).stem
        semantic_path = root / "cloud_masks" / f"{stem}_semantic.tif"
        unusable_path = root / "cloud_masks" / f"{stem}_unusable.tif"
        preview_path = root / "visualisations" / f"{stem}_preview.png"
        metadata_path = root / "metadata" / f"{stem}.json"

        write_mask(
            semantic_path,
            result.semantic_mask,
            profile,
            "uint8",
            255,
            "0 clear, 1 thick cloud, 2 thin cloud, 3 cloud shadow",
        )
        write_mask(
            unusable_path,
            result.unusable_mask,
            profile,
            "uint8",
            255,
            "0 usable, 1 unusable",
        )

        files = {
            "semantic_mask": str(semantic_path),
            "unusable_mask": str(unusable_path),
        }

        if bool(self.cfg["output"].get("save_class_scores", True)):
            class_names = ["clear", "thick_cloud", "thin_cloud", "cloud_shadow"]
            for class_index, class_name in enumerate(class_names):
                score_path = root / "class_scores" / f"{stem}_{class_name}_score.tif"
                write_mask(
                    score_path,
                    result.class_scores[class_index],
                    profile,
                    "float32",
                    None,
                    f"{class_name} score ({result.score_kind})",
                )
                files[f"{class_name}_score"] = str(score_path)

        if bool(self.cfg["output"].get("save_preview", True)):
            save_preview(preview_path, image, result.semantic_mask, result.unusable_mask)
            files["preview"] = str(preview_path)

        result.output_files = files
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(result.metadata(), indent=2),
            encoding="utf-8",
        )
        files["metadata"] = str(metadata_path)
        return result
