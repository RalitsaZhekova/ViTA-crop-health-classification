"""Transparent spectral crop-condition scoring for one crop region."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from prithvi_ground.health import HealthLayers

FloatArray = NDArray[np.floating[Any]]
BoolArray = NDArray[np.bool_]

CONDITION_ALGORITHM_VERSION = "spectral-condition-v1"
SCORED_INDEX_NAMES = ("ndvi", "gndvi", "evi", "savi")


@dataclass(frozen=True)
class ConditionConfig:
    """Configurable prototype priors, not universal agronomic thresholds."""

    ndvi_reference: tuple[float, float] = (0.20, 0.80)
    gndvi_reference: tuple[float, float] = (0.15, 0.70)
    evi_reference: tuple[float, float] = (0.10, 0.80)
    savi_reference: tuple[float, float] = (0.15, 0.80)
    ndvi_weight: float = 0.40
    gndvi_weight: float = 0.25
    evi_weight: float = 0.20
    savi_weight: float = 0.15
    median_weight: float = 0.70
    lower_quartile_weight: float = 0.30
    relative_anomaly_z: float = 2.5
    minimum_score_deficit: float = 10.0
    minimum_robust_scale: float = 5.0
    maximum_spatial_penalty: float = 20.0
    absolute_low_score: float = 35.0
    nominal_minimum: float = 75.0
    watch_minimum: float = 55.0
    moderate_minimum: float = 35.0
    minimum_analysis_pixels: int = 64
    minimum_analysis_percentage: float = 0.10
    target_evidence_pixels: int = 512
    target_evidence_percentage: float = 5.0
    watch_alert_percentage: float = 5.0

    def __post_init__(self) -> None:
        for name, limits in self.references.items():
            low, high = limits
            if not np.isfinite(low) or not np.isfinite(high) or low >= high:
                raise ValueError(f"{name} reference must contain finite low < high values")
        if any(weight < 0 for weight in self.weights.values()):
            raise ValueError("Index weights must be non-negative")
        if sum(self.weights.values()) <= 0:
            raise ValueError("At least one index weight must be positive")
        if not np.isclose(
            self.median_weight + self.lower_quartile_weight,
            1.0,
        ):
            raise ValueError("Median and lower-quartile weights must sum to 1")
        if self.relative_anomaly_z <= 0:
            raise ValueError("relative_anomaly_z must be positive")
        if self.minimum_score_deficit < 0 or self.minimum_robust_scale <= 0:
            raise ValueError("Robust anomaly limits are invalid")
        if self.maximum_spatial_penalty < 0:
            raise ValueError("maximum_spatial_penalty must be non-negative")
        if not (
            100 >= self.nominal_minimum > self.watch_minimum > self.moderate_minimum >= 0
        ):
            raise ValueError("Condition label thresholds must be strictly descending")
        if self.minimum_analysis_pixels <= 0 or self.target_evidence_pixels <= 0:
            raise ValueError("Pixel-count thresholds must be positive")
        if not 0 <= self.minimum_analysis_percentage <= 100:
            raise ValueError("minimum_analysis_percentage must be within 0..100")
        if not 0 < self.target_evidence_percentage <= 100:
            raise ValueError("target_evidence_percentage must be within (0, 100]")
        if not 0 <= self.watch_alert_percentage <= 100:
            raise ValueError("watch_alert_percentage must be within 0..100")

    @property
    def references(self) -> dict[str, tuple[float, float]]:
        return {
            "ndvi": self.ndvi_reference,
            "gndvi": self.gndvi_reference,
            "evi": self.evi_reference,
            "savi": self.savi_reference,
        }

    @property
    def weights(self) -> dict[str, float]:
        return {
            "ndvi": self.ndvi_weight,
            "gndvi": self.gndvi_weight,
            "evi": self.evi_weight,
            "savi": self.savi_weight,
        }


@dataclass(frozen=True)
class ConditionLayers:
    """Per-pixel score and anomaly evidence for map output."""

    component_scores: dict[str, FloatArray]
    condition_score: FloatArray
    robust_deficit_z: FloatArray
    relative_anomaly_mask: BoolArray
    low_vigor_mask: BoolArray
    alert_mask: BoolArray


@dataclass(frozen=True)
class ConditionAssessment:
    """Region-level spectral condition result with explicit limitations."""

    status: str
    label: str
    condition_score: float | None
    absolute_vigor_score: float | None
    spatial_penalty: float | None
    evidence_quality_score: float
    evidence_quality_label: str
    analysis_pixels: int
    analysis_percentage: float
    relative_anomaly_percentage: float | None
    low_vigor_percentage: float | None
    component_median_scores: dict[str, float | None]
    configuration: dict[str, Any]
    explanations: tuple[str, ...]
    limitations: tuple[str, ...]
    algorithm_version: str = CONDITION_ALGORITHM_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConditionResult:
    assessment: ConditionAssessment
    layers: ConditionLayers


def _linear_score(values: FloatArray, low: float, high: float) -> FloatArray:
    output = np.full(values.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(values)
    output[finite] = np.clip(
        100.0 * (values[finite] - low) / (high - low),
        0.0,
        100.0,
    )
    return output


def _combine_component_scores(
    component_scores: dict[str, FloatArray],
    analysis_mask: BoolArray,
    weights: dict[str, float],
) -> FloatArray:
    numerator = np.zeros(analysis_mask.shape, dtype=np.float64)
    denominator = np.zeros(analysis_mask.shape, dtype=np.float64)
    for name in SCORED_INDEX_NAMES:
        score = component_scores[name]
        available = analysis_mask & np.isfinite(score)
        numerator[available] += weights[name] * score[available]
        denominator[available] += weights[name]

    combined = np.full(analysis_mask.shape, np.nan, dtype=np.float32)
    usable = analysis_mask & (denominator > 0)
    combined[usable] = (numerator[usable] / denominator[usable]).astype(np.float32)
    return combined


def _evidence_quality(
    *,
    analysis_pixels: int,
    analysis_percentage: float,
    mean_crop_probability: float | None,
    config: ConditionConfig,
) -> tuple[float, str]:
    factors = [
        min(1.0, analysis_pixels / config.target_evidence_pixels),
        min(1.0, analysis_percentage / config.target_evidence_percentage),
    ]
    if mean_crop_probability is not None:
        if not 0 <= mean_crop_probability <= 1:
            raise ValueError("mean_crop_probability must be between 0 and 1")
        factors.append(mean_crop_probability)
    score = 100.0 * float(np.mean(factors))
    if score >= 75:
        label = "HIGH"
    elif score >= 45:
        label = "MEDIUM"
    else:
        label = "LOW"
    return score, label


def _label_for_score(score: float, config: ConditionConfig) -> str:
    if score >= config.nominal_minimum:
        return "Nominal"
    if score >= config.watch_minimum:
        return "Watch"
    if score >= config.moderate_minimum:
        return "Moderate anomaly"
    return "High anomaly"


def _empty_spatial_layers(
    component_scores: dict[str, FloatArray],
    pixel_score: FloatArray,
) -> ConditionLayers:
    return ConditionLayers(
        component_scores=component_scores,
        condition_score=pixel_score,
        robust_deficit_z=np.full(pixel_score.shape, np.nan, dtype=np.float32),
        relative_anomaly_mask=np.zeros(pixel_score.shape, dtype=bool),
        low_vigor_mask=np.zeros(pixel_score.shape, dtype=bool),
        alert_mask=np.zeros(pixel_score.shape, dtype=bool),
    )


def assess_crop_condition(
    health_layers: HealthLayers,
    *,
    mean_crop_probability: float | None = None,
    config: ConditionConfig | None = None,
) -> ConditionResult:
    """Assess spectral condition without inferring a disease or causal stressor."""
    cfg = config or ConditionConfig()
    missing = [name for name in SCORED_INDEX_NAMES if name not in health_layers.values]
    if missing:
        raise ValueError(f"Health layers are missing scored indices: {missing}")

    analysis_mask = np.asarray(health_layers.analysis_mask, dtype=bool)
    if analysis_mask.ndim != 2:
        raise ValueError("Health analysis mask must be a 2D array")
    for name, values in health_layers.values.items():
        if np.asarray(values).shape != analysis_mask.shape:
            raise ValueError(f"Health layer {name} does not match the analysis mask")

    component_scores: dict[str, FloatArray] = {}
    for name in SCORED_INDEX_NAMES:
        component_score = _linear_score(
            np.asarray(health_layers.values[name], dtype=np.float32),
            *cfg.references[name],
        )
        component_score[~analysis_mask] = np.nan
        component_scores[name] = component_score
    pixel_score = _combine_component_scores(component_scores, analysis_mask, cfg.weights)
    valid_score = analysis_mask & np.isfinite(pixel_score)
    analysis_pixels = int(np.count_nonzero(valid_score))
    total_pixels = int(analysis_mask.size)
    analysis_percentage = 100.0 * analysis_pixels / total_pixels if total_pixels else 0.0
    quality_score, quality_label = _evidence_quality(
        analysis_pixels=analysis_pixels,
        analysis_percentage=analysis_percentage,
        mean_crop_probability=mean_crop_probability,
        config=cfg,
    )

    limitations = (
        "The score represents spectral crop condition, not agronomic diagnosis.",
        "Prototype reference ranges are not crop-, cultivar-, season- or growth-stage-specific.",
        "A single scene cannot distinguish stress from harvest, senescence or fallow land.",
        "Evidence quality is a coverage indicator, not a calibrated probability of correctness.",
    )
    if (
        analysis_pixels < cfg.minimum_analysis_pixels
        or analysis_percentage < cfg.minimum_analysis_percentage
    ):
        assessment = ConditionAssessment(
            status="INSUFFICIENT_DATA",
            label="Insufficient data",
            condition_score=None,
            absolute_vigor_score=None,
            spatial_penalty=None,
            evidence_quality_score=quality_score,
            evidence_quality_label=quality_label,
            analysis_pixels=analysis_pixels,
            analysis_percentage=analysis_percentage,
            relative_anomaly_percentage=None,
            low_vigor_percentage=None,
            component_median_scores={name: None for name in SCORED_INDEX_NAMES},
            configuration=asdict(cfg),
            explanations=(
                "Too few clear, confident crop pixels were available for assessment.",
            ),
            limitations=limitations,
        )
        return ConditionResult(assessment, _empty_spatial_layers(component_scores, pixel_score))

    values = pixel_score[valid_score].astype(np.float64)
    median_score = float(np.median(values))
    lower_quartile_score = float(np.percentile(values, 25))
    absolute_vigor_score = (
        cfg.median_weight * median_score
        + cfg.lower_quartile_weight * lower_quartile_score
    )

    median_absolute_deviation = float(np.median(np.abs(values - median_score)))
    robust_scale = max(1.4826 * median_absolute_deviation, cfg.minimum_robust_scale)
    robust_deficit_z = np.full(pixel_score.shape, np.nan, dtype=np.float32)
    robust_deficit_z[valid_score] = (
        (median_score - pixel_score[valid_score]) / robust_scale
    ).astype(np.float32)
    score_deficit = median_score - pixel_score
    relative_anomaly_mask = (
        valid_score
        & (robust_deficit_z >= cfg.relative_anomaly_z)
        & (score_deficit >= cfg.minimum_score_deficit)
    )
    low_vigor_mask = valid_score & (pixel_score < cfg.absolute_low_score)
    alert_mask = relative_anomaly_mask | low_vigor_mask

    relative_anomaly_fraction = float(np.count_nonzero(relative_anomaly_mask) / analysis_pixels)
    low_vigor_fraction = float(np.count_nonzero(low_vigor_mask) / analysis_pixels)
    spatial_penalty = cfg.maximum_spatial_penalty * relative_anomaly_fraction
    final_score = float(np.clip(absolute_vigor_score - spatial_penalty, 0.0, 100.0))
    label = _label_for_score(final_score, cfg)
    alert_fraction = max(relative_anomaly_fraction, low_vigor_fraction)
    if label == "Nominal" and 100.0 * alert_fraction >= cfg.watch_alert_percentage:
        label = "Watch"

    component_medians = {
        name: float(np.median(score[valid_score & np.isfinite(score)]))
        if np.any(valid_score & np.isfinite(score))
        else None
        for name, score in component_scores.items()
    }
    explanations = [
        f"Absolute spectral-vigor component: {absolute_vigor_score:.1f}/100.",
    ]
    weak_components = [
        name.upper()
        for name, score in component_medians.items()
        if score is not None and score < 40.0
    ]
    if weak_components:
        explanations.append(
            "Low prototype-range support from: " + ", ".join(weak_components) + "."
        )
    if relative_anomaly_fraction > 0:
        explanations.append(
            f"{100.0 * relative_anomaly_fraction:.1f}% of analyzed crop pixels "
            "were robustly below the crop-region median."
        )
    else:
        explanations.append("No robust within-crop-region deficit was detected.")
    if low_vigor_fraction > 0:
        explanations.append(
            f"{100.0 * low_vigor_fraction:.1f}% of analyzed pixels fell below "
            "the absolute low-vigor score threshold."
        )
    explanations.append(
        "Interpret the label as a screening priority; field inspection or agronomic "
        "evidence is required to identify a cause."
    )

    assessment = ConditionAssessment(
        status="MEASURED",
        label=label,
        condition_score=final_score,
        absolute_vigor_score=absolute_vigor_score,
        spatial_penalty=spatial_penalty,
        evidence_quality_score=quality_score,
        evidence_quality_label=quality_label,
        analysis_pixels=analysis_pixels,
        analysis_percentage=analysis_percentage,
        relative_anomaly_percentage=100.0 * relative_anomaly_fraction,
        low_vigor_percentage=100.0 * low_vigor_fraction,
        component_median_scores=component_medians,
        configuration=asdict(cfg),
        explanations=tuple(explanations),
        limitations=limitations,
    )
    layers = ConditionLayers(
        component_scores=component_scores,
        condition_score=pixel_score,
        robust_deficit_z=robust_deficit_z,
        relative_anomaly_mask=relative_anomaly_mask,
        low_vigor_mask=low_vigor_mask,
        alert_mask=alert_mask,
    )
    return ConditionResult(assessment, layers)
