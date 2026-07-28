# Shared crop-condition analysis contract

This stage measures crop condition after cloud masking and crop
classification. It does not train a model and does not diagnose disease,
nutrient deficiency or crop health from a single image.

## Input contract

Each processing window contains co-registered floating-point surface
reflectance arrays:

```text
BLUE, GREEN, RED, NIR
```

Raw digital numbers and display-stretched RGB are not accepted. The caller
also supplies a Boolean analysis mask assembled from:

```text
crop pixel
AND clear pixel
AND valid reflectance
AND direct crop probability >= 0.645
```

The deployed crop raster is binary and has the following exact semantics:

```text
0   = non-crop
1   = crop
255 = unusable/nodata
```

The operational unusable raster uses `0 = usable` and `1 = unusable`. The
shared mask builder used by the payload validates these values and rejects
incompatible or corrupt masks instead of treating arbitrary non-zero values as crop.

The base model does not identify crop type, so fallow fields cannot yet be
excluded automatically. Temporal baselines must therefore prevent expected
seasonal low vegetation from being presented as stress.

The `0.645` health-analysis threshold was calibrated with
`epoch=14-binary_acc=0.7949.ckpt` on the 308-chip internal validation split. It
limits false crop detections to below 10% on that split. The normal
crop/non-crop map uses `0.49`, which maximizes binary accuracy. Both thresholds
must be recalibrated if the checkpoint or validation split changes.

## Measurements

```text
NDVI  = (NIR - RED) / (NIR + RED)
EVI   = 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)
GNDVI = (NIR - GREEN) / (NIR + GREEN)
SAVI  = 1.5 * (NIR - RED) / (NIR + RED + 0.5)
CVI   = (NIR * RED) / GREEN^2
VARI  = (GREEN - RED) / (GREEN + RED - BLUE)
ExG   = 2*GREEN - RED - BLUE
```

Visible brightness is the mean of Blue, Green and Red. Unusable pixels and
unstable divisions are represented as `NaN` in raster layers and omitted from
summary statistics.

## Spectral condition assessment

The prototype score combines bounded NDVI, GNDVI, EVI and SAVI component
scores. Each valid pixel's index is linearly mapped from its configured low/high
reference to 0/100 and clipped to that range. Default weights are 40% NDVI, 25%
GNDVI, 20% EVI and 15% SAVI. CVI, VARI, excess green and RGB brightness remain
diagnostics because their absolute ranges are more dependent on canopy
structure, soil, illumination and the sensor.

The absolute region component is 70% of the median pixel score plus 30% of the
lower-quartile pixel score. The final region score subtracts a penalty of up to
20 points according to the fraction of pixels that are robustly below the
crop-region median. Median absolute deviation is used with a minimum scale and
minimum absolute deficit so tiny homogeneous-scene noise is not labeled
anomalous.

Consequently, 0 means low spectral-vigor support under the configured prototype
references and 100 means strong support. The value is not percent healthy,
survival probability, calibrated confidence or a diagnosis. It must be read
with its index components, valid-pixel coverage, evidence-quality indicator,
crop/season context and explanations.

The fixed reference ranges and label boundaries are transparent prototype
priors. They are not universal agronomic truths. The emitted labels are
`Nominal`, `Watch`, `Moderate anomaly`, `High anomaly` and `Insufficient data`.
Every result includes limitations, component evidence and a separate evidence
quality score that must not be interpreted as a calibrated probability.

Formula basis:

- Huete (1988), *A soil-adjusted vegetation index*,
  <https://doi.org/10.1016/0034-4257(88)90106-X>;
- Huete et al. (2002), MODIS vegetation-index basis,
  <https://modis.gsfc.nasa.gov/MODIS/LAND/REPORTS/huete.2002.4.pdf>;
- Gitelson and Merzlyak (1998), chlorophyll-sensitive green/NIR behavior,
  <https://doi.org/10.1016/S0273-1177(97)01133-2>;
- Vincini, Frazzi and D'Alessio (2008), broad-band CVI at canopy scale,
  <https://doi.org/10.1007/s11119-008-9075-z>.

## Outputs

The payload writes compressed, tiled GeoTIFF intermediates while processing.
Routine downlink contains an aligned WebP scene and a lossless PNG
crop-condition heat map. The heat map colors only valid clear crop pixels; all
other pixels are transparent. Cloud classes remain factual JSON quality
measurements and separate payload diagnostics, not part of this client image.
The compact JSON contains:

- scene, region, sensor and acquisition time;
- algorithm and schema versions;
- usable-analysis pixel counts and percentage;
- mean, median, standard deviation and 10th/90th percentiles;
- relative image references, checksums and an adaptive interaction grid.

The processing statuses remain `MEASURED` and `INSUFFICIENT_DATA`. A measured
scene also receives a cautious spectral screening label. The label is not a
disease diagnosis; crop-, region- and growth-stage-aware temporal baselines
remain necessary for trend and early-warning claims.

The calculations operate on one image window at a time, so the payload streams
large scenes without loading a complete image into memory.
