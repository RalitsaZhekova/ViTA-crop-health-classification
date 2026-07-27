# Health analysis contract

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
ground mask builder validates these values and rejects incompatible or corrupt
masks instead of treating arbitrary non-zero values as crop.

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
scores. NDVI has the largest weight; GNDVI, EVI and SAVI provide supporting
canopy, chlorophyll-related and soil-adjusted evidence. CVI, VARI, excess green
and RGB brightness remain diagnostics because their absolute ranges are more
dependent on canopy structure, soil, illumination and the sensor.

The region score uses the median and lower quartile of the pixel scores, then
applies a limited penalty for pixels that are robustly below the crop-region
median. Median absolute deviation is used with a minimum scale and minimum
absolute deficit so tiny homogeneous-scene noise is not labeled anomalous.

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

Per-pixel layers will later be written as Cloud-Optimized GeoTIFFs. Compact
JSON observations contain:

- scene, region, sensor and acquisition time;
- algorithm and schema versions;
- usable-analysis pixel counts and percentage;
- mean, median, standard deviation and 10th/90th percentiles;
- references to raster assets.

The processing statuses remain `MEASURED` and `INSUFFICIENT_DATA`. A measured
scene also receives a cautious spectral screening label. The label is not a
disease diagnosis; crop-, region- and growth-stage-aware temporal baselines
remain necessary for trend and early-warning claims.

The calculations operate on one image window at a time, allowing the eventual
GeoTIFF pipeline to stream large scenes without loading a complete image into
memory.
