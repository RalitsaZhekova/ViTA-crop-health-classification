# Cloud-mask boundary

The reviewed cloud-detection project remains separate and has not been merged
or copied here. This preserves the earlier decision not to mix that work into
the crop repository prematurely.

Reviewed source:
`https://github.com/Gab1604/ViTA-SpaceChallenges2026/tree/feature/cloud-detection-reviewed/phase1/cloud_detection`

## Required adapter contract

Input:

- co-registered, radiometrically calibrated Balkan-1 or Sentinel-2 bands;
- sensor identifier;
- no-data mask and geospatial metadata.

Output:

- `cloud_unusable`: Boolean raster aligned exactly to the model output;
- optional cloud probability;
- quality metadata including algorithm version and sensor calibration;
- explicit shadow handling status.

Before Balkan-1 use, the existing detector must be validated for its 1.5 m
resolution, spectral response and 12-bit scale. A Sentinel-trained detector
must not be declared Balkan-1 compatible solely because both sensors expose
RGB and NIR.
