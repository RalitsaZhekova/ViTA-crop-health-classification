# Shared contracts

`shared/` contains small, sensor-independent definitions used by training,
payload inference and ground processing.

It owns:

- the four-band model input order and three-time-step shape;
- the exact training normalization values;
- the 13 fine classes and crop/no-crop mappings;
- the selected checkpoint identity and calibrated thresholds;
- JSON schemas exchanged between payload and ground.

It contains no model weights, satellite reconstruction, cloud detection,
health interpretation, database code or UI code.

The training normalization values apply to the numeric scale used by the
IBM-NASA training chips. Balkan-1 12-bit raw digital numbers must first be
radiometrically calibrated and mapped to an equivalent reflectance scale.
Applying these values directly to uncalibrated Balkan-1 DN is invalid.
