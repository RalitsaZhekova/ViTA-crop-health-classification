# Shared contracts

`shared/` contains small, sensor-independent definitions used by training,
payload inference and ground processing.

It owns:

- the four-band, single-image model input shape;
- the exact training normalization values;
- the preserved 13 fine classes and active crop/no-crop contract;
- the selected checkpoint identity and calibrated thresholds;
- sensor-neutral vegetation indices, RGB diagnostics and transparent condition scoring;
- JSON schemas exchanged between payload and ground.

It contains no model weights, satellite reconstruction, cloud detection,
scene orchestration, database code or UI code.

The training normalization values apply to the numeric scale used by the
IBM-NASA training chips. Balkan-1 12-bit raw digital numbers must first be
radiometrically calibrated and mapped to an equivalent reflectance scale.
Applying these values directly to uncalibrated Balkan-1 DN is invalid.
