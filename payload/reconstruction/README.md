# Balkan-1 reconstruction boundary

This component will convert raw, separately delivered Balkan-1 bands into one
co-registered scene. It is not implemented until real raw imagery and sensor
metadata are available.

## Required input

- one raster per source band;
- acquisition identifier and time;
- image dimensions, pixel depth and no-data value;
- spacecraft position/attitude or geolocation metadata;
- available camera calibration and distortion metadata;
- scene coordinates;
- optional processed reference product for comparison.

## Required output

- co-registered `float32` BLUE, GREEN, RED and NIR arrays;
- optional registered panchromatic array;
- CRS, affine transform, bounds and ground sampling distance;
- a no-data mask;
- per-band registration error and quality flags;
- a record of every transform applied.

Registration must be estimated from the data and metadata. Unknown altitude,
rotation or band offsets must never be replaced with assumed constants.

The output is handed to radiometric calibration, cloud masking and tiled model
inference. Health calculations do not run on the payload reconstruction stage.
