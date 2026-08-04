# Balkan-1 reconstruction boundary

The local workflow converts raw, separately delivered Balkan-1 bands into one
co-registered minimum L1A scene with `scripts/balkan1/process_l1a.py`. It uses
the delivered imagery and navigation evidence and does not invent calibration
or orbit data. Production L1B/L1C still requires the missing mission
calibration, terrain and atmospheric inputs documented below.

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

Local imagery stays under ignored `data/balkan1/` storage or at an external
path. Production workflow code is checked in under `scripts/balkan1/`, while
the optional `scripts/balkan1/preprocessors/` extension point is reserved for
additional real mission-backed implementations. The launchers pass a completed
GeoTIFF to the existing payload pipeline without copying source imagery into
`payload/`. Existing L1ORT products need an explicit reviewed band order because
their GeoTIFF descriptions are empty.
