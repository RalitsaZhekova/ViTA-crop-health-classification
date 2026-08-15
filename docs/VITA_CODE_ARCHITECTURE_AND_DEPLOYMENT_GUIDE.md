# ViTA payload: code, science, TensorRT, and deployment guide

This document explains the working ViTA payload implementation from beginning to end. It is written for a reader who does not need to know Python, machine learning, Docker, or TensorRT beforehand.

It describes the code that is currently in this repository and the configuration that passed the four-scene, warm, under-two-second acceptance test on the Jetson Orin. It does not describe a hypothetical redesign.

## 1. The result we have now

The payload computer runs one persistent Docker container. Inside that container:

- the cloud model is executed by native TensorRT;
- the Prithvi crop model is executed by native TensorRT;
- the remaining image preparation, health-index calculations, condition scoring, and product packaging run in Python, NumPy, Rasterio/GDAL, and PyTorch CUDA support code;
- the service stays warm between jobs, so models and TensorRT execution contexts are not rebuilt for each request;
- only three final products are sent to the ground computer: `scene.json`, `scene.webp`, and `condition.png`.

The accepted warm median payload times were:

| Scene | Sensor | Median payload time |
|---|---:|---:|
| Bulgaria/Thrace | Sentinel-2 | 0.762153252 s |
| Brazil/Mato Grosso | Sentinel-2 | 0.701562425 s |
| 3370_L1ORT | Balkan-1 | 1.946894024 s |
| 3408_L1ORT | Balkan-1 | 1.829396907 s |

All 12 measured runs—three repetitions of each of the four scenes—were below two seconds. The 3370 Balkan scene is the tightest case: its maximum accepted run was about 1.957 seconds, leaving only about 43 ms of margin.

These are **payload processing times**. They include intake, cloud detection, crop inference, health/condition calculations, and downlink-product packaging. They do not include container startup, TensorRT building, warm-up, the SSH tunnel, SCP transfer, or ground-dashboard ingestion.

## 2. The simplest mental model

There are four different things that are easy to confuse:

1. **PyTorch checkpoint**: the learned model weights. This is the original scientific source of truth.
2. **ONNX graph**: a portable description of the same calculation.
3. **TensorRT plan**: a GPU-specific executable built from the ONNX graph for this exact Orin/software stack.
4. **Docker image/container**: the packaged operating environment that contains Python, CUDA libraries, the application, and all dependencies. The TensorRT plans live on a writable mounted cache, not inside the immutable image.

The Docker image is built once when code/dependencies change. TensorRT plans are built offline on the actual Jetson when the model, graph, precision, input shapes, GPU, or relevant software stack changes. Normal scene requests reuse the already accepted plans.

## 3. End-to-end data flow

The processing path is:

```text
GeoTIFF already on the payload computer
        |
        v
validate path, sensor, bands, CRS, nodata, time, and calibration
        |
        +-- Balkan-1 only: create/reuse one cached 10 m UTM analysis raster
        |
        v
cloud detection -> semantic cloud/shadow/unusable masks
        |
        v
cloud gate: continue only when cloud percentage is below 60%
        |
        v
Prithvi crop segmentation -> crop probability and binary crop mask
        |
        v
clear + confident crop pixels only
        |
        v
vegetation indices -> pixel condition scores -> regional assessment
        |
        v
scene.webp + condition.png + scene.json
        |
        v
SHA-256-verified SCP downlink -> local dashboard
```

The orchestrator is [`pipeline.py`](../payload/src/prithvi_payload/pipeline.py). The warm HTTP service is [`service.py`](../payload/src/prithvi_payload/service.py). The Windows ground command is [`Invoke-VitaPayload.ps1`](../scripts/ground/Invoke-VitaPayload.ps1).

## 4. The Prithvi crop model

### 4.1 What the model does

The crop model is a two-class segmentation model. For every image pixel it produces two numbers, called logits: one for `non_crop` and one for `crop`. A softmax converts those logits into probabilities that add to one.

The model does not calculate crop health. Its only job is to identify which usable pixels look like crop. The later health stage calculates vegetation indices only inside that crop mask.

### 4.2 Architecture in the repository

The exact model declaration is [`payload/models/architecture.yaml`](../payload/models/architecture.yaml):

```yaml
model_factory: EncoderDecoderFactory
task: segmentation
model_args:
  backbone: prithvi_eo_v2_100_tl
  backbone_pretrained: false
  backbone_num_frames: 1
  backbone_coords_encoding:
    - time
    - location
  backbone_bands:
    - BLUE
    - GREEN
    - RED
    - NIR_NARROW
  rescale: true
  necks:
    - name: SelectIndices
      indices: [2, 5, 8, 11]
    - name: ReshapeTokensToImage
      effective_time_dim: 1
  decoder: UperNetDecoder
  decoder_channels: 256
  head_dropout: 0.2
  num_classes: 2
```

In plain language:

- `prithvi_eo_v2_100_tl` is the Prithvi EO V2 100M-parameter family backbone used to understand Earth-observation imagery;
- this model uses one date, not a time series;
- it expects blue, green, red, and narrow-NIR information;
- it also receives the acquisition year/day-of-year and latitude/longitude;
- intermediate Prithvi feature maps are selected and reshaped back into image-like maps;
- UPerNet decodes those features into a full per-pixel segmentation;
- the output has two classes: non-crop and crop.

The selected model metadata is [`payload/models/selected_model.yaml`](../payload/models/selected_model.yaml). The checkpoint is 382,645,915 bytes and is pinned to this SHA-256:

```text
c948977bffdaeb89ecf4f4d069db13c7ed81d4f3403eec9e257c45c235b1484e
```

The replacement policy is deliberately manual: another checkpoint cannot silently replace this one merely because a file with the same name appears.

### 4.3 Exact input contract

The shared constants are in [`bands.py`](../shared/src/prithvi_shared/bands.py):

```python
MODEL_BANDS = ("BLUE", "GREEN", "RED", "NIR_NARROW")
TIME_STEPS = 1
INPUT_HEIGHT = 224
INPUT_WIDTH = 224

NORMALIZATION_MEANS = (
    494.905781,
    815.239594,
    924.335066,
    2968.881459,
)
NORMALIZATION_STDS = (
    284.925432,
    357.848760,
    575.566823,
    896.601013,
)
```

The image tensor shape is:

```text
[batch, 4 bands, 1 time, 224 rows, 224 columns]
```

There are also two coordinate tensors:

```text
temporal_coords: [batch, 1, 2]  -> [year, day_of_year]
location_coords: [batch, 2]     -> [latitude, longitude]
```

The model was trained on approximately 0–10000 numeric-scale reflectance. Before inference, every band is normalized as:

```text
normalized = (input_value - training_mean) / training_standard_deviation
```

The implementation is in [`inference.py`](../payload/src/prithvi_payload/inference.py):

```python
normalized = (image - self._means) / self._stds
logits = self.model(normalized, temporal_coords, location_coords)
crop_probability = logits.softmax(dim=1)[:output_batch_size, 1]
crop_binary = (crop_probability >= CROP_CLASSIFICATION_THRESHOLD).to(
    dtype=torch.uint8
)
crop_confidence = torch.maximum(crop_probability, 1 - crop_probability)
```

The live shared thresholds are in [`calibration.py`](../shared/src/prithvi_shared/calibration.py):

```python
CROP_CLASSIFICATION_THRESHOLD = 0.49
HEALTH_ANALYSIS_CROP_THRESHOLD = 0.645
```

These values match the held-out model-validation record in `selected_model.yaml`. The calibrated Balkan path maps its reflectance into the same Sentinel-equivalent model domain and therefore uses the same two probability thresholds.

### 4.4 Checkpoint loading is fail-closed

`PayloadCropModel.load()` does not trust the checkpoint filename alone. It hashes the file and refuses to start when the digest differs. The architecture is rebuilt from YAML, the weights are loaded with `strict=True`, and missing or unexpected parameters cause failure.

Relevant code in [`inference.py`](../payload/src/prithvi_payload/inference.py):

```python
actual_digest = checkpoint_sha256(checkpoint_path)
if actual_digest != SELECTED_CHECKPOINT_SHA256:
    raise RuntimeError("Selected checkpoint checksum mismatch ...")

model.load_state_dict(_model_state(checkpoint), strict=True)
```

This is a production-safety decision: a service that cannot prove which weights it loaded is not allowed to report scientific results.

### 4.5 How a large scene becomes 224 x 224 model tiles

The crop-stage plan in [`crop_stage.py`](../payload/src/prithvi_payload/crop_stage.py) sets:

```python
"tile_size": 224,
"halo": 16,
"batch_size": 16,
```

Each tile is 224 x 224. Sixteen pixels around every edge form a halo, leaving a 192 x 192 central step between neighboring tiles:

```text
core = 224 - 2 * 16 = 192
```

Overlapping predictions are not cut together with a hard seam. [`crop_executor.py`](../payload/src/prithvi_payload/crop_executor.py) builds edge-tapered weights, accumulates weighted probabilities, divides by total weight, and applies the crop threshold only after blending:

```python
probability_sum[source_slice] += tile_probability[tile_slice] * weights
probability_weight[source_slice] += weights

probability = np.divide(
    sums,
    weights,
    out=np.zeros_like(sums),
    where=has_prediction,
)
binary = (probability >= crop_threshold).astype(np.uint8)
```

This reduces tile-boundary artifacts. Fully invalid/cloud-unusable areas are excluded, and a partial final batch is padded by repeating its last tile because the accepted crop TensorRT plan has a fixed physical batch size of 16. The repeated outputs are discarded after execution.

## 5. How Balkan-1 images were made to work with Prithvi

This is not a simple file-format conversion. Prithvi was trained with a Sentinel-like spectral and numeric contract, while the Balkan product has a different sensor response, a broad NIR band, very large native rasters, and a panchromatic band. Feeding the raw Balkan numbers directly into Prithvi would satisfy the tensor shape but would not satisfy the scientific input distribution.

The adaptation has two separate parts:

1. **spatial harmonization**: create one shared 10 m analysis grid;
2. **spectral/radiometric harmonization**: map Balkan blue/green/red/broad-NIR values into the Sentinel-like values expected by Prithvi.

### 5.1 Balkan input validation

[`scene_intake.py`](../payload/src/prithvi_payload/scene_intake.py) requires a preprocessed Balkan GeoTIFF with exactly five recognized roles:

```text
BLUE, GREEN, RED, NIR_BROAD, PANCHROMATIC
```

The panchromatic band is validated but is not used by the current cloud or crop models. The code records this explicitly as `pan_used_by_current_models: False`.

The input must also have:

- a valid CRS and affine transform;
- finite bounds and positive dimensions;
- band descriptions that can be resolved to logical roles;
- a recognized NIR band;
- an acquisition time;
- a matching `*.crop_calibration.json` sidecar for crop readiness.

Without an accepted sidecar, Balkan crop readiness becomes `BALKAN_1_SPECTRAL_HARMONISATION_REQUIRED`; the system does not silently pretend broad NIR is narrow NIR.

### 5.2 One shared 10 m analysis grid

The raw Balkan scenes are roughly 10,000–13,000 pixels on each axis. Reprojecting that full raster separately in the cloud, crop, health, and display stages would be both slow and inconsistent. [`balkan_analysis.py`](../payload/src/prithvi_payload/balkan_analysis.py) therefore materializes one four-band grid that every science stage shares.

The process is:

1. select Balkan blue, green, red, and broad-NIR;
2. choose a UTM CRS appropriate for the scene bounds;
3. calculate a 10 m target grid;
4. select the largest common embedded overview that is still at or above the target resolution;
5. read that overview and reproject/resample once using average resampling;
6. write a four-band float32 GeoTIFF in the logical order `BLUE, GREEN, RED, NIR_BROAD`;
7. cache it using a key derived from source SHA-256, algorithm version, resolution, band indices, and overview strategy.

The core code is:

```python
transform, width, height = calculate_default_transform(
    source.crs,
    target_crs,
    source.width,
    source.height,
    *source.bounds,
    resolution=10.0,
)

reproject(
    source=overview_values,
    destination=analysis_values,
    src_transform=overview_transform,
    src_crs=source.crs,
    src_nodata=nodata,
    dst_transform=transform,
    dst_crs=target_crs,
    dst_nodata=nodata,
    resampling=Resampling.average,
    init_dest_nodata=True,
)
```

The deployed environment sets:

```text
VITA_BALKAN_OVERVIEW_FAST_PATH=1
VITA_BALKAN_OVERVIEW_REQUIRED=1
```

That second setting is deliberate. If the reviewed fast path is unavailable, readiness fails instead of silently falling back to a 12–13 second full-resolution warp and destroying the latency guarantee.

The cached analysis grid does **not** perform the spectral calibration itself; its metadata records `radiometry_modified: False`. This preserves a reusable sensor-space intermediate. The crop and health stages apply the verified calibration when they need Sentinel-equivalent values.

### 5.3 Separate model band routes

The analysis raster advertises two routes:

```python
"cloud_detection": {
    "expected_logical_order": ["NIR_BROAD", "RED", "GREEN", "BLUE"],
    "source_band_indices": [4, 3, 2, 1],
},
"crop_classification": {
    "expected_logical_order": ["BLUE", "GREEN", "RED", "NIR_NARROW"],
    "source_band_indices": [1, 2, 3, 4],
},
```

The cloud model naturally consumes the Balkan broad NIR along with red/green/blue in its expected channel order. The crop route labels the model expectation as narrow NIR, but it can only become valid after the spectral adapter converts the Balkan broad-NIR response to the Sentinel/Prithvi target distribution.

### 5.4 How the spectral calibration was fitted

The fitting utility is [`scripts/balkan1/calibrate_crop_input.py`](../scripts/balkan1/calibrate_crop_input.py). For each Balkan scene, it uses a geographically corresponding Sentinel reference containing B02, B03, B04, and B8A. The reference is placed on a common comparison grid.

For each of the four bands:

1. valid paired Balkan/Sentinel pixels are collected;
2. the image is divided into alternating 16 x 16 checkerboard blocks;
3. one checkerboard half is used to fit and the other half is held out for validation;
4. training pixels are sorted by the Balkan value;
5. they are divided into 128 equal-count bins;
6. the median Balkan and median Sentinel value are calculated in each bin;
7. the pool-adjacent-violators algorithm (PAVA) forces the output curve to be non-decreasing;
8. at most 65 knots are stored for compact piecewise-linear interpolation;
9. the curve is tested on the spatially held-out pixels.

The relevant fitting code is:

```python
partitions = np.array_split(order, min(bins, order.size))
source_medians = np.asarray([np.median(source[item]) for item in partitions])
target_medians = np.asarray([np.median(target[item]) for item in partitions])
fitted_target = _isotonic(weighted_target, combined_weights)
```

This creates four monotonic lookup curves:

```text
Balkan BLUE      -> Sentinel B02-like values
Balkan GREEN     -> Sentinel B03-like values
Balkan RED       -> Sentinel B04-like values
Balkan NIR_BROAD -> Sentinel B8A/NIR_NARROW-like values
```

“Monotonic” means a brighter source value is never mapped below a darker source value merely to fit noise. This is a safer adapter than an unrestricted polynomial.

The current 3370 sidecar records, for example:

- 265,081 training pixels;
- 265,257 held-out validation pixels;
- held-out band correlations of approximately 0.836, 0.861, 0.898, and 0.930;
- minimum accepted per-band correlation 0.75;
- minimum accepted mean correlation 0.80.

The calibration is not accepted unless it has at least 10,000 held-out pixels, every band correlation is at least 0.75, the mean correlation is at least 0.80, and all stored curves obey their validity rules.

### 5.5 Why the calibration sidecar cannot be reused casually

[`balkan_crop_calibration.py`](../payload/src/prithvi_payload/balkan_crop_calibration.py) validates the complete sidecar before use. It checks:

- schema and adapter version;
- sensor and source/model band orders;
- source-band indices;
- acquisition date;
- 10 m analysis resolution;
- positive numeric scale;
- four increasing source-knot arrays;
- four non-decreasing target arrays;
- held-out validation status and correlations;
- Sentinel-reference provenance;
- exact Balkan source file size and SHA-256.

Therefore, copying `3370_L1ORT.crop_calibration.json` beside another image will fail. This is intentional. Calibration is a scientific assertion tied to a specific source image and reference, not a generic color preset.

### 5.6 How calibration is applied during inference

The live adapter is [`apply_calibration()`](../payload/src/prithvi_payload/balkan_crop_calibration.py):

```python
for index in range(4):
    result[index] = np.interp(
        image[index],
        curve["source_knots"],
        curve["target_values"],
    )
```

In the compact crop path, [`crop_executor.py`](../payload/src/prithvi_payload/crop_executor.py) does:

```python
model_bands = prepared_source * np.float32(multiplier)
model_bands = apply_calibration(model_bands, calibration)
```

Those Sentinel-like 0–10000 values then go through the same training means/standard deviations as normal Sentinel input before entering Prithvi.

For health calculations, [`condition_stage.py`](../payload/src/prithvi_payload/condition_stage.py) reuses the already calibrated model bands when available and divides by 10,000 to obtain reflectance:

```python
reflectance = model_band_product[:, row_slice, column_slice] / np.float32(10_000.0)
```

This detail matters: Balkan health indices are calculated from calibrated Sentinel-equivalent reflectance, not from the unusual raw Balkan digital-number ranges.

### 5.7 What made Balkan practical within two seconds

Several changes had to work together:

- one shared cached 10 m grid instead of repeated full-resolution reprojection;
- required use of embedded overviews;
- in-memory handoff of intermediate crop/condition arrays when within configured limits;
- native TensorRT cloud profiles at the actual no-data-adjusted Balkan patch sizes, 869 and 891;
- cloud batch size 4 for multi-patch Balkan scenes;
- a fixed-batch-16 mixed-FP16 TensorRT crop plan;
- eight CPU threads for condition metrics and grid work;
- low-cost transient raster and final image compression settings;
- startup warm-up of both fixed Balkan scenes and all accepted TensorRT profiles.

The Prithvi model itself was not retrained on Balkan-1. Balkan support is a validated sensor adapter in front of the pinned Prithvi model. That preserves one crop model while making the different sensor obey its input contract.

### 5.8 Adding another Balkan image

Adding a new Balkan image is more work than adding a same-format Sentinel image because it needs its own trustworthy spectral sidecar. The production path should be:

1. preprocess the Balkan image as a five-band GeoTIFF with correct descriptions, CRS, nodata, and overviews;
2. obtain a spatially corresponding calibrated Sentinel B02/B03/B04/B8A reference;
3. fit a sidecar with `scripts/balkan1/calibrate_crop_input.py`;
4. require its held-out validation to pass;
5. place the sidecar beside the TIFF as `<stem>.crop_calibration.json`;
6. run intake/deployment checks;
7. discover the operational OmniCloudMask patch size for the full prepared scene;
8. if that size is not 869 or 891, build and scientifically qualify a new fixed TensorRT cloud profile;
9. add the scene to warm-up/acceptance if it becomes part of the fixed demo or operational contract.

A template calibration command is:

```bash
python scripts/balkan1/calibrate_crop_input.py \
  --source data/balkan1/preprocessed/NEW_SCENE.tif \
  --reference data/balkan1/reference/NEW_SCENE_sentinel2.tif \
  --output data/balkan1/preprocessed/NEW_SCENE.crop_calibration.json \
  --acquired-at 2026-03-31T12:00:00+00:00
```

Use the real acquisition time and actual reference path. Do not use this template as proof that a new scene is qualified; the output validation and full-scene parity/latency tests are the proof.

## 6. Cloud detection and the crop gate

The cloud stage uses the pinned OmniCloudMask V4 ensemble. Its exportable wrapper in [`cloud_detection/tensorrt_backend.py`](../payload/src/cloud_detection/tensorrt_backend.py) averages the logits of exactly two component models:

```python
return (self.models[0](image) + self.models[1](image)) * 0.5
```

The semantic output is converted into cloud/shadow/unusable products. Crop inference is gated by cloud percentage. [`crop_stage.py`](../payload/src/prithvi_payload/crop_stage.py) uses a strict comparison:

```text
cloud_percentage < 60.0
```

At 60% or above, crop inference is skipped. This prevents the system from reporting a crop/health assessment when most evidence is unusable.

## 7. Health calculations

### 7.1 Which pixels are allowed into health analysis

[`health.py`](../shared/src/prithvi_shared/health.py) constructs the analysis mask:

```python
mask = (crop == 1) & ~unusable
mask &= ~nodata
mask &= np.isfinite(probability) & (probability >= minimum_crop_probability)
```

A health pixel must therefore be:

- classified as crop;
- not cloud/shadow/unusable;
- not source nodata;
- associated with a finite crop probability;
- at or above the health crop-probability threshold, currently 0.645.

This prevents non-crop and bad observations from influencing the condition score.

### 7.2 Reflectance requirement

Health inputs must be floating-point calibrated reflectance, not raw integer sensor values or display-stretched RGB. Values outside the configured physical screening interval `[-0.2, 2.0]` are excluded.

For Sentinel, the stage divides the scaled raster values by its verified reflectance scale. For Balkan, it uses the calibrated Sentinel-equivalent values and divides by 10,000.

### 7.3 Calculated indices

The code calculates eight layers. Let `B`, `G`, `R`, and `N` mean blue, green, red, and near-infrared reflectance:

```text
NDVI       = (N - R) / (N + R)
GNDVI      = (N - G) / (N + G)
EVI        = 2.5 * (N - R) / (N + 6R - 7.5B + 1)
SAVI       = 1.5 * (N - R) / (N + R + 0.5)
CVI        = (N * R) / (G * G)
VARI       = (G - R) / (G + R - B)
ExcessGreen= 2G - R - B
Brightness = (B + G + R) / 3
```

The implementation uses `_safe_ratio()`, which writes `NaN` instead of dividing by a zero or nearly zero denominator. That avoids infinities contaminating regional statistics.

Only NDVI, GNDVI, EVI, and SAVI contribute to the current condition score. CVI, VARI, excess green, and brightness remain useful measurements/diagnostic features but are not scored in `spectral-condition-v1`.

## 8. From vegetation indices to one condition assessment

The transparent scoring code is [`condition.py`](../shared/src/prithvi_shared/condition.py).

### 8.1 Convert each index to 0–100

The current prototype reference ranges and weights are:

| Index | Low reference | High reference | Weight |
|---|---:|---:|---:|
| NDVI | 0.20 | 0.80 | 0.40 |
| GNDVI | 0.15 | 0.70 | 0.25 |
| EVI | 0.10 | 0.80 | 0.20 |
| SAVI | 0.15 | 0.80 | 0.15 |

Each valid index value becomes a component score:

```python
100.0 * (value - low) / (high - low)
```

and is clipped to `[0, 100]`.

The per-pixel condition score is the weighted average of all finite components. The implementation divides by the sum of weights that are actually available, so one invalid index does not automatically turn the whole pixel into zero.

### 8.2 Regional absolute-vigor score

The system calculates the median and lower quartile of valid per-pixel condition scores:

```text
absolute_vigor = 0.70 * median + 0.30 * lower_quartile
```

The median describes the typical crop pixel. The lower quartile deliberately gives some influence to the weaker part of the field without letting a handful of extreme pixels dominate.

### 8.3 Spatial anomaly detection

The code compares pixels against robust whole-region statistics:

```text
robust_scale = max(1.4826 * MAD, 5.0)
deficit      = region_median - pixel_score
robust_z     = deficit / robust_scale
```

A pixel is a relative anomaly only when both are true:

```text
robust_z >= 2.5
deficit >= 10 points
```

A pixel is also marked as absolute low vigor when:

```text
pixel_score < 35
```

MAD means median absolute deviation. It is robust because a small number of extreme values does not distort it as strongly as standard deviation.

### 8.4 Final score and label

The spatial penalty is:

```text
spatial_penalty = 20 * relative_anomaly_fraction
```

The final score is:

```text
final_score = clamp(absolute_vigor - spatial_penalty, 0, 100)
```

Labels are:

| Final score | Label |
|---:|---|
| 75–100 | Nominal |
| 55–<75 | Watch |
| 35–<55 | Moderate anomaly |
| <35 | High anomaly |

Even a nominal numeric score is promoted to `Watch` when at least 5% of analyzed pixels are in the relative-anomaly or low-vigor alert set.

### 8.5 Evidence quality

Evidence quality is a coverage indicator, not a probability that the diagnosis is correct. It averages up to three factors:

```text
min(analysis_pixels / 512, 1)
min(analysis_percentage / 5%, 1)
mean crop probability
```

The mean is multiplied by 100. Labels are `HIGH` at 75 or above, `MEDIUM` at 45 or above, and otherwise `LOW`.

The assessment becomes `INSUFFICIENT_DATA` when there are fewer than 64 analysis pixels or they cover less than 0.1% of the scene.

### 8.6 Scientific limitation

The code intentionally reports **spectral crop condition**, not an agronomic diagnosis. A single scene cannot distinguish drought, nutrient stress, pests, harvest, senescence, fallow land, or a calibration problem by itself. The label is a screening priority that needs temporal or field evidence before assigning a cause.

## 9. How the models were accelerated with native TensorRT

### 9.1 Why native TensorRT

The earlier Torch-TensorRT path attempted to convert PyTorch FX/Dynamo operations during service startup. That exposed converter-specific failures, such as TensorRT rejecting an empty constant generated by a zero-channel concatenation. It also meant a new error could appear only after a long startup compilation.

The current architecture moves compilation out of service startup:

```text
verified PyTorch model
  -> torch.export fixed graph
  -> ONNX opset 18
  -> deterministic graph canonicalization
  -> TensorRT parser check
  -> trtexec plan build on the Orin
  -> full scientific parity tests
  -> checksum-sealed accepted.json
  -> native TensorRT Python runtime
```

The online service never calls Torch-TensorRT and never compiles a plan on the first request.

### 9.2 Step 1: export the verified PyTorch source

The crop graph uses fixed inputs:

```python
image = torch.zeros((16, 4, 1, 224, 224), device="cuda")
temporal = torch.zeros((16, 1, 2), device="cuda")
location = torch.zeros((16, 2), device="cuda")
exported = torch.export.export(model, (image, temporal, location), strict=False)
```

Cloud graphs are exported for only the reviewed operational shapes:

| Use | Patch | Physical batch | Plan precision |
|---|---:|---:|---|
| Balkan scene 3408 | 869 x 869 | 4 | strongly typed FP16 |
| Balkan scene 3370 | 891 x 891 | 4 | strongly typed FP16 |
| Sentinel scenes | 1000 x 1000 | 1 | strongly typed FP32 |

The two 700 x 700 Sentinel sources are reflect-padded by the operational cloud executor to one 1000 x 1000 tile with a 150-pixel halo. A physical batch of one avoids executing three duplicate padded tiles just to satisfy the global Balkan batch setting.

The 869 and 891 sizes come from OmniCloudMask’s no-data-aware patch-size adjustment on the complete prepared Balkan grids. They are not arbitrary benchmark sizes.

### 9.3 Step 2: export ONNX

[`tensorrt_builder.py`](../payload/src/prithvi_payload/tensorrt_builder.py) exports ONNX opset 18:

```python
program = torch.onnx.export(
    exported,
    args=(),
    f=None,
    input_names=list(input_names),
    output_names=list(output_names),
    opset_version=18,
    dynamo=True,
    external_data=False,
)
```

The exporter then applies narrowly checked canonicalizations for ONNX/TensorRT compatibility. It verifies expected rewrite counts so a future model graph cannot silently receive a different rewrite.

### 9.4 The exact fix for the original zero-channel concatenation error

EdgeNeXt emits a feature with shape `[N, 0, H, W]` so its feature-pyramid interface has the same number of levels as other encoders. PyTorch treats:

```text
cat([real_tensor, zero_channel_tensor], channel_dimension)
```

as exactly the real tensor. TensorRT 10.8 rejected the zero-element constant with `INetworkDefinition::addConstant`.

[`_remove_zero_channel_cat_noops()`](../payload/src/cloud_detection/tensorrt_backend.py) removes only a two-input channel concatenation where:

- exactly one input has zero channels;
- exactly one input is non-empty;
- the non-empty input shape already equals the output shape.

It then replaces the concatenation with the live input. This is an exact identity rewrite, not an approximation or a disabled layer.

### 9.5 Step 3: parse every ONNX graph before expensive builds

The builder first asks TensorRT to parse every ONNX graph. If any operator or graph contract is invalid, deployment stops before spending time on tactic search for the remaining models.

This was added because waiting through multiple long Orin builds only to discover a parser failure in a later graph is both slow and difficult to diagnose.

### 9.6 Step 4: build plans with `trtexec`

The builder constructs arguments equivalent to:

```text
--onnx=<graph>
--noTF32
--skipInference
--memPoolSize=workspace:4096
--timingCacheFile=<profile-specific-cache>
--builderOptimizationLevel=5
```

For the crop `mixed-fp16` plan it adds:

```text
--fp16
```

This retains the FP32 ONNX input/output contract while allowing TensorRT to choose FP16 tactics internally where supported. Numerically sensitive or unsupported operations can remain FP32.

For the fixed cloud plans it adds:

```text
--stronglyTyped
```

The 869/891 graphs themselves are FP16. The 1000 graph is FP32.

The maximum builder optimization level spends more time once, offline, to search tactics. The timing cache and checksummed plan reuse avoid paying that cost on identical future deployments.

### 9.7 Why the Sentinel cloud profile is FP32

Strongly typed FP16 was fast and passed the aggregate gate, but the Brazil Sentinel scene produced too many class-boundary differences relative to the source model. We did not loosen the science gate to make the build green. Instead, only the 1000-pixel Sentinel profile was promoted to FP32.

The final reviewed split is therefore:

- Balkan multi-patch cloud inference: FP16 for speed;
- Sentinel single-tile cloud inference: FP32 for parity;
- crop inference: FP32 I/O with mixed internal FP16 tactics.

### 9.8 Step 5: parity qualification

Building a plan only proves TensorRT can execute a graph. It does not prove the output is scientifically close enough.

The crop plan is tested on a balanced batch of real-scene tiles from all four scenes. Acceptance limits are:

```text
crop decision mismatch fraction <= 0.002
mean absolute crop-probability error <= 0.005
```

For mixed FP16, the reference is the established PyTorch CUDA-autocast-FP16 execution, because that is the operational numerical baseline the mixed plan replaces.

The cloud plans are evaluated on the complete prepared outputs for all four qualification scenes. Acceptance limits are unchanged:

```text
aggregate semantic class mismatch <= 0.001
each individual scene mismatch     <= 0.002
```

Only after all gates pass does the builder write an accepted manifest.

### 9.9 Step 6: seal artifacts

The accepted manifest records:

- status `accepted`;
- target GPU/software signature;
- source checkpoint/ensemble hashes;
- ONNX and plan hashes;
- I/O names, shapes, and dtypes;
- precision and batch/profile contracts;
- parity reports;
- builder and package versions.

The whole manifest receives a deterministic SHA-256. It is written atomically, so a failed build cannot leave a half-written accepted state. Candidate plans remain inert until the final manifest is accepted. Unaccepted artifacts are pruned after sealing.

### 9.10 Step 7: native execution

[`tensorrt_runtime.py`](../payload/src/prithvi_payload/tensorrt_runtime.py) loads TensorRT directly through its Python API. Before deserializing a plan it verifies:

- the manifest seal;
- `status == accepted`;
- exact current target signature;
- plan SHA-256;
- I/O names, shapes, dtypes, and input/output roles.

Execution uses the existing PyTorch CUDA tensors without a CPU round-trip:

```python
self._context.set_tensor_address(spec["name"], value.data_ptr())
stream = torch.cuda.current_stream(self.device)
self._context.execute_async_v3(stream_handle=stream.cuda_stream)
```

The cloud router selects a plan by patch height/width, enforces the logical batch range, converts to the plan’s reviewed input dtype, pads a short batch to the fixed physical batch, runs the plan, and discards padded outputs.

### 9.11 What is not being used

The accepted configuration does not use:

- Torch-TensorRT;
- Jetson DLA engines;
- INT8 quantization;
- NVIDIA Model Optimizer quantization;
- CUDA graphs;
- TF32.

TensorRT executes on the Orin GPU. DLA could be a later independent qualification project, but it is not part of the working under-two-second result.

## 10. Warm service architecture

[`PayloadRuntime`](../payload/src/prithvi_payload/service.py) is constructed once during FastAPI lifespan startup. It:

- loads the crop and cloud runtimes;
- prepares/caches the two fixed Balkan 10 m grids;
- prepares Sentinel warm-up inputs;
- warms every accepted cloud shape and both crop execution paths;
- keeps TensorRT engines and execution contexts resident;
- records acceleration and parity metadata returned in each result.

The service runs model construction, warm-up, and jobs on one persistent AnyIO worker because CUDA/cuDNN setup includes thread-local state. A non-blocking `threading.Lock` permits only one payload job at a time. A concurrent request receives a busy response instead of competing for GPU memory and invalidating latency expectations.

This is why a real demo must keep the container running. Starting a fresh process per image would include model loading and warm-up and would not represent the accepted SLO.

## 11. Docker architecture

### 11.1 Base image and installed application

[`deploy/Dockerfile.payload`](../deploy/Dockerfile.payload) starts from:

```dockerfile
ARG VITA_PAYLOAD_BASE_IMAGE=nvcr.io/nvidia/pytorch:25.01-py3-igpu
FROM ${VITA_PAYLOAD_BASE_IMAGE}
```

The accepted target reported:

```text
PyTorch 2.6.0a0+ecf3bae40a.nv25.01
CUDA 12.8
TensorRT 10.8.0.40
ONNX 1.17.0
ONNXScript 0.1.0
GPU Orin
```

The Dockerfile installs system packages such as GDAL and `tini`, installs pinned Python requirements, copies the repository packages, and installs the application. It creates a non-root `vita-payload` identity with the configured UID/GID (2002 by default), verifies that identity at build time, and launches:

```dockerfile
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["vita-payload-server"]
```

`tini` is a small init process that forwards signals and reaps child processes correctly.

### 11.2 Container security and storage

[`compose.payload.yaml`](../deploy/compose.payload.yaml) applies:

- non-root UID/GID;
- read-only root filesystem;
- all Linux capabilities dropped;
- `no-new-privileges`;
- loopback-only host port 8090;
- read-only data and model mounts;
- only two owned writable bind mounts;
- temporary filesystems for `/tmp` and `/home/vita`;
- a health check;
- `restart: unless-stopped`.

Mounts are conceptually:

```text
repository/data           -> /data          read-only
repository/payload/models -> /models        read-only
repository/runtime/payload-> /runtime       writable
repository/runtime/engines-> /engine-cache  writable
```

Keeping data/models read-only prevents an inference bug from modifying source imagery or learned weights. Runtime products and TensorRT artifacts have explicit writable locations.

### 11.3 Image build versus engine build

These are separate operations:

```text
docker compose build
```

creates/replaces the application image.

```text
python -m prithvi_payload.tensorrt_builder
```

creates accepted GPU plans in `/engine-cache`, which is mounted from `runtime/engines` on the Jetson.

Deleting only a container does not delete the image or engine cache. Rebuilding an image does not automatically make old engine plans valid; the manifest checks source and target identity.

## 12. Exact no-sudo deployment on the payload computer

Run these commands on the Jetson from the repository checkout:

```bash
cd /data/code/VITA
chmod +x deploy/payload/*.sh
./deploy/payload/probe.sh
./deploy/payload/preflight.sh
./deploy/payload/deploy.sh
```

No `sudo` command is required. Docker access must already be available to the logged-in account.

On the first deployment only, if `deploy/payload.env` does not exist:

```bash
cd /data/code/VITA
test -f deploy/payload.env || cp deploy/payload.env.example deploy/payload.env
```

Do not copy the example over a working `deploy/payload.env`; doing so can erase host-specific paths or settings.

The key accepted settings are:

```text
VITA_CROP_BACKEND=tensorrt
VITA_CROP_BATCH_SIZE=16
VITA_CROP_TRT_PRECISION=mixed-fp16
VITA_TRT_CUDAGRAPHS=0

VITA_CLOUD_BACKEND=tensorrt
VITA_CLOUD_INFERENCE_DTYPE=fp16
VITA_CLOUD_TRT_PRECISION=fp16
VITA_CLOUD_SENTINEL_TRT_PRECISION=fp32
VITA_CLOUD_BATCH_SIZE=4
VITA_CLOUD_WARMUP_PATCH_SIZES=869,891,1000

VITA_TRT_WORKSPACE_MIB=4096
VITA_TRT_BUILDER_OPTIMIZATION_LEVEL=5
VITA_WARMUP=1
VITA_SKIP_PERFORMANCE_ACCEPTANCE=0
```

### 12.1 What `probe.sh` does

The probe checks the host/GPU prerequisites before an expensive build. It is the earliest place to catch an unsupported host, missing Docker/NVIDIA runtime, wrong architecture, or missing target capability.

### 12.2 What `preflight.sh` does

Preflight checks repository state, required source data/model/calibration files, writable VITA-owned directories, environment settings, space expectations, and deployment assumptions. It should fail before image or TensorRT construction when a prerequisite is missing.

### 12.3 What `deploy.sh` does

The deployment script is the authoritative sequence. In simplified order it:

1. loads/validates the deployment environment;
2. removes rejected VITA TensorRT caches and stale acceptance-run output only within their validated VITA directories;
3. builds the complete image when none exists, or a small code overlay when the large dependency image already exists;
4. validates CUDA, ONNX export, direct TensorRT imports, and `trtexec` inside the final image;
5. validates the two writable mounts;
6. validates the two Sentinel and two Balkan inputs and sidecars;
7. stops the existing payload service before exclusive tactic search;
8. builds/reuses TensorRT candidates offline;
9. runs crop and complete-scene cloud parity gates;
10. atomically accepts the plans;
11. starts the service;
12. waits for Docker health;
13. verifies deployment/acceleration metadata;
14. runs three repetitions of all four scenes;
15. keeps the service only if scientific and two-second acceptance pass.

When `VITA_REPLACE_TRT_ARTIFACTS=1` is intentionally used for a space-constrained engine migration, the script stops/removes the old service container and deletes only the old sealed direct-TensorRT artifact set before constructing the replacement. This avoids holding old and new engine sets simultaneously. It does not delete source imagery, model weights, the Docker base image, or unrelated Docker data.

### 12.4 Manual equivalent, for understanding

Normally use `deploy.sh`, because it contains safety checks and cleanup. The central Docker operations it orchestrates are equivalent to:

```bash
cd /data/code/VITA

docker compose --env-file deploy/payload.env \
  -f deploy/compose.payload.yaml build

docker compose --env-file deploy/payload.env \
  -f deploy/compose.payload.yaml run --rm --no-deps \
  --entrypoint python payload -m prithvi_payload.tensorrt_builder

docker compose --env-file deploy/payload.env \
  -f deploy/compose.payload.yaml up -d

docker compose --env-file deploy/payload.env \
  -f deploy/compose.payload.yaml exec -T payload \
  python -m prithvi_payload.deployment_acceptance

docker compose --env-file deploy/payload.env \
  -f deploy/compose.payload.yaml exec -T payload \
  python -m prithvi_payload.performance_acceptance
```

This manual list is explanatory, not a recommendation to bypass `deploy.sh`.

## 13. Starting, proving, and stopping the accepted service

### Start an existing container without rebuilding

On the Jetson:

```bash
cd /data/code/VITA
docker compose --env-file deploy/payload.env \
  -f deploy/compose.payload.yaml start payload
curl --fail http://127.0.0.1:8090/healthz
```

If the container no longer exists but the image and accepted engine cache do:

```bash
cd /data/code/VITA
docker compose --env-file deploy/payload.env \
  -f deploy/compose.payload.yaml up -d payload
curl --fail http://127.0.0.1:8090/healthz
```

### Prove which container and acceleration are running

```bash
cd /data/code/VITA
docker compose --env-file deploy/payload.env \
  -f deploy/compose.payload.yaml ps payload

docker inspect --format '{{.State.Status}} {{.State.Health.Status}} {{.Config.Image}}' \
  "$(docker compose --env-file deploy/payload.env -f deploy/compose.payload.yaml ps -q payload)"

curl --fail http://127.0.0.1:8090/healthz
```

The deployment-acceptance response should report `crop_backend: tensorrt`, a crop engine count of 1, `cloud_backend: omnicloudmask_tensorrt_fp16`, and three cloud profiles/plans.

### Stop cleanly without deleting the image or engines

```bash
cd /data/code/VITA
docker compose --env-file deploy/payload.env \
  -f deploy/compose.payload.yaml stop payload
```

This is the correct end-of-demo command when the next demo should restart quickly. It preserves the stopped container, Docker image, warmed configuration, and accepted engine cache. Warm-up runs again when the process starts, but TensorRT is not rebuilt.

## 14. Running the full pipeline from the ground computer

The image is not uploaded. The PowerShell script sends only job metadata through an SSH tunnel. The payload service reads the already-present payload-relative path, processes it, and the ground script downloads only the three products with SHA-256 verification.

The parameter is `-PayloadInput`, not `-Input`.

### Sentinel full demo

From Windows PowerShell:

```powershell
Set-Location D:\ML_ComputerVision\prithvi_crop_head_starter
$Payload = 'space-challenges@10.11.250.25'

.\scripts\ground\Invoke-VitaPayload.ps1 `
  -SshTarget $Payload `
  -Sensor sentinel-2 `
  -PayloadInput sentinel2 `
  -Image S2_20260610T091331_T35TLG_bulgaria-thrace.tif `
  -RegionId sentinel-bulgaria-thrace-fulldemo
```

### Balkan full demo

```powershell
Set-Location D:\ML_ComputerVision\prithvi_crop_head_starter
$Payload = 'space-challenges@10.11.250.25'

.\scripts\ground\Invoke-VitaPayload.ps1 `
  -SshTarget $Payload `
  -Sensor balkan-1 `
  -PayloadInput balkan1/preprocessed/3370_L1ORT.tif `
  -RegionId balkan-3370-fulldemo
```

The ground script:

1. validates IDs and payload-relative paths;
2. opens an SSH local-forward tunnel to Jetson loopback port 8090;
3. polls `/healthz`;
4. posts the job request;
5. downloads `scene.json`, `scene.webp`, and `condition.png` with SCP;
6. checks each SHA-256 against the service response;
7. starts/reuses the local ground dashboard unless `-SkipDashboard` is specified;
8. ingests the completed bundle and prints the result.

The dashboard is normally available at:

```text
http://127.0.0.1:8000/
```

## 15. Why the architecture is arranged this way

| Decision | Reason | Consequence |
|---|---|---|
| Fixed crop batch 16 | Efficient stable Orin execution | Partial batches are padded and trimmed |
| Fixed cloud shapes | TensorRT is fastest and most predictable for reviewed operational shapes | A genuinely new patch size needs a new qualified plan |
| FP16 Balkan cloud, FP32 Sentinel cloud | Preserve Balkan speed while keeping Brazil Sentinel inside parity | Three cloud plans instead of one generic plan |
| Mixed-FP16 crop with FP32 I/O | Keep the established numeric interface while accelerating safe layers | Some layers remain FP32 |
| TF32 disabled | Avoid an additional unqualified numeric mode | Slightly less possible speed, clearer parity |
| Native TensorRT | Compile offline and remove Torch-TensorRT converter/runtime dependency | More explicit build/runtime code |
| SHA-sealed artifacts | Prevent stale/wrong engines from running | Engine plans are target-specific |
| Balkan 10 m cached grid | Avoid repeated enormous reprojections | Requires prewarming/cache preparation |
| Scene-bound Balkan calibration | Prevent scientifically invalid cross-scene reuse | New Balkan scenes need qualification work |
| In-memory compact handoffs | Avoid writing products that are never downlinked | Bounded by explicit memory limits |
| One warm worker and job lock | Preserve CUDA state and latency | One payload job at a time |
| Loopback service plus SSH tunnel | No exposed payload HTTP service | Ground orchestration requires SSH access |
| Three-file downlink | Small, deterministic, verifiable demo package | Full intermediate rasters remain payload-local/transient |

## 16. Tuning thresholds safely

There are different kinds of threshold, and changing one does not mean the same thing as changing another:

- crop classification threshold (`0.49`): controls which pixels become crop;
- health crop threshold (`0.645`): controls crop-probability evidence admitted to health analysis;
- cloud gate (`60%`): controls whether crop/health processing runs at all;
- index reference ranges: translate NDVI/GNDVI/EVI/SAVI into 0–100 component scores;
- anomaly thresholds: control relative and absolute alert masks;
- label thresholds: map the final score to Nominal/Watch/Moderate/High.

A threshold change normally does not require rebuilding TensorRT because thresholds are applied after model logits/probabilities. It does require regression tests and scientific validation because it changes reported decisions. If the threshold is moved into an exported graph later, that would change the ONNX/plan identity and would require rebuilding.

The clean long-term design would put reviewed operational thresholds in one versioned configuration/provenance record and include that version in `scene.json`. The current live values are centralized in `calibration.py` and match the selected-model validation record.

## 17. Main source files to read

Read these in roughly this order:

1. [`payload/src/prithvi_payload/pipeline.py`](../payload/src/prithvi_payload/pipeline.py) — stage orchestration.
2. [`payload/src/prithvi_payload/service.py`](../payload/src/prithvi_payload/service.py) — warm service, health, locking, and response.
3. [`payload/src/prithvi_payload/scene_intake.py`](../payload/src/prithvi_payload/scene_intake.py) — file/sensor/band/calibration validation.
4. [`payload/src/prithvi_payload/balkan_analysis.py`](../payload/src/prithvi_payload/balkan_analysis.py) — cached Balkan 10 m grid.
5. [`payload/src/prithvi_payload/balkan_crop_calibration.py`](../payload/src/prithvi_payload/balkan_crop_calibration.py) — calibration trust rules and interpolation.
6. [`scripts/balkan1/calibrate_crop_input.py`](../scripts/balkan1/calibrate_crop_input.py) — fitting the Balkan-to-Sentinel curves.
7. [`payload/src/prithvi_payload/cloud_executor.py`](../payload/src/prithvi_payload/cloud_executor.py) — cloud image preparation/execution.
8. [`payload/src/prithvi_payload/crop_stage.py`](../payload/src/prithvi_payload/crop_stage.py) — crop plan, gate, routes, and thresholds.
9. [`payload/src/prithvi_payload/crop_executor.py`](../payload/src/prithvi_payload/crop_executor.py) — tiling, calibration, batches, blending, and crop products.
10. [`payload/src/prithvi_payload/inference.py`](../payload/src/prithvi_payload/inference.py) — checkpoint loading, normalization, and crop probabilities.
11. [`shared/src/prithvi_shared/health.py`](../shared/src/prithvi_shared/health.py) — analysis mask and vegetation indices.
12. [`shared/src/prithvi_shared/condition.py`](../shared/src/prithvi_shared/condition.py) — scoring and labels.
13. [`payload/src/prithvi_payload/tensorrt_builder.py`](../payload/src/prithvi_payload/tensorrt_builder.py) — ONNX export, TensorRT construction, parity, and acceptance.
14. [`payload/src/prithvi_payload/tensorrt_runtime.py`](../payload/src/prithvi_payload/tensorrt_runtime.py) — native engine validation/execution.
15. [`payload/src/cloud_detection/tensorrt_backend.py`](../payload/src/cloud_detection/tensorrt_backend.py) — cloud shape/precision routing and zero-channel graph fix.
16. [`deploy/Dockerfile.payload`](../deploy/Dockerfile.payload) — payload image.
17. [`deploy/compose.payload.yaml`](../deploy/compose.payload.yaml) — runtime configuration, security, and mounts.
18. [`deploy/payload/deploy.sh`](../deploy/payload/deploy.sh) — authoritative Jetson deployment sequence.
19. [`scripts/ground/Invoke-VitaPayload.ps1`](../scripts/ground/Invoke-VitaPayload.ps1) — complete ground-to-payload demo.

## 18. Final concise summary

Prithvi receives one 224 x 224, four-band, one-date tile at a time, plus time and location, and outputs crop probability. Sentinel already resembles its expected sensor contract; Balkan is first placed on a cached 10 m UTM grid and passed through a source-bound, held-out-validated monotonic mapping learned against corresponding Sentinel B02/B03/B04/B8A data.

Both the crop model and the two-model cloud ensemble are exported through ONNX and built into native, target-specific TensorRT plans on the Orin. Plans are accepted only after checksum, target, I/O, full-scene semantic parity, crop probability/decision parity, startup, and four-scene latency gates pass; the warm container then reuses those plans for the full downlink-producing pipeline.
