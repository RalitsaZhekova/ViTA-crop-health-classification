# Input/Output Specification

## Input GeoTIFF

| GeoTIFF band | Sentinel-2 band | Meaning |
|---:|---|---|
| 1 | B08 | Near Infrared |
| 2 | B04 | Red |
| 3 | B03 | Green |
| 4 | B02 | Blue |

Recommended export properties:

- Sentinel-2 L1C from `COPERNICUS/S2_HARMONIZED`;
- 10 m resolution;
- unsigned 16-bit values;
- no cloud mask applied before export;
- CRS and affine transform preserved.

## Main outputs

- `<scene>_semantic.tif`: 0 clear, 1 thick cloud, 2 thin cloud, 3 cloud shadow;
- `<scene>_unusable.tif`: 0 usable, 1 unusable;
- one class-score GeoTIFF per semantic class;
- `<scene>.json`: statistics, decision, runtime and score type;
- `<scene>_preview.png`: quick visual inspection.

`score_kind=softmax_probability` means scores were obtained from the wrapped model logits.
`score_kind=hard_one_hot` means the package exposed only its official discrete prediction and the
score maps must not be interpreted as calibrated probabilities.
