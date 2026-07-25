# Model Card — CloudSEN12 dtacs4bands

## Purpose

Semantic cloud and cloud-shadow detection for Sentinel-2 L1C imagery using four spectral bands.

## Input contract

Exact channel order: `B08`, `B04`, `B03`, `B02`.
Values are top-of-atmosphere reflectance, normally obtained by dividing Sentinel-2 L1C digital
numbers by 10,000.

## Output classes

| Value | Meaning |
|---:|---|
| 0 | Clear |
| 1 | Thick cloud |
| 2 | Thin cloud |
| 3 | Cloud shadow |

## Model source

Loaded with `cloudsen12_models==1.0.2` using model name `dtacs4bands`.
The checkpoint SHA-256 expected by this project is:

```text
37205adce72fbbb65a3cfa8f47676c84ebf9b1555a27a3838d584072c954b22d
```

## Limitations

- trained for Sentinel-2 L1C, not future Balkan/EnduroSat imagery;
- bright surfaces, haze, snow, thin cloud, cloud boundaries and shadows remain difficult;
- output quality must be measured on expert-labelled validation data;
- downstream crop analysis must not use pixels marked unusable.

## Licence

Checkpoint: CC BY-NC 4.0. Code package: LGPL-3.0.
