# Model Card — OmniCloudMask V4

## Purpose

Sensor-agnostic semantic cloud and cloud-shadow detection for Sentinel-2 and
Balkan-1 reflectance products.

## Input contract

Native channel order is `Red`, `Green`, `NIR`. OmniCloudMask applies dynamic
per-patch, per-channel z-score normalization. The reviewed spatial range is
5-50 m for V4, with 10 m used for Balkan-1 and Sentinel-2 in this project.

## Output classes

| Value | Meaning |
|---:|---|
| 0 | Clear |
| 1 | Thick cloud |
| 2 | Thin cloud |
| 3 | Cloud shadow |

## Selected model

- package: `omnicloudmask==1.7.1`;
- model version: V4;
- ensemble members: RegNetY-004 and EdgeNeXt Small U-Net models;
- ensemble SHA-256:
  `ab8f039866d6714b249f850779b9523b5f6afb55ee891077a71db0fdebc9b529`;
- inference: FP32, CUDA when available;
- patch size / overlap: 1000 / 300.

## Limitations

- snow/ice, haze, bright surfaces, thin-cloud boundaries and shadows remain
  difficult cases;
- Balkan validation applies to L1ORT resampled to 10 m, not native 1.5 m model
  inference;
- confidence channels are model scores, not calibrated probabilities;
- cloud validation does not validate the separate Balkan crop transfer;
- downstream crop analysis must exclude all operationally unusable pixels.

## License

Code and official V4 weights: MIT. Preserve upstream attribution and notices.
