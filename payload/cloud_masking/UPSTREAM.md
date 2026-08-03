# Cloud pipeline provenance

The selected integration was reviewed from:

- repository: `https://github.com/Gab1604/Vita-CloudDetector`;
- commit: `a991819bf4462238b12dd0191e5dfaa35b69f39a`;
- package version: `vita-cloud-detector 0.1.0`;
- frozen model package: `omnicloudmask==1.7.1`;
- OmniCloudMask reference commit:
  `fbc6d3f5665eb3425fb2474cb3e6f574e2e71a1b`;
- model: V4 two-member ensemble;
- weights registry: `NickWright/OmniCloudMask` on Hugging Face.

The payload reuses the supplied repository's band mappings, validity rule,
10 m Balkan projection, V4 settings and class contract. Only the adapter around
those rules is local because this payload already owns windowed GeoTIFF I/O,
post-processing, crop masking, metadata and downlink interfaces.

Both V4 component hashes and a deterministic ensemble fingerprint are pinned in
`payload/models/omnicloudmask/model.yaml`. Runtime loading is offline-only:
missing or changed files fail startup rather than downloading silently.

## Licensing

The ViTA integration repository is MIT licensed. OmniCloudMask `1.7.1` is MIT
licensed, and the official Hugging Face model repository declares the V4
weights MIT. Preserve upstream notices and attribution when distributing the
software or model files.

## Validation boundary

The supplied zero-shot benchmark uses original Balkan-1 L1ORT products
resampled to 10 m UTM. It reports labelled-scene F1 values from 0.6932 to 0.9824
and very low false positives on clear scenes. Scene 3408 is reported as
99.7709% clear; this integration reproduced 99.7720% clear on the same complete
L1ORT image.

That evidence validates the cloud model transfer, not the separate Prithvi crop
model transfer. Sparse-label and snow/ice caveats documented by the supplied
repository still apply, and mission-specific monitoring remains required.
