# Cloud pipeline provenance

The payload cloud module was integrated from:

- repository: `https://github.com/Gab1604/ViTA-SpaceChallenges2026`;
- branch: `feature/cloud-detection-reviewed`;
- source directory: `phase1/cloud_detection`;
- reviewed commit: `a1ce4d3612ea60a39da679a0dc9db52cb59c9800`.

Its runtime package is integrated under `payload/src/cloud_detection/`;
configs, docs and the batch inference script are under
`payload/cloud_detection/`. The reviewed commit and Git history provide the
recovery point without retaining a second source tree.

The integrated runtime makes four deployment-specific changes:

- require and verify the pinned checkpoint before inference;
- resolve the weight directory relative to its configuration;
- render previews through a headless backend;
- reject partially or incorrectly described band stacks.

`PayloadCloudClassifier` remains the classifier-only public API. The complete
`CloudDetectionPipeline` additionally provides GeoTIFF I/O, four-class score
rasters, post-processing, previews, metadata and an unusable-pixel mask.

## Licensing gate

`cloudsen12_models==1.0.2` reports LGPL-3.0 for its software package. The
reviewed model card identifies the distributed `dtacs4bands` checkpoint as
CC BY-NC 4.0. That checkpoint is acceptable for this non-commercial prototype,
but it is not approved for the planned commercial/B2B service. Obtain separate
commercial permission or replace the checkpoint before production.

The reviewed ViTA branch does not contain a root or module-level `LICENSE`
file. Confirm the integration-code licence with the repository owner before
external distribution.

## Validation boundary

The reviewed project has no reported quantitative evaluation on its
expert-labelled holdout. The retained verification record confirms that all
semantic classes and the real checkpoint ran on CPU and CUDA. Runtime success
does not establish accuracy on Sentinel-2 or Balkan-1 imagery.
