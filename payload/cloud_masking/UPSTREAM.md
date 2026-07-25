# Cloud pipeline provenance

The payload cloud module was integrated from:

- repository: `https://github.com/Gab1604/ViTA-SpaceChallenges2026`;
- branch: `feature/cloud-detection-reviewed`;
- source directory: `phase1/cloud_detection`;
- reviewed commit: `a1ce4d3612ea60a39da679a0dc9db52cb59c9800`.

The complete reviewed source is preserved under
`training/external_snapshots/vita_cloud_detection_a1ce4d3/`. Its runtime
package is integrated under `payload/src/cloud_detection/`; configs, docs and
operational scripts are under `payload/cloud_detection/`; its tests are under
`tests/cloud_detection_upstream/`.

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

The reviewed project has deterministic unit tests but no reported quantitative
evaluation on its expert-labelled holdout. The integrated tests cover every
semantic class and the real checkpoint runs on CPU and CUDA. Passing those
tests proves that the component runs; it does not establish accuracy on
Sentinel-2 or Balkan-1 imagery.
