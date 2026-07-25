# Evaluation Plan

## Quantitative benchmark

Use held-out expert-labelled CloudSEN12 patches with the same four L1C bands.

Report separately:

- thick-cloud precision, recall, F1 and IoU;
- thin-cloud precision, recall, F1 and IoU;
- cloud-shadow precision, recall, F1 and IoU;
- binary cloud metrics after merging thick and thin cloud;
- binary unusable-pixel metrics after including shadow;
- confusion matrix.

## Operational testing

Run the model on geographically diverse Google Earth Engine exports and inspect:

- agricultural areas;
- bright bare soil;
- cities;
- water;
- haze;
- cloud edges;
- thin clouds;
- cloud shadows;
- snow if relevant.

## Onboard benchmark

Measure:

- checkpoint and container size;
- runtime per tile and per scene;
- peak resident memory;
- data rejected or retained after masking;
- repeatability across runs.

Do not report accuracy before expert-labelled samples have been evaluated.
