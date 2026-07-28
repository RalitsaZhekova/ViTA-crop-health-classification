# Interactive visualization

The implemented web client displays:

- the compact RGB scene preview;
- the aligned crop-condition heat map, with all pixels outside valid clear crop
  analysis left transparent;
- selectable cells populated from the JSON interaction grid;
- exact NDVI, EVI, GNDVI and SAVI summaries from JSON;
- region history and data-quality indicators.

The scene viewer supports zoom, pan, coordinate readout, overlay opacity and
grid toggling. It has no external runtime dependency: the API, CSS, JavaScript
and compact image assets are served together. This keeps the MVP demonstrable
without internet access and prevents a third-party map service from becoming a
mission dependency.

Business-ready examples must be generated from the selected model and labelled
as internal-validation examples, not held-out evaluation or agronomic diagnosis
maps. Generated screenshots are test outputs and are not retained in source.
