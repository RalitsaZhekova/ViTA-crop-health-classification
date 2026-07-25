# Technical basis

The implementation follows the official `IPL-UV/cloudsen12_models` API and model registry.

Verified upstream facts used by this project:

- `dtacs4bands` expects `B08`, `B04`, `B03`, `B02` in that exact order;
- it was trained on Sentinel-2 L1C top-of-atmosphere reflectance;
- input reflectance values are expected between 0 and 1;
- output classes are clear, thick cloud, thin cloud and cloud shadow;
- the distributed model is TorchScript and is loaded with `load_model_by_name`;
- weights are downloaded from the official Hugging Face repository when absent.

The GEE notebook therefore exports `COPERNICUS/S2_HARMONIZED` values as unsigned integers,
and production preprocessing divides them by 10,000 before inference.

The generated softmax score is a model confidence score, not a calibrated probability.
