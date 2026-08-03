# Technical basis

The implementation follows the frozen ViTA Cloud Detector integration and
OmniCloudMask `1.7.1` API.

Verified upstream facts used by this project:

- the model expects `Red`, `Green`, `NIR` in that order;
- output classes are clear, thick cloud, thin cloud and cloud shadow;
- dynamic per-patch z-score normalization enables cross-sensor value ranges;
- V4 uses two U-Net ensemble members with soft voting;
- the reviewed spatial range is 5-50 m and 10 m is recommended over finer
  imagery when extra edge detail is not required;
- Sentinel-2 loading defaults to B04, B03 and B8A;
- the supplied Balkan adapter maps L1ORT bands 3, 2 and 4 and reprojects them
  independently to 10 m UTM before inference;
- the frozen baseline uses FP32, 1000 px patches and 300 px overlap.

The payload retains Blue only for visualization and later stages. It preserves
the original four-band cloud-stage route, reorders to the native three-channel
model input inside the backend, and normalizes returned confidence channels to
sum to one without changing their winning class.
