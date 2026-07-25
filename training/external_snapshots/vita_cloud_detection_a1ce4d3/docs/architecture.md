# Architecture

```text
Sentinel-2 L1C GeoTIFF
B08, B04, B03, B02
        |
        v
Input validation and DN / 10,000 normalisation
        |
        v
Overlapping 512 x 512 tiling
        |
        v
CloudSEN12 dtacs4bands pretrained inference
        |
        v
Softmax class scores when available
        |
        v
Overlap blending and scene reconstruction
        |
        v
Semantic mask: clear / thick / thin / shadow
        |
        v
Small-region removal and configurable dilation
        |
        v
Unusable-pixel mask and operational decision
        |
        v
GeoTIFF outputs, JSON metadata and preview
```

The model backend is isolated from raster I/O, tiling, post-processing and decision logic. A future
Balkan-compatible model can therefore replace CloudSEN12 without changing the team interface.
