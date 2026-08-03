# Architecture

```text
Sentinel-2 B8A/B04/B03/B02     Balkan-1 NIR/RED/GREEN/BLUE L1ORT
              |                              |
              |                     independent reprojection
              |                         to 10 m UTM
              +---------------+--------------+
                              |
                              v
                 Red / Green / NIR adapter
                              |
                              v
             OmniCloudMask 1.7.1, V4 ensemble
                1000 px patches, 300 px overlap
                              |
                              v
             clear / thick / thin / shadow mask
                              |
                              v
              existing filtering and dilation
                              |
                              v
       source-grid semantic / invalid / unusable masks
                              |
                              v
          unchanged crop, condition and downlink stages
```

The backend remains isolated from scene intake, post-processing, decisions and
downlink packaging. Balkan source pixels are never rewritten; only the cloud
analysis grid is temporary, and its categorical masks are returned to the
original source grid with nearest-neighbor reprojection.
