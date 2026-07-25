# Integration Guide

```python
from cloud_detection.pipeline import CloudDetectionPipeline

pipeline = CloudDetectionPipeline.from_yaml(
    "payload/cloud_detection/configs/cloud_detector.yaml"
)
result = pipeline.predict_file("scene.tif", "outputs")

clear_pixels = result.unusable_mask == 0
```

The cropland and vegetation-condition modules must ignore every pixel where
`result.unusable_mask == 1`.

Recommended integration metadata:

```json
{
  "cloud_percentage": 25.4,
  "shadow_percentage": 4.8,
  "unusable_percentage": 33.1,
  "usable_percentage": 66.9,
  "decision": "PROCESS_CLEAR_AREAS",
  "score_kind": "softmax_confidence"
}
```
