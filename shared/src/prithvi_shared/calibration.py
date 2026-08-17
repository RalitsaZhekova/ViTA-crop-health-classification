"""Identity and calibrated thresholds for the selected model."""

SELECTED_CHECKPOINT_NAME = "prithvi_crop_binary_single_frame_v1_weights.pt"
SELECTED_CHECKPOINT_SHA256 = "c948977bffdaeb89ecf4f4d069db13c7ed81d4f3403eec9e257c45c235b1484e"
# Local operational demo settings: broaden the visible crop mask and admit
# moderately confident crop pixels into the health map.
CROP_CLASSIFICATION_THRESHOLD = 0.3
HEALTH_ANALYSIS_CROP_THRESHOLD = 0.3
