"""Model-specific calibrated operating thresholds."""

# Calibrated on the 308-chip internal validation split for
# epoch=02-macro_f1=0.5062.ckpt. Re-run prithvi-evaluate-binary before changing
# checkpoints or data splits.
CROP_CLASSIFICATION_THRESHOLD = 0.615
HEALTH_ANALYSIS_CROP_THRESHOLD = 0.76
