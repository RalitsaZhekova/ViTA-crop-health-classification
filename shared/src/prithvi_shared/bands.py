"""Model input bands, shape and training normalization."""

MODEL_BANDS = ("BLUE", "GREEN", "RED", "NIR_NARROW")
TIME_STEPS = 3
INPUT_HEIGHT = 224
INPUT_WIDTH = 224

# TerraTorch MultiTemporalCropClassification training statistics, in the
# numeric scale of the IBM-NASA source rasters.
NORMALIZATION_MEANS = (
    494.905781,
    815.239594,
    924.335066,
    2968.881459,
)
NORMALIZATION_STDS = (
    284.925432,
    357.848760,
    575.566823,
    896.601013,
)


def validate_band_order(bands: tuple[str, ...]) -> None:
    """Reject an input whose band order differs from the trained contract."""
    if bands != MODEL_BANDS:
        raise ValueError(f"Expected band order {MODEL_BANDS}, got {bands}")
