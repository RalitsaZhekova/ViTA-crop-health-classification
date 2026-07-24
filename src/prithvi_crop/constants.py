"""Shared, auditable constants for the IBM-NASA crop dataset and model."""

from __future__ import annotations

DATASET_ID = "ibm-nasa-geospatial/multi-temporal-crop-classification"
# Current official revision resolved and recorded during repository validation.
DATASET_REVISION = "f285bb27c8f623a0fb6a44a6fd953c3ad34007d6"

SPECTRAL_BANDS = (
    "BLUE",
    "GREEN",
    "RED",
    "NIR_NARROW",
    "SWIR_1",
    "SWIR_2",
)
MODEL_BANDS = SPECTRAL_BANDS[:4]
TIME_STEPS = 3
RASTER_BAND_ORDER = SPECTRAL_BANDS * TIME_STEPS

CLASS_NAMES = (
    "Natural Vegetation",
    "Forest",
    "Corn",
    "Soybeans",
    "Wetlands",
    "Developed / Barren",
    "Open Water",
    "Winter Wheat",
    "Alfalfa",
    "Fallow / Idle Cropland",
    "Cotton",
    "Sorghum",
    "Other",
)
NUM_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = -1

EXPECTED_TOTAL_CHIPS = 3_854
EXPECTED_TRAINING_CHIPS = 3_083
EXPECTED_VALIDATION_CHIPS = 771
EXPECTED_IMAGE_SIZE = (224, 224)
EXPECTED_CRS = "EPSG:5070"

METADATA_FILES = ("training_data.txt", "validation_data.txt", "chips_df.csv")
ARCHIVES = {
    "training": "training_chips.tgz",
    "validation": "validation_chips.tgz",
}
