"""Fine-class and crop/no-crop mappings."""

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

NON_CROP_CLASS_IDS = (0, 1, 4, 5, 6)
CROP_PROBABILITY_CLASS_IDS = (2, 3, 7, 8, 9, 10, 11, 12)
BINARY_CROP_TARGET_IDS = (2, 3, 7, 8, 9, 10, 11)
BINARY_IGNORE_CLASS_IDS = (12,)

# Fallow and Other are excluded from vegetation-health calculations.
HEALTH_CROP_CLASS_IDS = (2, 3, 7, 8, 10, 11)


def class_name(class_id: int) -> str:
    """Return a validated human-readable class name."""
    try:
        return CLASS_NAMES[class_id]
    except IndexError as error:
        raise ValueError(f"Unknown class ID: {class_id}") from error
