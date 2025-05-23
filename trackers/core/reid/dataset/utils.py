from trackers.log import get_logger

logger = get_logger(__name__)


def validate_tracker_id_to_images(
    tracker_id_to_images: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Validates a dictionary that maps tracker IDs to lists of image paths for the
    `TripletsDataset` for training ReID models using triplet loss.

    Args:
        tracker_id_to_images (dict[str, list[str]]): The tracker ID to images
            dictionary.

    Returns:
        dict[str, list[str]]: The validated tracker ID to images dictionary.
    """
    valid_tracker_ids = {}
    for tracker_id, image_paths in tracker_id_to_images.items():
        if len(image_paths) < 2:
            logger.warning(
                f"Tracker ID '{tracker_id}' has less than 2 images. "
                f"Skipping this tracker ID."
            )
        else:
            valid_tracker_ids[tracker_id] = image_paths

    if len(valid_tracker_ids) < 2:
        raise ValueError(
            "Tracker ID to images dictionary must contain at least 2 items "
            "to select negative samples."
        )

    return valid_tracker_ids
