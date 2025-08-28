import numpy as np
import supervision as sv

from trackers.log import get_logger

logger = get_logger(__name__)


def relabel_ids(detections: sv.Detections) -> sv.Detections:
    """
    Relabels `tracker_id`s to be contiguous integers starting from 0.

    Handles potential NaN or non-integer IDs gracefully. IDs that cannot be
    processed are left as -1 in the output.

    Args:
        detections (sv.Detections): The detections object whose `tracker_id`s
            need relabeling.

    Returns:
        sv.Detections: The detections object with relabeled `tracker_id`s.
        Returns the original object if no valid IDs are found or if input is empty.
    """
    if len(detections) == 0 or detections.tracker_id is None:
        return detections

    # Filter out potential NaN values
    valid_ids_mask = ~np.isnan(detections.tracker_id)
    if not np.any(valid_ids_mask):
        # All IDs were NaN or array was empty after filtering
        return detections

    # Get unique integer IDs
    try:
        unique_ids = np.unique(detections.tracker_id[valid_ids_mask].astype(int))
    except ValueError:
        logger.warning(
            "Could not convert tracker IDs to integers during relabeling. Skipping."
        )
        return detections

    if len(unique_ids) == 0:
        return detections

    # Now unique_ids contains only valid integers
    max_id: int = np.max(unique_ids)
    min_id: int = np.min(unique_ids)

    offset = 0
    if min_id < 0:
        logger.warning(
            f"Negative tracker IDs found ({min_id}). Shifting IDs for relabeling."
        )
        offset = -min_id
        max_id += offset

    if np.isnan(max_id):
        logger.warning("Max ID is NaN during relabeling after offset. Skipping.")
        return detections

    map_size = int(max_id) + 1
    id_map = np.full(map_size, fill_value=-1, dtype=int)
    new_id_counter = 0
    # Initialize new_ids based on the original tracker_id shape and type
    new_ids = np.full_like(detections.tracker_id, fill_value=-1, dtype=int)

    # Iterate through the original positions where IDs were valid
    original_indices = np.where(valid_ids_mask)[0]
    for i in original_indices:
        original_id = int(detections.tracker_id[i]) + offset
        if original_id >= map_size or original_id < 0:
            logger.warning(
                f"Original ID {original_id - offset} out of bounds for map "
                "during relabeling. Skipping ID."
            )
            continue

        if id_map[original_id] == -1:
            id_map[original_id] = new_id_counter
            new_ids[i] = new_id_counter
            new_id_counter += 1
        else:
            new_ids[i] = id_map[original_id]

    detections.tracker_id = new_ids
    return detections
