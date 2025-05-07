import zipfile

from trackers.log import get_logger

logger = get_logger(__name__)


def unzip_file(source_zip_path: str, target_dir_path: str) -> None:
    """
    Extracts all files from a zip archive.

    Args:
        source_zip_path (str): The path to the zip file.
        target_dir_path (str): The directory to extract the contents to.
            If the directory doesn't exist, it will be created.

    Raises:
        FileNotFoundError: If the zip file doesn't exist.
        zipfile.BadZipFile: If the file is not a valid zip file or is corrupted.
        Exception: If any other error occurs during extraction.
    """
    with zipfile.ZipFile(source_zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir_path)
    print(f"Successfully extracted '{source_zip_path}' to '{target_dir_path}'")


def validate_tracker_id_to_images(
    tracker_id_to_images: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Validate the tracker ID to images dictionary.

    Args:
        tracker_id_to_images (dict[str, list[str]]): The tracker ID to images
            dictionary.

    Returns:
        dict[str, list[str]]: The validated tracker ID to images dictionary.
    """
    # Create a copy of the dictionary to avoid modifying during iteration
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
