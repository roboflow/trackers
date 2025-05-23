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
    logger.info(f"Successfully extracted '{source_zip_path}' to '{target_dir_path}'")
