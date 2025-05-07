import secrets
import zipfile


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


def secure_sample(population: list[str], k: int = 1) -> list[str]:
    """Securely sample k elements from the population.

    Args:
        population (list[str]): List of elements to sample from
        k (int): Number of elements to sample

    Returns:
        List of k elements from the population
    """
    if k == 1:
        return [secrets.choice(population)]

    # For multiple samples, shuffle and take first k
    result = list(population)
    for i in range(len(result) - 1, 0, -1):
        j = secrets.randbelow(i + 1)
        result[i], result[j] = result[j], result[i]
    return result[:k]
