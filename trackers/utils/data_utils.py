import secrets
import zipfile


def unzip_file(source_zip_path: str, target_dir_path: str) -> None:
    """
    Extracts all files from a zip archive.

    Args:
        source_zip_path (str): The path to the zip file.
        target_dir_path (str): The directory to extract the contents to.
            If the directory doesn't exist, it will be created.
    """
    try:
        with zipfile.ZipFile(source_zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir_path)
        print(f"Successfully extracted '{source_zip_path}' to '{target_dir_path}'")
    except FileNotFoundError:
        print(f"Error: Zip file '{source_zip_path}' not found.")
    except zipfile.BadZipFile:
        print(f"Error: '{source_zip_path}' is not a valid zip file or is corrupted.")
    except Exception as e:
        print(f"An error occurred: {e}")


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
