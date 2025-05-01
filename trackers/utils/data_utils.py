import zipfile


def unzip_file(zip_filepath, extract_to_path):
    """
    Extracts all files from a zip archive.

    Args:
        zip_filepath (str): The path to the zip file.
        extract_to_path (str): The directory to extract the contents to.
            If the directory doesn't exist, it will be created.
    """
    try:
        with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
            zip_ref.extractall(extract_to_path)
        print(f"Successfully extracted '{zip_filepath}' to '{extract_to_path}'")
    except FileNotFoundError:
        print(f"Error: Zip file '{zip_filepath}' not found.")
    except zipfile.BadZipFile:
        print(f"Error: '{zip_filepath}' is not a valid zip file or is corrupted.")
    except Exception as e:
        print(f"An error occurred: {e}")
