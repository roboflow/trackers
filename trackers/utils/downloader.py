# trackers/utils/downloader.py

import zipfile
from pathlib import Path
import requests
from tqdm import tqdm


def download_with_progress(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))

    with open(tmp, "wb") as f, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc=dst.name,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    tmp.rename(dst)


def extract_zip(zip_path: Path, output_dir: Path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
