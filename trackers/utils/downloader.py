import asyncio
import os.path
import shutil
from tempfile import TemporaryDirectory
from typing import Optional
from urllib.parse import urlparse

import aiofiles
import aiohttp
from tqdm.asyncio import tqdm as async_tqdm


class _AsyncFileDownloader:
    """Asynchronously downloads files with
        support for multipart downloading and progress bars.

    This class handles downloading files from URLs, automatically determining whether
    to use multipart downloading based on server support for content length.
    It displays progress using tqdm.
    """

    def __init__(self, part_size_mb: int = 10, default_chunk_size: int = 8192):
        """Initializes the AsyncFileDownloader.

        Args:
            part_size_mb (int): The size of each part in megabytes for multipart downloads.
            default_chunk_size (int): The default chunk size in bytes for reading content.
        """  # noqa: E501
        self.part_size = part_size_mb * 1024**2
        self.default_chunk_size = default_chunk_size

    async def _get_content_length(self, url: str) -> Optional[int]:
        """Retrieves the content length of a file from a URL.

        Args:
            url (str): The URL of the file.

        Returns:
            Optional[int]: The content length in bytes if available, otherwise None.
        """
        async with aiohttp.ClientSession() as session:
            async with session.head(url) as request:
                return request.content_length

    def _parts_generator(self, size: int, start: int = 0):
        """Generates byte ranges for multipart downloading.

        Args:
            size (int): The total size of the file in bytes.
            start (int): The starting byte offset. Defaults to 0.

        Yields:
            Tuple[int, int]: A tuple representing the start and end byte of a part.
        """
        while size - start > self.part_size:
            yield start, start + self.part_size
            start += self.part_size
        yield start, size

    async def _download_part(
        self, url: str, headers: dict, save_path: str, progress_bar: async_tqdm
    ):
        """Downloads a single part of a file.

        Args:
            url (str): The URL to download from.
            headers (dict): HTTP headers to use for the request (e.g., for Range).
            save_path (str): The local path to save the downloaded part.
            progress_bar (async_tqdm): An instance of tqdm to update download progress.
        """
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url) as request:
                async with aiofiles.open(save_path, "wb") as file:
                    async for chunk in request.content.iter_chunked(
                        self.default_chunk_size
                    ):
                        await file.write(chunk)
                        progress_bar.update(len(chunk))

    async def process_url(
        self,
        url: str,
        save_dir: Optional[str] = None,
        output_filename: Optional[str] = None,
    ) -> str:
        """Downloads a file from a URL, handling multipart downloads and progress.

        If the server provides content length, the file is downloaded in parts.
        Otherwise, a direct download is attempted. Progress is displayed using tqdm.

        Args:
            url (str): The URL of the file to download.
            save_dir (Optional[str]): The directory to save the downloaded file.
                Defaults to the current working directory.
            output_filename (Optional[str]): The desired filename for the downloaded file.
                If None, it's inferred from the URL.

        Returns:
            str: The full path to the downloaded file.
        """  # noqa: E501
        if output_filename is None:
            output_filename = os.path.basename(urlparse(url).path)

        if save_dir is None:
            save_dir = os.path.abspath(".")
        final_save_path = os.path.join(save_dir, output_filename)
        os.makedirs(save_dir, exist_ok=True)
        tmp_dir = TemporaryDirectory(prefix=output_filename, dir=save_dir)
        try:
            size = await self._get_content_length(url)
            if size is None:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as request:
                        content_length = request.content_length
                        with async_tqdm(
                            total=content_length,
                            unit="B",
                            unit_scale=True,
                            desc=f"Downloading {output_filename}",
                            leave=True,
                        ) as pbar:
                            async with aiofiles.open(final_save_path, "wb") as file:
                                async for chunk in request.content.iter_chunked(
                                    self.default_chunk_size
                                ):
                                    await file.write(chunk)
                                    pbar.update(len(chunk))
                return final_save_path

            tasks = []
            file_parts = []
            with async_tqdm(
                total=size,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {output_filename}",
                leave=True,
            ) as pbar:
                for number, sizes in enumerate(self._parts_generator(size)):
                    part_file_name = os.path.join(
                        tmp_dir.name, f"{output_filename}.part{number}"
                    )
                    file_parts.append(part_file_name)
                    tasks.append(
                        self._download_part(
                            url,
                            {"Range": f"bytes={sizes[0]}-{sizes[1] - 1}"},
                            part_file_name,
                            pbar,
                        )
                    )

                await asyncio.gather(*tasks)

            with open(final_save_path, "wb") as wfd:
                for f_part_path in file_parts:
                    with open(f_part_path, "rb") as fd:
                        shutil.copyfileobj(fd, wfd)
            return final_save_path
        finally:
            tmp_dir.cleanup()


def download_file(
    url: str, part_size_mb: int = 10, default_chunk_size: int = 8192
) -> str:
    """Asynchronously downloads files with support for multipart downloading and progress bars.

    This class handles downloading files from URLs, automatically determining whether
    to use multipart downloading based on server support for content length.
    It displays progress using tqdm.

    Args:
        url (str): The URL to download the model file from.
        part_size_mb (int): The size of each part in megabytes for multipart downloads.
        default_chunk_size (int): The default chunk size in bytes for reading content.

    Returns:
        str: The local path to the downloaded file.
    """  # noqa: E501

    downloader = _AsyncFileDownloader(
        part_size_mb=part_size_mb, default_chunk_size=default_chunk_size
    )
    if not url:
        raise ValueError("URL cannot be empty.")
    if not urlparse(url).scheme:
        raise ValueError("Invalid URL. Please provide a valid URL.")
    if not urlparse(url).netloc:
        raise ValueError("Invalid URL. Please provide a valid URL.")
    if not urlparse(url).path:
        raise ValueError("Invalid URL. Please provide a valid URL.")

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            future = asyncio.ensure_future(downloader.process_url(url))
            file_path = loop.run_until_complete(future)
        else:
            file_path = loop.run_until_complete(downloader.process_url(url))
    except RuntimeError:
        file_path = asyncio.run(downloader.process_url(url))
    print(f"File downloaded to {file_path}.")
    return file_path
