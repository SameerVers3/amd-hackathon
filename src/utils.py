import os
import tempfile
from urllib.parse import urlparse

import requests
from tenacity import retry, stop_after_attempt, wait_exponential


# downloads video from url to dest_path, with retries on failure
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _download(url: str, dest_path: str) -> None:
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


# gives a local path for the video after downloading it if necessary
def fetch_clip(video_url: str, workdir: str) -> str:
    parsed = urlparse(video_url)

    if parsed.scheme in ("", "file"):
        local_path = parsed.path if parsed.scheme == "file" else video_url
        if os.path.exists(local_path):
            return local_path
        raise FileNotFoundError(f"Local video path not found: {local_path}")

    suffix = os.path.splitext(parsed.path)[1] or ".mp4"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, dir=workdir)
    os.close(fd)
    _download(video_url, tmp_path)
    return tmp_path
