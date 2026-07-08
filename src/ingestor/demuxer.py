import json
import logging
import subprocess
from typing import Generator, Tuple

log = logging.getLogger(__name__)




def get_video_metadata(video_path: str) -> dict:
    """
    Return video metadata via ``ffprobe``.

    Returns
    -------
    dict
        Keys: ``duration_sec``, ``fps``, ``width``, ``height``, ``has_audio``.
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    probe = json.loads(result.stdout)

    # Locate the first video and audio streams
    video_stream = None
    audio_stream = None
    for stream in probe.get("streams", []):
        codec_type = stream.get("codec_type")
        if codec_type == "video" and video_stream is None:
            video_stream = stream
        elif codec_type == "audio" and audio_stream is None:
            audio_stream = stream

    if video_stream is None:
        raise ValueError(f"No video stream found in {video_path}")

    # Parse FPS from r_frame_rate (e.g. "30/1", "30000/1001")
    fps_raw = video_stream.get("r_frame_rate", "30/1")
    parts = fps_raw.split("/")
    fps = float(parts[0]) / float(parts[1]) if len(parts) == 2 else float(parts[0])

    # Duration: prefer format-level (more reliable), fall back to stream
    duration = float(
        probe.get("format", {}).get(
            "duration",
            video_stream.get("duration", 0),
        )
    )

    return {
        "duration_sec": duration,
        "fps": fps,
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "has_audio": audio_stream is not None,
    }




def extract_audio(
    video_path: str,
    output_wav_path: str,
    sample_rate: int = 16_000,
) -> str:
    """
    Extract the audio track to a 16 kHz mono WAV file.

    Parameters
    ----------
    video_path : str
        Path to the source video.
    output_wav_path : str
        Destination path for the WAV file.
    sample_rate : int
        Target sample rate in Hz (default 16 000).

    Returns
    -------
    str
        The ``output_wav_path`` on success.

    Raises
    ------
    RuntimeError
        If FFmpeg exits with a non-zero return code.
    """
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",                     # discard video
        "-acodec", "pcm_s16le",    # 16-bit signed little-endian PCM
        "-ar", str(sample_rate),   # target sample rate
        "-ac", "1",                # mono
        "-y",                      # overwrite without asking
        output_wav_path,
    ]
    log.info("Extracting audio: %s → %s (sr=%d)", video_path, output_wav_path, sample_rate)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log.error("FFmpeg audio extraction failed:\n%s", result.stderr)
        raise RuntimeError(f"FFmpeg audio extraction failed (rc={result.returncode})")

    log.info("Audio extracted successfully: %s", output_wav_path)
    return output_wav_path



def create_frame_generator(
    video_path: str,
    metadata: dict,
    target_fps: float | None = None,
) -> Generator[Tuple[int, bytes, int, int], None, None]:
    """
    Lazily yield ``(timestamp_ms, raw_rgb_bytes, width, height)`` from a video.

    Frames are read one-at-a-time from an FFmpeg stdout pipe → O(1) memory.

    Parameters
    ----------
    video_path : str
        Path to the source video.
    metadata : dict
        Output of :func:`get_video_metadata` (provides width, height, fps).
    target_fps : float or None
        If provided, FFmpeg's ``fps`` filter is applied so that only frames
        at this rate are decoded.  Useful for reducing decode work when the
        source FPS is much higher than needed.

    Yields
    ------
    tuple[int, bytes, int, int]
        ``(timestamp_ms, raw_rgb24_bytes, width, height)``
    """
    width = metadata["width"]
    height = metadata["height"]
    effective_fps = target_fps if target_fps is not None else metadata["fps"]

    vf_filters = []
    if target_fps is not None:
        vf_filters.append(f"fps={target_fps}")

    cmd = [
        "ffmpeg",
        "-i", video_path,
    ]
    if vf_filters:
        cmd += ["-vf", ",".join(vf_filters)]
    cmd += [
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-v", "quiet",
        "-",
    ]

    frame_size = width * height * 3  # RGB24: 3 bytes per pixel

    log.info(
        "Frame generator started: %dx%d @ %.2f effective fps",
        width, height, effective_fps,
    )
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        frame_index = 0
        while True:
            raw_bytes = process.stdout.read(frame_size)
            if len(raw_bytes) < frame_size:
                break  # end of stream or partial read

            timestamp_ms = int((frame_index / effective_fps) * 1000)
            yield (timestamp_ms, raw_bytes, width, height)
            frame_index += 1
    finally:
        process.stdout.close()
        process.wait()

    log.info("Frame generator finished: yielded %d frames", frame_index)
