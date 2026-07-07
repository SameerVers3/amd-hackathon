import logging
import os
from typing import List, Tuple

import numpy as np
from PIL import Image

from .config import IngestorConfig
from . import demuxer


log = logging.getLogger(__name__)


# Frame Conversion Helpers 


def _rgb_bytes_to_grayscale(
    raw_bytes: bytes,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Convert raw RGB24 bytes to a 2-D uint8 grayscale array.

    Uses the ITU-R BT.601 luminance formula:
        Y = 0.299·R + 0.587·G + 0.114·B
    """
    rgb = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((height, width, 3))
    gray = (
        0.299 * rgb[:, :, 0]
        + 0.587 * rgb[:, :, 1]
        + 0.114 * rgb[:, :, 2]
    ).astype(np.uint8)
    return gray


def _save_frame_as_jpeg(
    raw_bytes: bytes,
    width: int,
    height: int,
    output_path: str,
    quality: int = 85,
) -> str:
    """Save raw RGB24 bytes as a JPEG file.  Returns ``output_path``."""
    rgb = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((height, width, 3))
    img = Image.fromarray(rgb, "RGB")
    img.save(output_path, "JPEG", quality=quality)
    img.close()
    return output_path


# Motion Analysis


def compute_mad(
    frame_current: np.ndarray,
    frame_previous: np.ndarray,
) -> float:
    """
    Mean Absolute Difference between two grayscale frames.
    """
    diff = np.abs(
        frame_current.astype(np.float32) - frame_previous.astype(np.float32)
    )
    return float(np.mean(diff))


def classify_motion(
    mad_value: float,
    epsilon_low: float,
    epsilon_high: float,
) -> str:
    """
    Classify inter-frame motion into one of three density classes.

    Returns ``"static"``, ``"normal"``, or ``"high"``.
    """
    if mad_value < epsilon_low:
        return "static"
    if mad_value <= epsilon_high:
        return "normal"
    return "high"


# Sampling-Interval Logic [todo: need to took at the math here]


def _keep_interval_frames(
    motion_class: str,
    analysis_fps: float,
    config: IngestorConfig,
) -> float:
    """
    How many analysis frames to wait between kept frames.

    E.g. at ``analysis_fps=5`` and target 2 FPS to keep every 2.5th frame.
    """
    target_fps = {
        "static": config.fps_static,
        "normal": config.fps_normal,
        "high": config.fps_high,
    }[motion_class]

    keep_interval_sec = 1.0 / target_fps
    return keep_interval_sec * analysis_fps


#  Uniform Sub-sampling (Budget Enforcement)


def _uniform_subsample(
    frames: List[Tuple[int, str, str]],
    n_max: int,
) -> List[Tuple[int, str, str]]:
    """
    Uniformly sub-sample ``frames`` down to ``n_max``, always keeping the
    first and last frame.  Frames that are dropped have their JPEG files
    deleted from disk.
    """
    if n_max >= len(frames):
        return frames
    if n_max <= 0:
        return []
    if n_max == 1:
        # Keep only the first frame; clean up the rest
        for _, path, _ in frames[1:]:
            _safe_remove(path)
        return [frames[0]]

    # Pick n_max indices uniformly across the range
    indices = sorted(set(
        int(round(i * (len(frames) - 1) / (n_max - 1))) for i in range(n_max)
    ))

    selected = set(indices)
    for i, (_, path, _) in enumerate(frames):
        if i not in selected:
            _safe_remove(path)

    return [frames[i] for i in indices]


def _safe_remove(path: str) -> None:
    """Remove a file, ignoring errors if it doesn't exist."""
    try:
        os.remove(path)
    except OSError:
        pass


# Main Entry Point 


def adaptive_sample_frames(
    video_path: str,
    config: IngestorConfig,
) -> List[Tuple[int, str, str]]:
    """
    Adaptively sample frames from a video based on motion density.

    1. Decode at ``config.analysis_fps`` via FFmpeg's ``fps`` filter.
    2. Compute MAD between consecutive frames to classify each frame.
    3. Keep frames according to the adaptive rate for each motion class.
    4. Enforce the linear frame cap via uniform sub-sampling.

    Parameters
    ----------
    video_path : str
        Path to the source video.
    config : IngestorConfig
        Pipeline configuration.

    Returns
    -------
    list[tuple[int, str, str]]
        Each element is ``(timestamp_ms, image_path, motion_density)``.
    """
    metadata = demuxer.get_video_metadata(video_path)
    duration_sec = metadata["duration_sec"]
    orig_width = metadata["width"]
    orig_height = metadata["height"]

    # Frame budget: linear scaling
    n_max = max(1, int(config.frames_per_minute * (duration_sec / 60.0)))

    analysis_fps = config.analysis_fps  # decode at this rate

    log.info(
        "Adaptive sampling: duration=%.1fs, analysis_fps=%.1f, "
        "resolution=%dx%d, frame_budget=%d",
        duration_sec, analysis_fps, orig_width, orig_height, n_max,
    )

    # Ensure output directory exists
    os.makedirs(config.frames_output_dir, exist_ok=True)

    # Build a frame generator at the analysis FPS
    frame_gen = demuxer.create_frame_generator(
        video_path, metadata, target_fps=analysis_fps,
    )

    sampled_frames: List[Tuple[int, str, str]] = []
    prev_gray: np.ndarray | None = None
    last_kept_index = -999  # large negative so the first frame is always kept
    frame_index = 0

    for timestamp_ms, raw_bytes, width, height in frame_gen:
        gray = _rgb_bytes_to_grayscale(raw_bytes, width, height)

        # First frame: always keep 
        if prev_gray is None:
            motion_class = "normal"  # default label for frame 0
            frame_path = _write_frame(
                raw_bytes, width, height, timestamp_ms, config,
            )
            sampled_frames.append((timestamp_ms, frame_path, motion_class))
            last_kept_index = frame_index
            prev_gray = gray
            frame_index += 1
            continue

        # Compute motion 
        mad = compute_mad(gray, prev_gray)
        motion_class = classify_motion(mad, config.epsilon_low, config.epsilon_high)

        #  Decide whether to keep this frame 
        keep_interval = _keep_interval_frames(motion_class, analysis_fps, config)
        frames_since_last = frame_index - last_kept_index

        if frames_since_last >= keep_interval:
            frame_path = _write_frame(
                raw_bytes, width, height, timestamp_ms, config,
            )
            sampled_frames.append((timestamp_ms, frame_path, motion_class))
            last_kept_index = frame_index

        prev_gray = gray
        frame_index += 1

    log.info("Sampled %d frames (budget: %d)", len(sampled_frames), n_max)

    # Budget enforcement 
    if len(sampled_frames) > n_max:
        log.warning(
            "Frame count %d exceeds budget %d — uniformly sub-sampling",
            len(sampled_frames), n_max,
        )
        sampled_frames = _uniform_subsample(sampled_frames, n_max)
        log.info("After sub-sampling: %d frames", len(sampled_frames))

    return sampled_frames


def _write_frame(
    raw_bytes: bytes,
    width: int,
    height: int,
    timestamp_ms: int,
    config: IngestorConfig,
) -> str:
    """Save a single frame as JPEG and return the file path."""
    filename = f"frame_{timestamp_ms:08d}.jpg"
    path = os.path.join(config.frames_output_dir, filename)
    _save_frame_as_jpeg(raw_bytes, width, height, path, quality=config.jpeg_quality)
    return path
