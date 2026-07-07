import logging
import math
import os
from typing import Dict, List, Tuple

from PIL import Image

from .config import IngestorConfig

log = logging.getLogger(__name__)


# ─── Visual Token Estimation ─────────────────────────────────────────────────


def estimate_visual_tokens(image_path: str, config: IngestorConfig) -> int:
    """
    Estimate the number of visual tokens an image will produce, based on
    the shortest-side resolution and ``config.token_tiers``.

    The image is opened only to read dimensions, then immediately closed.
    """
    with Image.open(image_path) as img:
        width, height = img.size

    shortest_side = min(width, height)

    # Check tiers in descending order of threshold
    for threshold, tokens in sorted(
        config.token_tiers.items(), key=lambda x: x[0], reverse=True
    ):
        if shortest_side >= threshold:
            return tokens

    # Fallback: the lowest-tier value
    return min(config.token_tiers.values()) if config.token_tiers else 256



def _downscale_image(
    image_path: str,
    target_resolution: Tuple[int, int],
    quality: int = 85,
) -> str:
    """
    Downscale an image to ``target_resolution`` and save it as a new JPEG.

    The original file is deleted after the downscaled version is written.

    Returns the path to the new (downscaled) file.
    """
    dir_name = os.path.dirname(image_path)
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    new_path = os.path.join(dir_name, f"{name}_ds{ext}")

    with Image.open(image_path) as img:
        img_resized = img.resize(target_resolution, Image.LANCZOS)
        img_resized.save(new_path, "JPEG", quality=quality)

    try:
        os.remove(image_path)
    except OSError:
        pass

    return new_path




def enforce_token_budget(
    chunk_frames: List[Dict],
    config: IngestorConfig,
) -> List[Dict]:
    """
    Ensure the total visual tokens for a chunk stay under
    ``config.max_visual_tokens_per_chunk``.

    Strategy (in order):
      1. Downscale all images in the chunk to ``config.downscale_resolution``.
      2. If still over budget, drop the lowest-motion-priority frames until
         the budget is met.

    Returns the (possibly modified) list of frame dicts.
    """
    total_tokens = sum(f["token_estimate"] for f in chunk_frames)

    if total_tokens <= config.max_visual_tokens_per_chunk:
        return chunk_frames

    log.info(
        "Chunk token budget exceeded: %d > %d — downscaling %d frames",
        total_tokens, config.max_visual_tokens_per_chunk, len(chunk_frames),
    )

    for frame in chunk_frames:
        new_path = _downscale_image(
            frame["image_path"],
            config.downscale_resolution,
            quality=config.jpeg_quality,
        )
        frame["image_path"] = new_path
        frame["token_estimate"] = estimate_visual_tokens(new_path, config)

    total_tokens = sum(f["token_estimate"] for f in chunk_frames)
    if total_tokens <= config.max_visual_tokens_per_chunk:
        return chunk_frames

    log.warning(
        "Still over budget after downscaling (%d > %d) — dropping frames",
        total_tokens, config.max_visual_tokens_per_chunk,
    )

    motion_priority = {"high": 2, "normal": 1, "static": 0}

    chunk_frames.sort(
        key=lambda f: (
            motion_priority.get(f.get("motion_density", "normal"), 1),
            -f["timestamp_ms"],
        ),
        reverse=True,
    )

    kept: List[Dict] = []
    running_tokens = 0
    for frame in chunk_frames:
        if running_tokens + frame["token_estimate"] <= config.max_visual_tokens_per_chunk:
            kept.append(frame)
            running_tokens += frame["token_estimate"]
        else:
            # Clean up dropped frame from disk
            try:
                os.remove(frame["image_path"])
            except OSError:
                pass

    # Restore chronological order
    kept.sort(key=lambda f: f["timestamp_ms"])
    return kept




def aggregate_asr_words(
    words: List[Dict],
    window_start: float,
    window_end: float,
) -> Dict:
    """
    Group word-level ASR data into a single text string for a time window.

    A word is included if its *start* time falls within ``[window_start,
    window_end)``.  Words that cross the boundary are attributed to the
    window where they start (i.e. truncated to the prior window).

    Returns a dict matching the ``asr_data`` schema in the deliverable JSON.
    """
    window_words = [
        w for w in words
        if w["start"] >= window_start and w["start"] < window_end
    ]

    if not window_words:
        return {
            "text": "",
            "word_count": 0,
            "start_sec": round(window_start, 3),
            "end_sec": round(window_end, 3),
        }

    text = " ".join(w["word"] for w in window_words)
    return {
        "text": text,
        "word_count": len(window_words),
        "start_sec": round(float(window_words[0]["start"]), 3),
        "end_sec": round(float(window_words[-1]["end"]), 3),
    }



def _determine_dominant_motion(motion_classes: List[str]) -> str:
    """
    Pick the dominant motion class for a chunk.

    Rule: any ``"high"`` frame promotes the whole chunk to ``"high"``;
    otherwise majority vote between ``"normal"`` and ``"static"``.
    """
    if not motion_classes:
        return "normal"

    counts = {"static": 0, "normal": 0, "high": 0}
    for mc in motion_classes:
        counts[mc] = counts.get(mc, 0) + 1

    if counts["high"] > 0:
        return "high"

    return max(counts, key=counts.get)




def build_temporal_chunks(
    sampled_frames: List[Tuple[int, str, str]],
    asr_words: List[Dict],
    video_duration: float,
    config: IngestorConfig,
) -> List[Dict]:
    """
    Merge sampled frames and ASR words into the Phase 1 deliverable schema.

    Creates non-overlapping windows of ``config.window_sec`` seconds.
    For each window:
      • Collect frames within the time range
      • Estimate visual tokens and enforce the per-chunk budget
      • Aggregate ASR words into a sentence string
      • Flag chunks with no ASR data

    Parameters
    ----------
    sampled_frames : list[tuple[int, str, str]]
        ``(timestamp_ms, image_path, motion_density)`` from the sampler.
    asr_words : list[dict]
        ``{"word": str, "start": float, "end": float}`` from the transcriber.
    video_duration : float
        Total video duration in seconds.
    config : IngestorConfig
        Pipeline configuration.

    Returns
    -------
    list[dict]
        ``TemporalChunk`` dicts conforming to the Phase 1 deliverable schema.
    """
    num_windows = max(1, math.ceil(video_duration / config.step_sec))
    chunks: List[Dict] = []

    for chunk_id in range(num_windows):
        t_start = chunk_id * config.step_sec
        t_end = min(t_start + config.window_sec, video_duration)

        # ── Collect frames in this window ─────────────────────────────
        chunk_frame_data: List[Dict] = []
        motion_classes: List[str] = []

        for ts_ms, img_path, motion in sampled_frames:
            ts_sec = ts_ms / 1000.0
            if t_start <= ts_sec < t_end:
                token_est = estimate_visual_tokens(img_path, config)
                chunk_frame_data.append({
                    "timestamp_ms": ts_ms,
                    "image_path": img_path,
                    "token_estimate": token_est,
                    "motion_density": motion,
                })
                motion_classes.append(motion)

        chunk_frame_data = enforce_token_budget(chunk_frame_data, config)

        visual_data = [
            {
                "timestamp_ms": fd["timestamp_ms"],
                "image_path": fd["image_path"],
                "token_estimate": fd["token_estimate"],
            }
            for fd in chunk_frame_data
        ]

        asr_data = aggregate_asr_words(asr_words, t_start, t_end)

        dominant_motion = _determine_dominant_motion(motion_classes)

        total_visual_tokens = sum(fd["token_estimate"] for fd in visual_data)

        chunk = {
            "chunk_id": chunk_id,
            "time_start_sec": round(t_start, 3),
            "time_end_sec": round(t_end, 3),
            "visual_data": visual_data,
            "asr_data": asr_data,
            "metadata": {
                "total_visual_tokens": total_visual_tokens,
                "motion_density": dominant_motion,
                "asr_is_empty": asr_data["word_count"] == 0,
            },
        }

        if asr_data["word_count"] == 0:
            log.info("Chunk %d [%.1f–%.1fs] has no ASR data (silent)", chunk_id, t_start, t_end)

        chunks.append(chunk)

    log.info("Built %d temporal chunks from %.1fs video", len(chunks), video_duration)
    return chunks
