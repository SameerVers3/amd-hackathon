import argparse
import json
import logging
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from .config import IngestorConfig
from . import demuxer
from . import sampler
from . import transcriber
from . import chunker

log = logging.getLogger(__name__)

# demuxer -> sampler + transcriber (parallel) -> chunker -> manifest



def ingest_video(
    video_path: str,
    config: Optional[IngestorConfig] = None,
) -> List[Dict]:
    """
    run the ingestor pipeline on a single video file, returning the temporal chunks.

    Steps
    -----
    1. Validate the input file.
    2. Probe video metadata (duration, resolution, FPS, audio presence).
    3. In parallel:
       a. Extract audio to transcribe via Fireworks Whisper API.
       b. Adaptively sample visual frames based on motion density.
    4. Merge visual and ASR data into temporal chunks with token budgeting.

    Parameters
    ----------
    video_path : str
        Absolute or relative path to the input video.
    config : IngestorConfig or None
        Pipeline configuration.  If ``None``, defaults are used.

    Returns
    -------
    list[dict]
        ``TemporalChunk`` dicts conforming to the Phase 1 deliverable schema.
    """
    if config is None:
        config = IngestorConfig()

    # Validate input 
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Prepare output directory
    os.makedirs(config.frames_output_dir, exist_ok=True)

    # Probe metadata
    log.info("Probing video metadata: %s", video_path)
    metadata = demuxer.get_video_metadata(video_path)
    duration_sec = metadata["duration_sec"]
    has_audio = metadata["has_audio"]

    log.info(
        "Video: %.1fs | %dx%d | %.2f fps | audio=%s",
        duration_sec,
        metadata["width"],
        metadata["height"],
        metadata["fps"],
        has_audio,
    )

    # Parallel: audio + visual pipelines 
    asr_words: List[Dict] = []
    sampled_frames: List = []

    with tempfile.TemporaryDirectory() as audio_workdir:

        def _run_audio_pipeline() -> List[Dict]:
            """Extract audio and transcribe it."""
            log.info("ASR pipeline temporarily disabled per user request.")
            return []
            
            # if not has_audio:
            #     log.info("No audio stream — skipping transcription")
            #     return []
            # 
            # wav_path = os.path.join(audio_workdir, "audio.wav")
            # demuxer.extract_audio(video_path, wav_path, config.audio_sample_rate)
            # return transcriber.transcribe_audio(wav_path, config)

        def _run_visual_pipeline() -> List:
            """Adaptively sample frames."""
            return sampler.adaptive_sample_frames(video_path, config)

        with ThreadPoolExecutor(max_workers=2) as pool:
            future_audio = pool.submit(_run_audio_pipeline)
            future_visual = pool.submit(_run_visual_pipeline)

            # Wait for both; propagate exceptions immediately
            for future in as_completed([future_audio, future_visual]):
                exc = future.exception()
                if exc is not None:
                    log.error("Pipeline task failed: %s", exc, exc_info=True)
                    raise exc

            asr_words = future_audio.result()
            sampled_frames = future_visual.result()

    log.info("Audio pipeline: %d words transcribed", len(asr_words))
    log.info("Visual pipeline: %d frames sampled", len(sampled_frames))

    # Temporal chunking
    chunks = chunker.build_temporal_chunks(
        sampled_frames, asr_words, duration_sec, config,
    )

    log.info("ingestion complete: %d temporal chunks", len(chunks))
    return chunks


#  Manifest I/O 


def save_manifest(chunks: List[Dict], output_path: str) -> str:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, output_path)

    log.info("Manifest saved: %s (%d chunks)", output_path, len(chunks))
    return output_path


# CLI Entry Point 


def main() -> int:
    """
    CLI entry point for the ingestor.

    Usage::
        python -m ingestor --video-path /path/to/video.mp4 --output manifest.json
    """
    parser = argparse.ArgumentParser(
        description="Phase 1: Adaptive Ingestion & Temporal-Semantic Alignment",
    )
    parser.add_argument(
        "--video-path",
        required=True,
        help="Path to the input video file",
    )
    parser.add_argument(
        "--output",
        default="manifest.json",
        help="Output path for the JSON manifest (default: manifest.json)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional JSON file with config overrides "
             "(keys matching IngestorConfig field names)",
    )

    args = parser.parse_args()

    config = IngestorConfig()
    if args.config:
        with open(args.config) as f:
            overrides = json.load(f)
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                log.warning("Unknown config key ignored: %s", key)

    chunks = ingest_video(args.video_path, config)
    save_manifest(chunks, args.output)

    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    sys.exit(main())
