import json
import logging
import os
import subprocess
from typing import Dict, List

from faster_whisper import WhisperModel

from .config import IngestorConfig

log = logging.getLogger(__name__)


# Silent-Audio Detection 

def handle_silent_audio(wav_path: str) -> bool:
    """
    Return ``True`` if the audio file appears silent or has no meaningful
    content (missing stream, zero duration, etc.).

    Uses ``ffprobe`` to inspect the audio stream without decoding the whole
    file.
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        wav_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        probe = json.loads(result.stdout)
        streams = probe.get("streams", [])

        audio_stream = next(
            (s for s in streams if s.get("codec_type") == "audio"),
            None,
        )
        if audio_stream is None:
            log.info("No audio stream in %s — treating as silent", wav_path)
            return True

        duration = float(audio_stream.get("duration", 0))
        if duration <= 0:
            log.info("Audio duration ≤ 0 in %s — treating as silent", wav_path)
            return True

        return False

    except Exception as exc:
        log.warning("Could not probe audio file %s: %s — treating as silent", wav_path, exc)
        return True


# Faster-Whisper Local Transcription

def transcribe_audio(wav_path: str, config: IngestorConfig) -> List[Dict]:
    """
    Transcribe an audio file using the local faster-whisper model.

    Parameters
    ----------
    wav_path : str
        Path to a 16 kHz mono WAV file.
    config : IngestorConfig
        Pipeline configuration specifying model, device, and compute type.

    Returns
    -------
    list[dict]
        Word-level dicts: ``{"word": str, "start": float, "end": float}``.
        Returns an empty list for silent / empty audio.
    """
    # Skip processing for silent audio
    if handle_silent_audio(wav_path):
        log.info("Audio is silent / empty — returning no words")
        return []

    log.info(
        "Loading local faster-whisper model: %s (device=%s, compute=%s)",
        config.whisper_model, config.whisper_device, config.whisper_compute_type
    )

    try:
        model = WhisperModel(
            config.whisper_model, 
            device=config.whisper_device, 
            compute_type=config.whisper_compute_type
        )
    except Exception as e:
        log.error("Failed to load faster-whisper model: %s", e)
        return []

    log.info("Transcribing audio file: %s", os.path.basename(wav_path))
    
    words: List[Dict] = []
    try:
        segments, info = model.transcribe(wav_path, word_timestamps=True)
        for segment in segments:
            if segment.words:
                for w in segment.words:
                    word_text = w.word.strip()
                    if word_text:
                        words.append({
                            "word": word_text,
                            "start": round(float(w.start), 3),
                            "end": round(float(w.end), 3),
                        })
    except Exception as e:
        log.error("Transcription failed: %s", e)
        return []

    log.info("Transcribed %d words from %s", len(words), os.path.basename(wav_path))
    return words
