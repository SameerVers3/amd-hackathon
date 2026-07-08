import json
import logging
import os
import subprocess
from typing import Dict, List

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

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


#  Fireworks Whisper API Call (we can try wisper too but let's see)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
)
def _call_fireworks_whisper(wav_path: str, config: IngestorConfig) -> dict:
    """
    POST the audio file to Fireworks AI's Whisper endpoint.

    Retries up to 3 times with exponential back-off on transient failures.
    """
    url = f"{config.fireworks_api_base}/audio/transcriptions"

    headers = {
        "Authorization": f"Bearer {config.fireworks_api_key}",
    }

    with open(wav_path, "rb") as audio_file:
        files = {
            "file": (os.path.basename(wav_path), audio_file, "audio/wav"),
        }
        data = {
            "model": config.whisper_model,
            "response_format": "verbose_json",
            "temperature": str(config.whisper_temperature),
            "language": config.whisper_language,
        }

        log.info(
            "Calling Fireworks Whisper API: model=%s, file=%s",
            config.whisper_model,
            os.path.basename(wav_path),
        )
        response = requests.post(
            url,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )
        response.raise_for_status()

    return response.json()



def _parse_verbose_response(response_json: dict) -> List[Dict]:
    """
    Extract word-level timestamps from a Fireworks ``verbose_json`` response.

    Falls back to segment-level timestamps (with uniform word-time
    distribution) if word-level data is absent.

    Returns
    -------
    list[dict]
        Each dict has keys ``word`` (str), ``start`` (float), ``end`` (float).
    """
    words: List[Dict] = []

    #  Primary: word-level from segments 
    segments = response_json.get("segments", [])
    for segment in segments:
        for w in segment.get("words", []):
            word_text = w.get("word", "").strip()
            if word_text:
                words.append({
                    "word": word_text,
                    "start": float(w.get("start", 0)),
                    "end": float(w.get("end", 0)),
                })

    if words:
        return words

    # A: top-level "words" key 
    for w in response_json.get("words", []):
        word_text = w.get("word", "").strip()
        if word_text:
            words.append({
                "word": word_text,
                "start": float(w.get("start", 0)),
                "end": float(w.get("end", 0)),
            })

    if words:
        return words

    # Fallback B: segment-level with uniform distribution 
    if segments:
        log.warning("No word-level timestamps; distributing segment text uniformly")
        for segment in segments:
            text = segment.get("text", "").strip()
            if not text:
                continue

            seg_words = text.split()
            seg_start = float(segment.get("start", 0))
            seg_end = float(segment.get("end", 0))
            seg_duration = seg_end - seg_start

            if not seg_words:
                continue

            word_duration = seg_duration / len(seg_words)
            for i, w in enumerate(seg_words):
                words.append({
                    "word": w,
                    "start": round(seg_start + i * word_duration, 3),
                    "end": round(seg_start + (i + 1) * word_duration, 3),
                })

    return words


def transcribe_audio(wav_path: str, config: IngestorConfig) -> List[Dict]:
    """
    Transcribe an audio file using the Fireworks AI Whisper API.

    Parameters
    ----------
    wav_path : str
        Path to a 16 kHz mono WAV file.
    config : IngestorConfig
        Pipeline configuration (must include a valid ``fireworks_api_key``).

    Returns
    -------
    list[dict]
        Word-level dicts: ``{"word": str, "start": float, "end": float}``.
        Returns an empty list for silent / empty audio.

    Raises
    ------
    ValueError
        If ``FIREWORKS_API_KEY`` is not set.
    """
    if not config.fireworks_api_key:
        raise ValueError(
            "FIREWORKS_API_KEY is not set. "
            "Set it in your environment or .env file."
        )

    # Skip API call for silent audio
    if handle_silent_audio(wav_path):
        log.info("Audio is silent / empty — returning no words")
        return []

    response_json = _call_fireworks_whisper(wav_path, config)
    words = _parse_verbose_response(response_json)

    log.info("Transcribed %d words from %s", len(words), os.path.basename(wav_path))
    return words
