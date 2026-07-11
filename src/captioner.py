"""
Verified-Scene Video Captioning Pipeline.

3-step approach inspired by top-scoring competitor:
  Step 1: describe_scene()  — VLM sees 3 frames → 2-4 factual sentences
  Step 2: verify_description() — VLM sees same 3 frames + draft → corrects errors
  Step 3: write_caption() — Text-only, per style → 25-60 word caption
"""

import base64
import json
import io
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type
from PIL import Image

log = logging.getLogger(__name__)

REQUIRED_STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]

# ── Configuration ───────────────────────────────────────────────────────────

API_KEY = os.environ.get("FIREWORKS_API_KEY", "")
API_BASE = "https://api.fireworks.ai/inference/v1"
MODEL = os.environ.get("FIREWORKS_MODEL", "accounts/fireworks/models/kimi-k2p6")
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "4"))
FRAME_WIDTH = 768
MAX_RETRIES = 5

# ── Retry-safe HTTP errors ──────────────────────────────────────────────────

RETRY_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


class RetryableHTTPError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {detail}")


# ── Fireworks API Client ────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential_jitter(initial=1, max=32, jitter=2),
    retry=retry_if_exception_type((RetryableHTTPError, requests.RequestException)),
    reraise=True,
)
def _chat(messages: List[Dict], max_tokens: int = 700, temperature: float = 0.2,
          json_mode: bool = False, reasoning_effort: Optional[str] = None) -> str:
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code in RETRY_STATUS_CODES:
        raise RetryableHTTPError(resp.status_code, resp.text[:500])
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ── Frame Extraction ────────────────────────────────────────────────────────

def _probe_duration(video_path: str) -> Optional[float]:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json", video_path,
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        data = json.loads(result.stdout)
        dur = data.get("format", {}).get("duration")
        return float(dur) if dur else None
    except Exception:
        return None


def _sample_timestamps(duration: float, max_frames: int) -> List[float]:
    if duration <= 0 or max_frames <= 0:
        return []
    if max_frames == 1:
        return [max(duration / 2, 0)]
    start = min(0.5, max(duration * 0.05, 0))
    end = max(duration - 0.5, start)
    if end <= start:
        return [max(duration / 2, 0)]
    timestamps = []
    for i in range(max_frames):
        pos = i / (max_frames - 1)
        ts = round(start + (end - start) * pos, 3)
        timestamps.append(ts)
    return timestamps


def extract_frames(video_path: str, frame_dir: str, max_frames: int = MAX_FRAMES) -> List[str]:
    """Extract anchor frames from video. Returns list of frame file paths."""
    os.makedirs(frame_dir, exist_ok=True)
    duration = _probe_duration(video_path)
    if not duration:
        duration = 60.0  # fallback
    timestamps = _sample_timestamps(duration, max_frames)

    frames: List[str] = []
    for i, ts in enumerate(timestamps, start=1):
        out_path = os.path.join(frame_dir, f"frame_{i:03d}.jpg")
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{ts:.3f}",
            "-i", video_path,
            "-frames:v", "1",
            "-vf", f"scale={FRAME_WIDTH}:-1",
            "-q:v", "3",
            out_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            if os.path.exists(out_path):
                frames.append(out_path)
        except Exception:
            pass
    return frames


def _frame_to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ── Step 1: Describe Scene ──────────────────────────────────────────────────

def describe_scene(frames: List[str]) -> str:
    """VLM sees frames → produces 2-4 factual sentences."""
    content: List[Dict[str, Any]] = []
    content.append({
        "type": "text",
        "text": (
            "These are frames sampled across a short video clip, in chronological order. "
            "Note the setting, main subjects, specific action or motion, camera/scene changes, "
            "and any readable on-screen text. Write 2-4 dense, factual sentences. "
            "Be specific, do not generalize, and do not mention frames or analysis. "
            "Do not infer an exact city, country, brand context, or organization unless clearly readable. "
            "If an exact location is not directly visible, use generic wording like 'city street', 'office', or 'indoor room'. "
            "Do not speculate about motives, unseen objects, identity, deadlines, or hidden context. "
            "Use generic human descriptions unless a specific identity is directly relevant and visually certain."
        ),
    })
    for frame in frames:
        content.append({"type": "image_url", "image_url": {"url": _frame_to_data_url(frame)}})
    return _chat(
        [{"role": "user", "content": content}],
        max_tokens=220, temperature=0.2,
    ).strip()


# ── Step 2: Verify Description ──────────────────────────────────────────────

def verify_description(frames: List[str], draft: str) -> str:
    """VLM sees frames + draft → corrects any hallucinations."""
    content: List[Dict[str, Any]] = []
    content.append({
        "type": "text",
        "text": (
            f"Here is a draft description of these video frames:\n{draft}\n\n"
            "Check it against the actual frames. If it is accurate and specific, repeat it unchanged. "
            "If anything is wrong, too generic, or unsupported, correct it. "
            "Remove exact quoted text, brand names, signs, ethnicity, identity labels, or location claims unless "
            "they are clearly visible and central in the frames. Prefer generic wording when unsure. "
            "Output only the final factual description. Do not mention frames, AI, uncertainty, or analysis."
        ),
    })
    for frame in frames:
        content.append({"type": "image_url", "image_url": {"url": _frame_to_data_url(frame)}})
    return _chat(
        [{"role": "user", "content": content}],
        max_tokens=220, temperature=0.1,
    ).strip()


# ── Step 3: Write Caption ───────────────────────────────────────────────────

STYLE_PROMPTS = {
    "formal": (
        "Write a formal, professional, objective caption. Factual tone, no humor. "
        "Style example only: 'The subject proceeds through the marked route without deviation.'"
    ),
    "sarcastic": (
        "Write a sarcastic caption: dry, ironic, lightly mocking, grounded in the specific action described. "
        "Use patterns like 'because apparently...', 'clearly...', 'naturally...', or 'of course...' when they fit. "
        "The caption must be recognizably sarcastic; do not return a plain formal description. "
        "Style example only: 'The subject surveys its kingdom of one bench with the confidence of a landlord.'"
    ),
    "humorous_tech": (
        "Write a funny caption using technology, software, programming, network, game engine, or debugging references. "
        "The caption must include at least one clear tech reference such as queue, deploy, debug, API, latency, "
        "log, cache, scheduler, pipeline, rollback, commit, runtime, or bug. "
        "Do not return a plain formal description. "
        "Style example only: '404: graceful landing not found.'"
    ),
    "humorous_non_tech": (
        "Write a funny everyday-humor caption with no technical jargon, relatable and light-hearted. "
        "The caption must include a light everyday joke or comparison; do not return a plain formal description. "
        "Style example only: 'Confidence level: main character. Execution level: blooper reel.'"
    ),
}

CREATIVE_STYLES = {"sarcastic", "humorous_tech", "humorous_non_tech"}

TECH_KEYWORDS = {
    "api", "bug", "cache", "commit", "debug", "deploy", "latency",
    "log", "pipeline", "queue", "rollback", "runtime", "scheduler",
}
SARCASM_MARKERS = {
    "apparently", "because", "clearly", "naturally", "of course",
    "obviously", "serious", "thrilling",
}


def write_caption(description: str, style: str, prior_captions: List[str]) -> str:
    """Text-only caption generation from verified description."""
    variety_note = ""
    if prior_captions:
        variety_note = (
            "\n\nCaptions already written for this clip in other styles. "
            "Use a different sentence structure and comedic angle: "
            + " | ".join(prior_captions)
        )

    prompt = (
        f"{STYLE_PROMPTS.get(style, 'Match the requested style.')}\n\n"
        f"Factual description of the video:\n{description}\n\n"
        "Write ONE caption, 25 to 60 words, as if you personally watched the video. "
        "Never mention computer vision, models, detection, frames, prompts, pipelines, or uncertainty. "
        "Do not add events, objects, speech, or motives that are not in the description. "
        "Humor may exaggerate the importance of visible actions only. Do not invent new actions "
        "such as sniffing, speaking, demanding, performing, or reacting unless explicitly observed. "
        "Do not quote signs, brands, or identity labels unless they are explicitly present in the factual description. "
        "Output only the caption text."
        f"{variety_note}"
    )

    temp = 0.75 if style in CREATIVE_STYLES else 0.2
    caption = _chat(
        [{"role": "user", "content": prompt}],
        max_tokens=140, temperature=temp, reasoning_effort="none"
    ).strip().strip('"')

    # Style retry: if sarcastic/tech caption is still missing keywords, force them
    if _needs_style_retry(style, caption):
        if style == "humorous_tech":
            retry_prompt = (
                "Rewrite the caption. It was too formal and lacked tech flavor. "
                "You MUST include at least one clear technology, programming, or software engineering term "
                "(like cache, api, deploy, debug, latency). Keep the exact same facts."
            )
        else:
            retry_prompt = (
                "Rewrite the caption. It was too formal. "
                "You MUST make the tone obviously sarcastic, dry, and ironic. "
                "Use phrases like 'clearly', 'naturally', or 'of course'. Keep the exact same facts."
            )

        caption = _chat(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": caption},
             {"role": "user", "content": retry_prompt}],
            max_tokens=140, temperature=temp, reasoning_effort="none"
        ).strip().strip('"')

    return caption


def _needs_style_retry(style: str, caption: str) -> bool:
    normalized = caption.lower()
    if style == "humorous_tech":
        return not any(word in normalized for word in TECH_KEYWORDS)
    if style == "sarcastic":
        return not any(marker in normalized for marker in SARCASM_MARKERS)
    return False


# ── Clean Caption ────────────────────────────────────────────────────────────

def _clean_caption(text: str) -> str:
    text = text.strip()
    text = text.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    return text


# ── Main Entry Point ────────────────────────────────────────────────────────

def generate_captions(video_path: str, styles: List[str]) -> Dict[str, str]:
    """
    Full 3-step verified-scene captioning pipeline.
    """
    with tempfile.TemporaryDirectory() as workdir:
        frame_dir = os.path.join(workdir, "frames")

        # Extract 3 anchor frames
        log.info("Step 1: Extracting %d anchor frames from %s", MAX_FRAMES, video_path)
        frames = extract_frames(video_path, frame_dir, MAX_FRAMES)
        if not frames:
            log.error("No frames extracted from %s", video_path)
            return {s: "" for s in styles}

        # Step 1: Describe
        log.info("Step 1: Describing scene from %d frames", len(frames))
        draft = describe_scene(frames)
        log.info("Draft description: %s", draft[:200])

        # Step 2: Verify
        log.info("Step 2: Verifying description")
        verified = verify_description(frames, draft)
        log.info("Verified description: %s", verified[:200])

        # Step 3: Write captions per style
        captions: Dict[str, str] = {}
        prior: List[str] = []
        for style in styles:
            log.info("Step 3: Writing caption for style=%s", style)
            caption = write_caption(verified, style, prior)
            caption = _clean_caption(caption)
            captions[style] = caption
            prior.append(caption)
            log.info("Caption [%s]: %s", style, caption[:150])

    return captions
