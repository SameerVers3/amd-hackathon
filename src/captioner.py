"""
Single-Call MiniMax-M3 Captioning Pipeline
Architecture adapted from Competitor 5 (scored 0.90)

Key differences from our previous 0.68 approach:
- ONE API call per video instead of 5
- All 4 captions generated together in JSON
- No second model (Kimi removed)
- No Whisper (removed complexity)
- Sequential processing (no rate limits)
"""

import base64
import glob
import json
import logging
import os
import re
import subprocess
import tempfile
from typing import Any, Dict, List

import requests
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type

log = logging.getLogger(__name__)

REQUIRED_STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]

# ── Configuration ───────────────────────────────────────────────────────────

API_KEY = os.environ.get("FIREWORKS_API_KEY", "")
API_BASE = "https://api.fireworks.ai/inference/v1"
MODEL = "accounts/fireworks/models/minimax-m3"
FRAME_WIDTH = 512
MAX_FRAMES = 10

# ── Retry-safe HTTP errors ──────────────────────────────────────────────────

RETRY_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


class RetryableHTTPError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {detail}")


# ── System Prompt (adapted from Competitor 5's proven prompts) ──────────────

SYSTEM_PROMPT = """
You are an expert multimodal video captioning system.

You receive multiple frames sampled chronologically from a short video clip.

Treat all frames as one continuous video.

Infer the overall scene, actions, subjects, objects, and setting.

Your highest priority is visual accuracy.


OBJECTIVES

Priority 1: Produce captions that accurately describe the visible content.
Priority 2: Make every requested style clearly different.
Priority 3: Write natural, fluent English.


VISUAL GROUNDING RULES

Always:
✓ Describe only what is visible.
✓ Mention important actions, objects, interactions, and the overall setting.
✓ Summarize the entire clip instead of individual frames.

Never:
✗ Invent dialogue, names, brands, locations, occupations, emotions (unless visually obvious), sounds, text on signs, or events outside the clip.
If uncertain, use generic wording.

Good: "A person walks through a market."
Bad: "John happily walks through Times Square after buying groceries."


STYLE CONSISTENCY

Every caption must describe the SAME video.
Only the writing style should change.
Do not simply replace a few words.
Each style should sound like it was written by a different person.
Avoid repeated sentence structures, wording, or phrases across styles.


STYLE: FORMAL

Role: Professional visual description
Definition: Write like a professional journalist, documentary narrator, or museum curator. Neutral, objective, factual, polished. Describe only observable visual information. Never speculate about emotions, identities, intentions, or events outside the clip.
Rules:
• Professional tone.
• Third-person perspective.
• Complete grammatical sentences.
• No jokes, sarcasm, slang, emojis, or exclamation marks.
• Describe the overall clip instead of individual frames.
Calibration Examples:
• "A cyclist rides through a tree-lined intersection while nearby traffic moves steadily."
• "An orange kitten walks through a garden filled with green plants and fallen leaves."
• "An office employee works at a computer inside a modern open-plan workspace."
• "Several children play soccer on a grassy field during daylight."
• "A chef prepares food in a commercial kitchen using stainless-steel equipment."


STYLE: SARCASTIC

Role: Dry, deadpan internet sarcasm
Definition: Use subtle, clever sarcasm while remaining factually accurate. The humor should come from ironic observation instead of insults, cruelty, or absurd exaggeration.
Rules:
• Keep factual accuracy.
• Use one sarcastic observation.
• Never invent events or insult people.
• No profanity.
• Keep sarcasm light and witty.
Calibration Examples:
• "Wow, another person answering emails. History is being made."
• "A cat walking through a garden. Truly groundbreaking cinema."
• "Nothing says excitement like someone staring at a spreadsheet."
• "Breaking news: traffic continues to exist."
• "The meeting appears to be going exactly as everyone dreamed."


STYLE: HUMOROUS_TECH

Role: Technology and programming humor
Definition: Describe the actual scene while making the joke using software, programming, AI, computer science, engineering, gaming, robotics, or technology references. The technical joke should naturally fit the visual content.
Rules:
• Always describe the visible scene.
• Use programming or software analogies.
• AI, debugging, APIs, algorithms, versioning, memory, networking, and operating systems are encouraged.
• Avoid random technical buzzwords.
• Keep the joke understandable.
Calibration Examples:
• "The cat successfully updated its pathfinding algorithm and avoided every obstacle."
• "The office worker appears to be compiling deadlines with several unresolved warnings."
• "Those autumn trees clearly deployed Color Palette v2.0."
• "The cyclist's route optimization algorithm is performing well."
• "The chef is running a highly parallel cooking process with minimal latency."


STYLE: HUMOROUS_NON_TECH

Role: Relatable observational comedy
Definition: Use lighthearted everyday humor without any technology, software, gaming, programming, engineering, or internet jargon. Imagine a family-friendly stand-up comedian making a quick observation.
Rules:
• No technical references whatsoever.
• Playful but natural.
• Family friendly.
• Never invent events.
• Base every joke on the visible scene.
Calibration Examples:
• "That cat is walking around like it owns the whole neighborhood."
• "Everyone looks just busy enough to avoid making eye contact."
• "Those trees clearly coordinated their outfits this season."
• "Someone definitely practiced that move before today."
• "The chef looks one spilled ingredient away from becoming a TV star."


LENGTH

Each caption should:
• contain 15–40 words
• be 1–2 sentences
• remain concise


OUTPUT FORMAT

Return ONLY valid JSON.

Do NOT include:
- markdown
- explanations
- reasoning
- notes
- code fences

The JSON MUST contain EXACTLY these keys:

{
    "formal": "...",
    "sarcastic": "...",
    "humorous_tech": "...",
    "humorous_non_tech": "..."
}

Every value must be a string.
Nothing may appear before or after the JSON.
""".strip()


USER_PROMPT = """
Generate captions for the following styles: formal, sarcastic, humorous_tech, humorous_non_tech

Requirements:
- Base every caption ONLY on the provided video frames.
- Treat the frames as one continuous video.
- Do NOT describe frames individually.
- Preserve the same factual content across every style.
- Change ONLY the writing style and humor.
- If something is uncertain, describe it generically.
- Never hallucinate details.
- Return ONLY the required JSON object.
""".strip()


# ── Fireworks API Client ────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=2, max=30, jitter=3),
    retry=retry_if_exception_type((RetryableHTTPError, requests.RequestException)),
    reraise=True,
)
def _call_vision(messages: List[Dict], max_tokens: int = 4000, temperature: float = 0.7) -> str:
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code in RETRY_STATUS_CODES:
        raise RetryableHTTPError(resp.status_code, resp.text[:500])
    resp.raise_for_status()
    data = resp.json()
    try:
        content = data["choices"][0]["message"].get("content")
        if content is None:
            raise KeyError("content is None")
        return content
    except (KeyError, IndexError):
        log.warning("Malformed response (missing content), triggering retry")
        raise RetryableHTTPError(500, "Malformed response from Fireworks")


# ── Frame Extraction ────────────────────────────────────────────────────────

def extract_frames(video_path: str, frame_dir: str) -> List[str]:
    """Extract exactly 1 FPS up to MAX_FRAMES."""
    os.makedirs(frame_dir, exist_ok=True)
    out_pattern = os.path.join(frame_dir, "frame_%04d.jpg")

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps=1,scale={FRAME_WIDTH}:-1",
        "-q:v", "3", out_pattern
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        log.warning("ffmpeg frame extraction failed: %s", e.stderr)
        return []

    frames = sorted(glob.glob(os.path.join(frame_dir, "frame_*.jpg")))
    if len(frames) > MAX_FRAMES:
        sampled = []
        for i in range(MAX_FRAMES):
            idx = i * (len(frames) - 1) // (MAX_FRAMES - 1)
            sampled.append(frames[idx])
        return sampled
    return frames


def _frame_to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ── Build Messages ──────────────────────────────────────────────────────────

def _build_messages(frames: List[str]) -> List[Dict]:
    content = [{"type": "text", "text": USER_PROMPT}]
    for frame in frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": _frame_to_data_url(frame)}
        })
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


# ── Parse JSON Output ───────────────────────────────────────────────────────

def _parse_captions(raw: str) -> Dict[str, str]:
    cleaned = raw.strip()

    # Strip markdown fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Model may have wrapped JSON in prose — extract the first JSON object
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON found in output: {raw[:300]}")
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in output: {raw[:300]}") from exc

    missing = [s for s in REQUIRED_STYLES if s not in parsed]
    if missing:
        raise ValueError(f"Missing styles: {missing}")

    return {s: str(parsed[s]).strip() for s in REQUIRED_STYLES}


# ── Main Entry Point ────────────────────────────────────────────────────────

def generate_captions(video_path: str, styles: List[str]) -> Dict[str, str]:
    with tempfile.TemporaryDirectory() as workdir:
        frame_dir = os.path.join(workdir, "frames")

        log.info("Extracting 1 FPS frames from %s", video_path)
        frames = extract_frames(video_path, frame_dir)
        if not frames:
            log.error("No frames extracted from %s", video_path)
            return {s: "" for s in styles}

        log.info("Sending %d frames to MiniMax-M3 (single-call)", len(frames))
        messages = _build_messages(frames)

        try:
            raw = _call_vision(messages)
            captions = _parse_captions(raw)
        except (ValueError, json.JSONDecodeError) as e:
            log.warning("First attempt parse failed: %s. Retrying with strict prompt...", e)
            messages[0]["content"] += "\nReturn ONLY the raw JSON object, nothing else."
            raw = _call_vision(messages)
            captions = _parse_captions(raw)

        for style, cap in captions.items():
            log.info("Caption [%s]: %s", style, cap[:100])

    return captions
