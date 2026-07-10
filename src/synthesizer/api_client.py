import asyncio
import json
import logging
import random
import re
import aiohttp
from typing import List, Optional

from .config import SynthesizerConfig
from .prompts import build_system_prompt

log = logging.getLogger(__name__)

def sanitize_caption(api_response: str) -> str:
    # Extracts the caption from the <caption> tags and strips the <analysis> block.
    match = re.search(r'<caption>\s*(.*?)\s*</caption>', api_response, re.DOTALL | re.IGNORECASE)
    if match:
        caption = match.group(1).strip()
        return caption.strip('"').strip("'")
        
    # Fallback: if there is an opening tag but it got truncated before the closing tag
    match = re.search(r'<caption>\s*(.*)', api_response, re.DOTALL | re.IGNORECASE)
    if match:
        caption = match.group(1).strip()
        return caption.strip('"').strip("'")
        
    # Final fallback: manually strip out the analysis block if tags were completely missed
    if "<analysis>" in api_response:
        api_response = re.sub(r'<analysis>.*?</analysis>', '', api_response, flags=re.DOTALL | re.IGNORECASE)
        api_response = re.sub(r'<analysis>.*', '', api_response, flags=re.DOTALL | re.IGNORECASE) # if closing tag missing
        
    log.warning("No <caption> tags found in response. Returning cleaned raw text.")
    return api_response.strip().strip('"').strip("'")

def build_multimodal_payload(timeline_json: str, b64_frames: List[str]) -> List[dict]:
    # Interleaves timeline JSON and frames
    content = [{"type": "text", "text": "Here are the anchor frames from the video:"}]
    for b64 in b64_frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
    content.append({"type": "text", "text": f"Here is the semantic timeline:\n{timeline_json}"})
    return [{"role": "user", "content": content}]

async def call_31b_api(
    session: aiohttp.ClientSession,
    style: str,
    timeline_json: str,
    b64_frames: List[str],
    config: SynthesizerConfig
) -> Optional[str]:
    # Calls Fireworks API for a specific style
    url = f"{config.fireworks_api_base}{config.chat_endpoint}"
    headers = {
        "Authorization": f"Bearer {config.fireworks_api_key}",
        "Content-Type": "application/json"
    }
    
    params = config.style_sampling.get(style, {"temperature": 0.5, "top_p": 0.9})
    system_prompt = build_system_prompt(style)
    user_message = build_multimodal_payload(timeline_json, b64_frames)
    
    payload = {
        "model": config.model_id,
        "messages": [{"role": "system", "content": system_prompt}] + user_message,
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "max_tokens": config.max_tokens
    }
    
    for attempt in range(config.max_retries):
        try:
            async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                if response.status == 200:
                    data = await response.json()
                    choice = data.get("choices", [{}])[0]
                    message = choice.get("message", {})
                    content = message.get("content")
                    
                    if not content:
                        reasoning = message.get("reasoning_content", "")
                        finish = choice.get("finish_reason")
                        log.error("API returned no content. Finish reason: %s. Reasoning: %s...", finish, reasoning[:100])
                        return None
                        
                    return sanitize_caption(content)
                    
                elif response.status == 429:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(wait_time)
                else:
                    text = await response.text()
                    log.error("API failed %d for style %s: %s", response.status, style, text)
                    if response.status < 500:
                        return None
                    await asyncio.sleep((2 ** attempt) + random.uniform(0, 1))
                    
        except Exception as e:
            log.warning("Exception during API call for style %s (Attempt %d): %s: %s", style, attempt + 1, type(e).__name__, str(e))
            await asyncio.sleep((2 ** attempt) + random.uniform(0, 1))
            
    log.error("Max retries exceeded for style %s", style)
    return None
