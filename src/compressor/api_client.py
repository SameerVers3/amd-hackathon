import asyncio
import json
import logging
import random
from typing import Optional

import aiohttp

from .config import CompressorConfig
from .schema import get_fireworks_schema

log = logging.getLogger(__name__)

def extract_json_after_think(response_text: str) -> Optional[dict]:
    # Slices the CoT response to extract the JSON payload after </think>
    think_end_tag = "</think>"
    
    if think_end_tag in response_text:
        json_str = response_text.split(think_end_tag, 1)[1].strip()
    else:
        # Fallback: if no think tag, assume the whole thing is JSON
        json_str = response_text.strip()
        log.warning("No </think> tag found in the response. Attempting to parse raw output.")
        
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        log.error("Failed to decode JSON after CoT truncation: %s\nPayload: %s", e, json_str[:200])
        return None


async def call_e4b_api(
    session: aiohttp.ClientSession,
    markup_text: str,
    config: CompressorConfig
) -> Optional[dict]:
    # Makes the HTTP call to Fireworks with jittered exponential backoff
    url = f"{config.fireworks_api_base}{config.chat_endpoint}"
    headers = {
        "Authorization": f"Bearer {config.fireworks_api_key}",
        "Content-Type": "application/json"
    }
    
    messages = [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": markup_text}
    ]
    
    payload = {
        "model": config.e4b_model,
        "messages": messages,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
        "response_format": get_fireworks_schema()
    }
    
    for attempt in range(config.max_retries):
        try:
            # We use a longer timeout since the context could be huge and CoT generation takes time
            async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data["choices"][0]["message"]["content"]
                    
                    parsed_json = extract_json_after_think(content)
                    if parsed_json is not None:
                        return parsed_json
                        
                    # If parsing failed, retry
                    log.warning("Valid JSON not found in response on attempt %d/%d", attempt + 1, config.max_retries)
                    
                elif response.status == 429:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    log.warning("HTTP 429 Too Many Requests. Retrying in %.2fs (Attempt %d/%d)", 
                                wait_time, attempt + 1, config.max_retries)
                    await asyncio.sleep(wait_time)
                    
                else:
                    text = await response.text()
                    log.error("API request failed with HTTP %d: %s", response.status, text)
                    if response.status >= 500:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        await asyncio.sleep(wait_time)
                    else:
                        return None
                        
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            log.warning("Network error: %s. Retrying in %.2fs (Attempt %d/%d)", 
                        type(e).__name__, wait_time, attempt + 1, config.max_retries)
            await asyncio.sleep(wait_time)
            
    log.error("Max retries exceeded for E4B API call.")
    return None
