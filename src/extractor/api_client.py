"""
Handles the Google AI Studio API integration using the OpenAI compatible endpoint.
"""

import asyncio
import json
import logging
import random
import time
from typing import List

import aiohttp

from .config import ExtractorConfig
from .schema import get_fireworks_schema

log = logging.getLogger(__name__)


class TokenBucketRateLimiter:
    """
    Local token bucket algorithm for TPM (Tokens Per Minute) limit enforcement.
    """
    def __init__(self, tpm_limit: int):
        self.capacity = tpm_limit
        self.tokens = tpm_limit
        self.last_update = time.monotonic()
        self.fill_rate = tpm_limit / 60.0  # tokens per second
        self.lock = asyncio.Lock()

    async def consume(self, amount: int):
        """Wait until `amount` tokens are available, then consume them."""
        while True:
            async with self.lock:
                now = time.monotonic()
                elapsed = now - self.last_update
                
                # Replenish
                self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
                self.last_update = now
                
                if self.tokens >= amount:
                    self.tokens -= amount
                    return
                
                # Calculate wait time if not enough tokens
                deficit = amount - self.tokens
                wait_time = deficit / self.fill_rate
            
            # Wait outside the lock
            log.debug("Rate limiter: waiting %.2fs for %d tokens", wait_time, amount)
            await asyncio.sleep(wait_time)


def build_prompt_messages(asr_text: str, b64_frames: List[str], chunk_id: int) -> List[dict]:
    """
    Constructs the multimodal payload structure using base64 encoded strings.
    """
    content = [
        {"type": "text", "text": f"Chunk ID: {chunk_id}"},
    ]
    
    for b64 in b64_frames:
        content.append({
            "type": "image_url", 
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
        
    content.append({
        "type": "text", 
        "text": f"Audio Transcript: {asr_text}\nExtract atomic facts into valid JSON."
    })
    
    return [{"role": "user", "content": content}]


async def call_api(
    session: aiohttp.ClientSession, 
    messages: List[dict], 
    config: ExtractorConfig,
    rate_limiter: TokenBucketRateLimiter,
    estimated_tokens: int,
    max_retries: int = 5
) -> dict | None:
    """
    Makes the HTTP call to Google AI Studio with jittered exponential backoff.
    Consumes estimated tokens before making the request.
    """
    url = f"{config.google_api_base}{config.chat_endpoint}"
    headers = {
        "Authorization": f"Bearer {config.google_api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": config.google_model,
        "messages": messages,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
        "response_format": {"type": "json_object"}
    }
    
    for attempt in range(max_retries):
        # Wait for TPM bucket capacity
        await rate_limiter.consume(estimated_tokens)
        
        try:
            async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data["choices"][0]["message"]["content"]
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        log.error("API returned invalid JSON string: %s", content)
                        return None
                        
                elif response.status == 429:
                    # Too Many Requests - hit rate limit
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    log.warning("HTTP 429 Too Many Requests. Retrying in %.2fs (Attempt %d/%d)", 
                                wait_time, attempt + 1, max_retries)
                    await asyncio.sleep(wait_time)
                    
                else:
                    text = await response.text()
                    log.error("API request failed with HTTP %d: %s", response.status, text)
                    
                    # For 5xx errors, also apply backoff
                    if response.status >= 500:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        await asyncio.sleep(wait_time)
                    else:
                        return None # Non-retriable error
                        
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            log.warning("Network error: %s. Retrying in %.2fs (Attempt %d/%d)", 
                        type(e).__name__, wait_time, attempt + 1, max_retries)
            await asyncio.sleep(wait_time)
            
    log.error("Max retries exceeded for API call.")
    return None
