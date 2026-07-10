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

async def _evaluate_caption(
    session: aiohttp.ClientSession,
    caption: str,
    style: str,
    config: SynthesizerConfig
) -> dict:
    url = f"{config.fireworks_api_base}{config.chat_endpoint}"
    headers = {
        "Authorization": f"Bearer {config.fireworks_api_key}",
        "Content-Type": "application/json"
    }
    
    judge_prompt = (
        f"You are a strict LLM-Judge. Evaluate the following video caption based on the requested style: {style.upper()}.\n"
        "Criteria:\n"
        "1. Accuracy: Does it describe a realistic visual scene?\n"
        "2. Style Match: Does it perfectly match the tone without being overly long?\n"
        "3. Length: Is it 1-2 sentences?\n\n"
        f"Caption to evaluate:\n{caption}\n\n"
        "Respond ONLY with a valid JSON object: {\"score\": <integer 1-10>, \"feedback\": \"<your critique and instructions on how to improve it>\"}"
    )
    
    payload = {
        "model": "accounts/fireworks/models/glm-5p2",
        "messages": [{"role": "user", "content": judge_prompt}],
        "temperature": 0.1,
        "max_tokens": 500,
        "response_format": {"type": "json_object"}
    }
    
    try:
        async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status == 200:
                data = await response.json()
                content = data["choices"][0]["message"]["content"]
                return json.loads(content)
    except Exception as e:
        log.warning("Judge failed: %s", e)
        
    return {"score": 10, "feedback": ""}  # Fail open

async def call_31b_api(
    session: aiohttp.ClientSession,
    style: str,
    timeline_json: str,
    b64_frames: List[str],
    config: SynthesizerConfig
) -> Optional[str]:
    url = f"{config.fireworks_api_base}{config.chat_endpoint}"
    headers = {
        "Authorization": f"Bearer {config.fireworks_api_key}",
        "Content-Type": "application/json"
    }
    
    params = config.style_sampling.get(style, {"temperature": 0.5, "top_p": 0.9})
    base_system_prompt = build_system_prompt(style)
    user_message = build_multimodal_payload(timeline_json, b64_frames)
    
    best_caption = None
    best_score = -1
    
    # Self-Judge Loop (Max 3 iterations)
    for iteration in range(3):
        system_prompt = base_system_prompt
        
        payload = {
            "model": config.model_id,
            "messages": [{"role": "system", "content": system_prompt}] + user_message,
            "temperature": params["temperature"],
            "top_p": params["top_p"],
            "max_tokens": config.max_tokens
        }
        
        caption = None
        for attempt in range(config.max_retries):
            try:
                async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status == 200:
                        data = await response.json()
                        raw_content = data["choices"][0]["message"]["content"]
                        if raw_content:
                            caption = sanitize_caption(raw_content)
                            break
                    elif response.status == 429:
                        await asyncio.sleep((2 ** attempt) + random.uniform(0, 1))
            except Exception as e:
                log.warning("Exception during API call: %s", e)
                await asyncio.sleep((2 ** attempt) + random.uniform(0, 1))
                
        if not caption:
            log.error("Failed to generate caption for style %s", style)
            return best_caption
            
        # Call Self-Judge
        judge_result = await _evaluate_caption(session, caption, style, config)
        score = judge_result.get("score", 0)
        feedback = judge_result.get("feedback", "")
        
        log.info("Style: %s | Iteration: %d | Score: %d/10 | Feedback: %s", style, iteration+1, score, feedback)
        
        if score > best_score:
            best_score = score
            best_caption = caption
            
        if score >= 8:
            log.info("Caption reached target score %d/10. Early exiting judge loop.", score)
            break
            
        # Append feedback for next iteration
        user_message.append({"role": "assistant", "content": f"<caption>\n{caption}\n</bbox>"})
        user_message.append({"role": "user", "content": f"JUDGE FEEDBACK: Your previous caption scored {score}/10. Critique: {feedback}\nPlease generate a new, improved caption fixing these issues."})
        
    return best_caption
