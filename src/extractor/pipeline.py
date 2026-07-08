import argparse
import asyncio
import json
import logging
import os
import sys
from typing import List, Dict

import aiohttp

from .config import ExtractorConfig
from .api_client_fw import TokenBucketRateLimiter, build_prompt_messages, call_api
from .image_utils import prep_images
from .schema import validate_and_deduplicate

log = logging.getLogger(__name__)


async def process_chunk(
    session: aiohttp.ClientSession, 
    chunk: dict, 
    config: ExtractorConfig,
    rate_limiter: TokenBucketRateLimiter,
    semaphore: asyncio.Semaphore
) -> dict:
    async with semaphore:
        chunk_id = chunk.get("chunk_id", -1)
        log.info("Processing chunk_id %d", chunk_id)
        
        # prep images and text
        visual_data = chunk.get("visual_data", [])
        asr_text = chunk.get("asr_data", {}).get("text", "")
        
        b64_frames = await prep_images(visual_data, config)
        
        messages = build_prompt_messages(asr_text, b64_frames, chunk_id)
        
        if config.system_prompt:
            messages.insert(0, {"role": "system", "content": config.system_prompt})
            
        estimated_visual_tokens = chunk.get("metadata", {}).get("total_visual_tokens", 1000)
        estimated_prompt_tokens = len(asr_text.split()) + len(config.system_prompt.split()) + 50
        total_estimated = estimated_visual_tokens + estimated_prompt_tokens + config.max_tokens
        
        api_response = await call_api(session, messages, config, rate_limiter, total_estimated)
        
        atomic_facts = None
        if api_response:
            atomic_facts = validate_and_deduplicate(api_response)
            
        if atomic_facts is None:
            log.warning("Chunk %d failed validation or API call.", chunk_id)
            atomic_facts = {
                "visual_objects": [],
                "actions": [],
                "setting": "",
                "camera_or_lighting": "",
                "audio_visual_correlation": ""
            }
            
        return {
            "chunk_id": chunk_id,
            "time_start_sec": chunk.get("time_start_sec", 0.0),
            "time_end_sec": chunk.get("time_end_sec", 0.0),
            "asr_text": asr_text,
            "atomic_facts": atomic_facts
        }


async def async_process_manifest(manifest_path: str, output_path: str, config: ExtractorConfig):
    """
    Loads manifest, dispatches tasks concurrently, and writes the output.
    """
    if not config.fireworks_api_key:
        raise ValueError("FIREWORKS_API_KEY is not set.")
        
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    log.info("Loaded %d chunks from %s", len(manifest), manifest_path)
    
    rate_limiter = TokenBucketRateLimiter(config.tpm_limit)
    semaphore = asyncio.Semaphore(config.max_concurrent_requests)
    
    connector = aiohttp.TCPConnector(limit_per_host=50)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            process_chunk(session, chunk, config, rate_limiter, semaphore)
            for chunk in manifest
        ]
        
        results = await asyncio.gather(*tasks)
        
    results.sort(key=lambda x: x["chunk_id"])
    
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    with open(output_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    log.info("Successfully wrote %d extracted chunks to %s", len(results), output_path)
    return results


def process_manifest(manifest_path: str, output_path: str, config: ExtractorConfig = None):
    """
    Sync wrapper to start the asyncio event loop.
    """
    if config is None:
        config = ExtractorConfig()
    asyncio.run(async_process_manifest(manifest_path, output_path, config))


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Asynchronous JSON Extraction")
    parser.add_argument("--manifest", required=True, help="Input Phase 1 manifest JSON")
    parser.add_argument("--output", default="phase2_output.json", help="Output JSON path")
    args = parser.parse_args()
    
    process_manifest(args.manifest, args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
