import argparse
import asyncio
import json
import logging
import os

import aiohttp

from .config import CompressorConfig
from .api_client import call_e4b_api
from .serializer import serialize_chunks
from .schema import validate_payload

log = logging.getLogger(__name__)

async def async_compress_timeline(input_path: str, output_path: str, config: CompressorConfig):
    # Loads manifest, serializes it, calls E4B, validates, and saves JSON
    if not config.fireworks_api_key:
        raise ValueError("FIREWORKS_API_KEY is not set.")
        
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input manifest not found: {input_path}")
        
    with open(input_path, "r") as f:
        phase2_data = json.load(f)
        
    if not phase2_data:
        log.warning("Input manifest is empty.")
        return []
        
    log.info("Loaded %d atomic chunks from %s", len(phase2_data), input_path)
    
    # 1. Serialize
    markup_text = serialize_chunks(phase2_data)
    log.info("Serialized timeline to dense markup (%d chars)", len(markup_text))
    
    # 2. Call API (Single-shot for standard video length)
    # Using connection pooling even for single shot to reuse DNS/SSL if needed elsewhere
    connector = aiohttp.TCPConnector(limit_per_host=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        api_response = await call_e4b_api(session, markup_text, config)
        
    if not api_response:
        log.error("Failed to generate semantic compression.")
        return None
        
    # 3. Validate Output
    validated_payload = validate_payload(api_response)
    if not validated_payload:
        log.error("Failed to validate Final Deliverable Schema.")
        return None
        
    scenes = validated_payload.get("scenes", [])
    
    # 4. Save
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    with open(output_path, "w") as f:
        json.dump(validated_payload, f, ensure_ascii=False, indent=2)
        
    log.info("Successfully compressed timeline into %d scenes at %s", len(scenes), output_path)
    return validated_payload


def compress_timeline(input_path: str, output_path: str, config: CompressorConfig = None):
    # Sync wrapper to run the async compression pipeline
    if config is None:
        config = CompressorConfig()
    return asyncio.run(async_compress_timeline(input_path, output_path, config))


def main():
    parser = argparse.ArgumentParser(description="Semantic Orchestration")
    parser.add_argument("--input", required=True, help="Input JSON array")
    parser.add_argument("--output", default="output.json", help="Output compressed JSON")
    args = parser.parse_args()
    
    compress_timeline(args.input, args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
