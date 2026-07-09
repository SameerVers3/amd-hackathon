import argparse
import asyncio
import json
import logging
import os

import aiohttp

from .config import SynthesizerConfig
from .api_client import call_31b_api
from .selector import select_anchor_frames

log = logging.getLogger(__name__)

async def async_synthesize(input_path: str, frames_dir: str, output_path: str, config: SynthesizerConfig):
    # Main orchestrator for generating 4 style captions concurrently
    if not config.fireworks_api_key:
        raise ValueError("FIREWORKS_API_KEY is not set.")
        
    with open(input_path, "r") as f:
        phase3_data = json.load(f)
        
    scenes = phase3_data.get("scenes", [])
    if not scenes:
        log.warning("No scenes found in input.")
        return
        
    timeline_str = json.dumps(scenes, indent=2)
    
    # 1. Select Anchor Frames
    try:
        b64_frames = select_anchor_frames(scenes, frames_dir)
        log.info("Selected %d anchor frames.", len(b64_frames))
    except Exception as e:
        log.error("Failed to extract anchor frames: %s", e)
        return
        
    # 2. Concurrent Dispatch
    styles = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
    
    connector = aiohttp.TCPConnector(limit_per_host=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            call_31b_api(session, style, timeline_str, b64_frames, config)
            for style in styles
        ]
        results = await asyncio.gather(*tasks)
        
    # 3. Assemble Payload
    captions = {style: result for style, result in zip(styles, results) if result}
    
    if len(captions) < len(styles):
        log.warning("Some styles failed to generate.")
        
    final_payload = {
        "video_id": os.path.basename(frames_dir) or "unknown_video",
        "captions": captions
    }
    
    # 4. Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final_payload, f, indent=2)
        
    log.info("Successfully generated captions at %s", output_path)

def synthesize_captions(input_path: str, frames_dir: str, output_path: str, config: SynthesizerConfig = None):
    # Sync wrapper
    if config is None:
        config = SynthesizerConfig()
    asyncio.run(async_synthesize(input_path, frames_dir, output_path, config))

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Synthesizer")
    parser.add_argument("--input", required=True, help="Input Phase 3 JSON")
    parser.add_argument("--frames-dir", required=True, help="Directory containing Phase 1 frames")
    parser.add_argument("--output", default="phase4_output.json", help="Output final JSON")
    args = parser.parse_args()
    
    synthesize_captions(args.input, args.frames_dir, args.output)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
