import os
import json
import logging
import asyncio
import tempfile
from typing import Dict, List

from ingestor.pipeline import ingest_video, save_manifest
from ingestor.config import IngestorConfig
from extractor.pipeline import async_process_manifest
from extractor.config import ExtractorConfig
from compressor.pipeline import async_compress_timeline
from compressor.config import CompressorConfig
from synthesizer.pipeline import async_synthesize
from synthesizer.config import SynthesizerConfig

REQUIRED_STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
log = logging.getLogger(__name__)

async def _run_full_pipeline(video_path: str, styles: List[str]) -> Dict[str, str]:
    with tempfile.TemporaryDirectory() as workdir:
        # Ingestor
        frames_dir = os.path.join(workdir, "frames")
        ingest_config = IngestorConfig(frames_output_dir=frames_dir)
        log.info("Starting Phase 1: Ingestion for %s", video_path)
        chunks = ingest_video(video_path, ingest_config)
        manifest_path = os.path.join(workdir, "manifest.json")
        save_manifest(chunks, manifest_path)
        
        # Extractor
        extracted_path = os.path.join(workdir, "extracted.json")
        extractor_config = ExtractorConfig()
        log.info("Starting Phase 2: Extraction")
        await async_process_manifest(manifest_path, extracted_path, extractor_config)
        
        # Compressor
        compressed_path = os.path.join(workdir, "compressed.json")
        compress_config = CompressorConfig()
        log.info("Starting Phase 3: Compression")
        await async_compress_timeline(extracted_path, compressed_path, compress_config)
        
        # Synthesizer
        synthesized_path = os.path.join(workdir, "synthesized.json")
        synth_config = SynthesizerConfig()
        log.info("Starting Phase 4: Synthesis")
        await async_synthesize(compressed_path, frames_dir, synthesized_path, synth_config)
        
        # Parse final output
        if os.path.exists(synthesized_path):
            with open(synthesized_path, "r") as f:
                data = json.load(f)
                captions = data.get("captions", {})
                return {style: captions.get(style, "") for style in styles}
        
    log.warning("Pipeline completed but no synthesizer output found.")
    return {style: "" for style in styles}

def generate_captions(video_path: str, styles: List[str]) -> Dict[str, str]:
    """
    Executes the entire multi-agent video captioning pipeline.
    """
    try:
        return asyncio.run(_run_full_pipeline(video_path, styles))
    except Exception as e:
        log.error("Pipeline failed for %s: %s", video_path, e, exc_info=True)
        return {style: "" for style in styles}
