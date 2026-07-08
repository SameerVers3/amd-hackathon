import base64
import io
import logging
from typing import List
from PIL import Image
import os

from .config import ExtractorConfig

log = logging.getLogger(__name__)


def _image_to_base64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def _downscale_if_needed(image_path: str, force_downscale: bool, max_size: tuple) -> str:
    if not os.path.exists(image_path):
        log.warning("Image path not found: %s", image_path)
        return ""
        
    try:
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
                
            if force_downscale:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
            return _image_to_base64(img)
    except Exception as e:
        log.error("Failed to process image %s: %s", image_path, e)
        return ""


async def prep_images(visual_data: List[dict], config: ExtractorConfig) -> List[str]:
    num_frames = len(visual_data)
    force_downscale = num_frames > config.downscale_threshold
    
    if force_downscale:
        log.debug("Chunk has %d frames (> %d) - triggering dynamic downscaling to %s", 
                  num_frames, config.downscale_threshold, config.downscale_resolution)
                  
    b64_frames = []
    for frame_data in visual_data:
        b64 = _downscale_if_needed(
            frame_data["image_path"], 
            force_downscale, 
            config.downscale_resolution
        )
        if b64:
            b64_frames.append(b64)
            
    return b64_frames
