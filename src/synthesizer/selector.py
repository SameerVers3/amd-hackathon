import os
import re
import math
import base64
from typing import List, Dict

def find_closest_frame(target_sec: float, frames_dir: str) -> str:
    # Finds the base64 string of the frame closest to the target timestamp in the directory.
    if not os.path.exists(frames_dir):
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
        
    closest_file = None
    min_diff = float("inf")
    
    # Assuming frames are named like frame_0001.23.jpg or frame_1.2.jpg
    # Actually, Phase 1 might just use standard names. Let's find any numbers in filename.
    for filename in os.listdir(frames_dir):
        if not filename.endswith(".jpg"):
            continue
            
        match = re.search(r"(\d+(\.\d+)?)", filename)
        if match:
            frame_sec = float(match.group(1))
            diff = abs(frame_sec - target_sec)
            if diff < min_diff:
                min_diff = diff
                closest_file = os.path.join(frames_dir, filename)
                
    if not closest_file:
        raise FileNotFoundError(f"No valid frames found in {frames_dir}")
        
    from PIL import Image
    import io
    with Image.open(closest_file) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        # Downscale to prevent payload bloat
        img.thumbnail((768, 768), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

def select_anchor_frames(scenes: List[Dict], frames_dir: str) -> List[str]:
    # Selects the 4 anchor frames as per Phase 4 requirements and returns base64 list
    if not scenes:
        return []
        
    timestamps = []
    
    # 1. Start of Scene 0
    timestamps.append(scenes[0].get("time_start_sec", 0.0))
    
    # 2. Start of Middle Scene
    middle_idx = len(scenes) // 2
    timestamps.append(scenes[middle_idx].get("time_start_sec", 0.0))
    
    # 3. End of Final Scene
    timestamps.append(scenes[-1].get("time_end_sec", 0.0))
    
    # 4. Highest key_actions count
    highest_actions_scene = max(scenes, key=lambda s: len(s.get("key_actions", [])))
    timestamps.append(highest_actions_scene.get("time_start_sec", 0.0))
    
    b64_frames = []
    for ts in timestamps:
        b64_frames.append(find_closest_frame(ts, frames_dir))
        
    return b64_frames
