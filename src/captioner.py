import os
from typing import Dict, List

REQUIRED_STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]


def generate_captions(video_path: str, styles: List[str]) -> Dict[str, str]:
    # input is video path (local) and list of styles
    # output is dictionary mapping style to caption

    # pipeline goes here
    
    filename = os.path.basename(video_path)
    placeholder = {
        style: f"[PLACEHOLDER-{style}] caption for {filename} not yet implemented"
        for style in styles
    }
    return placeholder
