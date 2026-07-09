from typing import List

def serialize_chunks(chunks: List[dict]) -> str:
    # Flattens the parsed JSON array into a custom markup string
    lines = []
    
    for chunk in chunks:
        start = chunk.get("time_start_sec", 0.0)
        end = chunk.get("time_end_sec", 0.0)
        asr = chunk.get("asr_text", "").strip()
        
        facts = chunk.get("atomic_facts", {})
        
        vis = ", ".join(facts.get("visual_objects", []))
        act = ", ".join(facts.get("actions", []))
        setting = facts.get("setting", "").strip()
        lit = facts.get("camera_or_lighting", "").strip()
        
        header = f"[T:{start:.1f}-{end:.1f}] ASR: \"{asr}\""
        body = f"VIS: {vis} | ACT: {act} | SET: {setting} | LIT: {lit}"
        
        lines.append(header)
        lines.append(body)
        lines.append("---")
        
    return "\n".join(lines)
