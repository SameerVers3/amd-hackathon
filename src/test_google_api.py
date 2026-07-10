import asyncio
import json
import os
import aiohttp

async def test():
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("No GOOGLE_API_KEY")
        return
        
    url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gemma-2-27b-it",
        "messages": [
            {"role": "user", "content": "Return a JSON object with a key 'greeting' and value 'hello'"}
        ],
        "response_format": {"type": "json_object"}
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            print(resp.status)
            print(await resp.text())

asyncio.run(test())
