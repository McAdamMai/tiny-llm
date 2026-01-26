# app/services/streaming.py
import json
from typing import Any

async def stream_generator(brain: Any, prompt:str, max_new_tokens:int, raw_model:bool=False):
    """
    Yields data in Server-Sent Events (SSE) format.
    Decoupled from FastAPI routers for better testing and reusability.
    """
    async for token in brain.generate_interator(prompt=prompt, max_new_tokens=max_new_tokens):
        if raw_model:
            chunk_date = {
                "choice": {
                    "text": token,
                    "finish_reason": None
                }
                "usage": None
            }
            else:
                # Chat format (standard)
                chunk_data = {
                    "text": token,
                    "finish_reason": None
                }
            yield f"data: {json.dumps(chunk_data)}\n\n"
            
    yield "data: [DONE]\n\n"