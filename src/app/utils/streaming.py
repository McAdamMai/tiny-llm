# app/services/streaming.py
import json
from typing import Any
from app.core.modelManager import ModelManager

async def stream_generator(modelManager: ModelManager, prompt:str, max_new_tokens:int, raw_model:bool=False):
    """
    Yields data in Server-Sent Events (SSE) format.
    Decoupled from FastAPI routers for better testing and reusability.
    """

    # 1. Call the manager (ensure generate_iterator is the "queue-safe" version)
    iterator = modelManager.generate_iterator(
        model_id=model_id, 
        prompt=prompt, 
        max_tokens=max_new_tokens
    )

    async for token in iterator:
        if raw_model:
            chunk_data = {
                "choice": {
                    "text": token,
                    "finish_reason": None
                },
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