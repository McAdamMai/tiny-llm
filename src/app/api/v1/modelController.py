import asyncio
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Annotated, Union

from app.schemas.schemas import GenerateRequest, GenerateResponse, GenerateStreamChunk
from app.core.modelManager import ModelManager, get_manager
from app.utils.streaming import stream_generator

router = APIRouter(prefix="/v1", tags=["chat"])

# Type alias
BrainManager = Annotated[ModelManager, Depends(get_manager)]

@router.get("/health/live")
async def liveness_probe():
    # Fake liveness probe
    return {"status": "online"}

# Even though the code inside the router is simple, keeping the route itself async is a best practice in FastAPI
@router.get("/health/ready")
async def readiness_probe(manager: BrainManager):
    if not manager.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model Manager is still loading in VRAM"
        )
    return {"status": "ready"}

@router.post("/completions", response_model=None)
async def generate_completion(
    request: GenerateRequest, 
    manager: BrainManager
) -> Union[GenerateResponse, StreamingResponse]:
    
    try:
        # === BRANCH 1: STREAMING ===
        if request.stream:
            # Get the iterator from your manager (the one yielding raw tokens)
            raw_iterator = await manager.generate_iterator(
                model_id=request.model,
                prompt=request.prompt,
                max_tokens=request.max_new_tokens
            )

            # Define a helper to format tokens into Server-Sent Events (SSE)
            async def sse_generator():
                async for token in raw_iterator:
                    # Wrap the raw string in a JSON object for the client
                    chunk_data = GenerateStreamChunk(text=token).model_dump_json()
                    yield f"data: {chunk_data}\n\n"
                
                # Signal the end of the stream
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                sse_generator(), # Use this helper to generate json iterator
                media_type="text/event-stream"
            )

        # === BRANCH 2: NON-STREAMING ===
        else:
            output = await manager.generate_completion(
                model_id=request.model,
                prompt=request.prompt,
                max_tokens=request.max_new_tokens
            )
            
            # Return the Pydantic model directly
            return GenerateResponse(
                model=request.model,
                text=output["text"],
                usage=output["usage"]
            )

    except asyncio.QueueFull:
        raise HTTPException(status_code=503, detail="Server busy")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

"""# --- Inference Logic ---
@router.post("/chat/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest, manager: BrainManager):
    if request.stream:
        return StreamingResponse(
            stream_generator(
                modelManager=manager,       # Pass the manager instance
                model_id=request.model,     # Pass the model ID from request
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                raw_mode=False
            ),
            media_type="text/event-stream"
        )
        
    # Non-streaming
    output = await manager.generate(
            model_id=request.model,
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens
        )
    return {
            "model": request.model, 
            "text": output["text"],
            "usage": output["usage"]
        }
"""
# --- Resource Management ---

@router.delete("/manage/unload/{model_id}")
async def unload_model(model_id: str, manager: BrainManager):
    try:
        manager.unload_model(model_id)
        return {"detail": f"Model {model_id} unloaded successfully."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )