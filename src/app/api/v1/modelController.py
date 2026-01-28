import asyncio
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import StreamingResponse
from typing import Annotated

from app.schemas.chat import GenerateRequest, GenerateResponse
from app.core.manager import ModelManager, get_manager
from app.services.streaming import stream_generator

router = APIRouter(prefix="/v1", tags=["chat"])

# Type alias
BrainManager = Annotated[ModelManager, Depends(get_manager)]

@router.get("/health/live")
async def liveness_probe():
    return {"status": "online"}

@router.get("/health/ready")
async def readiness_probe(manager: BrainManager):
    if not manager.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model Manager is still loading in VRAM"
        )
    return {"status": "ready"}

# --- Inference Logic ---

@router.post("/chat/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest, manager: ModelManager = Depends(get_model_manager)):
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

@router.post("/completion", response_model=GenerateResponse)
async def generate_completion(request: GenerateRequest, manager: BrainManager):
    """
    Raw text completion (Autocompletion/IDE style)
    """
    brain = await manager.get_model(request.model)
    
    if request.stream:
        return StreamingResponse(
            stream_generator(
                brain,
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                raw_mode=True 
            ),
            media_type="text/event-stream"
        )

    # Run blocking inference in executor
    loop = asyncio.get_running_loop() 
    output = await loop.run_in_executor(
        None,
        lambda: brain(request.prompt, max_tokens=request.max_new_tokens)
    )
    
    return {
        "model": request.model, 
        "text": output["choices"][0]["text"],
        "usage": output["usage"]
    }

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