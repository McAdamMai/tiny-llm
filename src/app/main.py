# main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.modelManager import get_manager
from app.api.v1.modelController import router as chat_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP LOGIC ---
    print("System starting...")
    
    # Get the singleton instance
    manager = get_manager()

    # start_background_tasks -> queue create asyncio -> _worker() -> _event.wait()
    manager.start_background_tasks() # <--- Starts the Queue Worker
    
    # OPTIONAL: Only warm up if you want to trade startup time for user latency
    # You might want to wrap this in an "if settings.ENABLE_WARMUP:" check
    try:
        await manager.warmup_models("gemma-3-1b")
    except Exception as e:
        print(f"Warmup failed (non-critical): {e}")
        
    yield # API is running here
    
    # --- SHUTDOWN LOGIC ---
    print("System shutting down...")
    # Clean up VRAM on exit (Good practice for local dev)
    manager.unload_model("gemma-3-1b")
    # manager -> queue -> _worker -> task.cancel()
    await manager.shutdown()

# Initialize FastAPI app with lifespan
app = FastAPI(lifespan=lifespan, title="TinyLLM Service")

# Register the Router
app.include_router(chat_router)