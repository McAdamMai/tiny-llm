# main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.modelManager import ModelManager  # Import the CLASS, not the function
from app.api.v1.modelController import router as chat_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    print("System starting...")
    
    # 1. Initialize the SINGLETON here
    manager = ModelManager()
    
    # 2. Attach it to the app state (The Bridge)
    app.state.manager = manager 

    # 3. Start the worker on THIS instance
    manager.start_background_tasks() 
    
    try:
        # Check env var in real app, hardcoded for now
        # await manager.warmup_models("gemma-3-1b")
        pass
    except Exception as e:
        print(f"Warmup failed: {e}")
        
    yield 
    
    # --- SHUTDOWN ---
    print("System shutting down...")
    manager.unload_model("gemma-3-1b")
    await manager.shutdown()

app = FastAPI(lifespan=lifespan, title="TinyLLM Service")
app.include_router(chat_router)