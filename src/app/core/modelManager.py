import asyncio
import os
import gc
from typing import AsyncGenerator, Dict, Any, Optional
from llama_cpp import Llama
from concurrent.futures import ThreadPoolExecutor
from .interenceQueue import InferenceQueue
from fastapi import Request

# --- The brain wrapper ---
class LlamaBrain:
    def __init__(self, model: Llama, model_id: str):
        self.model = model
        self.model_id = model_id
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def generate(self, prompt: str, max_new_tokens: int) -> Dict[str, Any]:
        """
        Async wrapper that offloads blocking C++ inference to a thread.
        This method returns a Coroutine, which fits perfectly with the Queue.
        """
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self._executor,
            lambda: self.model.create_completion(
                prompt=prompt,
                max_tokens=max_new_tokens,
                stop=["<eos>", "<|endoftext|>"], 
                echo=False
            )
        )

        return {
            "text": result["choices"][0]["text"],
            "usage": result["usage"]
        }
    
    async def generate_iterator(self, prompt: str, max_new_tokens: int) -> AsyncGenerator[str, None]:
        """
        Async generator that yields tokens one by one.
        """
        loop = asyncio.get_running_loop()
        stream_iterator = await loop.run_in_executor(
            self._executor,
            lambda: self.model.create_completion(
                prompt=prompt,
                max_tokens=max_new_tokens,
                stream=True,
                echo=False
            )
        )
        for chunk in stream_iterator:
            # Yielding will freeze the main thread for milliseconds, 
            #asyncio.sleep(0) allows other tasks to run while waiting for the next token.
            await asyncio.sleep(0)  # Yield control to event loop
            token = chunk["choices"][0]["text"]
            yield token

    def __call__(self, prompt: str, max_tokens: int):
        """Allows the instance to be called like a function (for raw completion endpoint)"""
        return self.model.create_completion(prompt=prompt, max_tokens=max_tokens)


# --- The Model Manager ---
class ModelManager:
    def __init__(self):
        self._models: Dict[str, LlamaBrain] = {}
        self.model_registry = {
            "gemma-3-1b": "models/gemma-3-1b/google_gemma-3-1b-it-Q4_K_M.gguf"
        }
        self._load_lock = asyncio.Lock()
        # TBD add max queue size to config
        self._queue = InferenceQueue(max_size=100)

    # --- PUBLIC API ---
    # no needs to be asynchronized
    def start_background_tasks(self):
        self._queue.start_worker()

    # When called, it will free VRAM
    def unload_model(self, model_id: str):
        """Frees VRAM by unloading the model."""
        if model_id in self._models:
            print(f"---  Unloading model: {model_id} ---")
            # Deleting the Llama object triggers C++ destructor
            del self._models[model_id].model 
            del self._models[model_id]
            gc.collect()
            print(f"--- {model_id} unloaded successfully ---")
    
    def is_ready(self) -> bool:
        return len(self._models) > 0

    # Default model_id is set to "gemma-3-1b"
    async def warmup_models(self, model_id: str = "gemma-3-1b"):
        print(f"--- Warming up engine: {model_id} ---")
        try:
            brain = await self._get_model(model_id)
        except Exception as e:
            print(f"Warmup failed: {e}")

    async def shutdown(self):
        """Gracefully shuts down the worker."""
        await self._queue.shutdown()
    
    async def generate_completion(self, model_id: str, prompt: str, max_tokens: int):
        brain = await self._get_model(model_id)
        # use lambda to defer execution, it packs the function and its arguments and send to the queue. worker will call it later (other threads).
        return await self._queue.enqueue(
            lambda: brain.generate(prompt, max_tokens)
        )

    async def generate_iterator(self, model_id: str, prompt: str, max_tokens: int) -> AsyncGenerator[str, None]:
        brain = await self._get_model(model_id)
        generator =  self._queue.enqueue_stream(
            lambda: brain.generate_iterator(prompt, max_tokens)
        )
        async for token in generator:
            yield token

    async def generate_chat_completion(self, model_id:str, prompt: str, max_tokens: int):
        return None  # TBD

    # --- INTERNAL METHOD ---
    
    async def _get_model(self, model_id: str) -> LlamaBrain:
        """
        Retrieves a loaded model. If not loaded, safeguards VRAM and loads it.
        """
        if model_id not in self._models:
            async with self._load_lock:
                # Double-check after acquiring lock
                if model_id not in self._models:
                    await self._safe_load(model_id)
        return self._models[model_id]

    async def _safe_load(self, model_id: str):
        """Ensures VRAM is clear before loading a new heavy model."""
        print(f"---  Safe Loading: {model_id} ---")
        
        # 1. Unload others first
        current_models = list(self._models.keys())
        for existing_model in current_models:
            if existing_model != model_id:
                self.unload_model(existing_model)

        # 2. Load the requested model if missing
        if model_id not in self._models:
            self._models[model_id] = await self._load_model(model_id)

    async def _load_model(self, model_id: str) -> LlamaBrain:
        if model_id not in self.model_registry:
            raise ValueError(f"Model {model_id} not found in registry.")
        
        model_path = self.model_registry[model_id]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file for {model_id} not found at {model_path}.")
        
        print(f"--- Loading model from {model_path} ---")

        loop = asyncio.get_running_loop()
        try: 
            llama_instance = await loop.run_in_executor(
                None, lambda: Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    n_threads=6,
                    n_gpu_layers=-1, 
                    verbose=True
                )
            )
            return LlamaBrain(llama_instance, model_id)
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise e


# Singleton
manager_instance = ModelManager()

def get_manager(request: Request):
    return request.app.state.manager
