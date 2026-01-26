import asyncio
import os
import gc
from typing import AsyncGenerator, Dict, Any, Optional
from llama_cpp import Llama
from concurrent.futures import ThreadPoolExecutor

# --- The brain wrapper ---
class LlamaBrain:
    def __init__(self, model: Llama, model_id: str):
        self.model = model
        self.model_id = model_id
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def generate(self, prompt: str, max_new_tokens: int) -> Dict[str, Any]:
        """
        Non-streaming generation.
        Runs in a separate thread to avoid blocking the main event loop.
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
    
    async def get_model(self, model_id: str) -> LlamaBrain:
        """
        Retrieves a loaded model. If not loaded, safeguards VRAM and loads it.
        """
        if model_id not in self._models:
            async with self._load_lock:
                # Double-check after acquiring lock
                if model_id not in self._models:
                    await self.safe_load(model_id)
        return self._models[model_id]

    async def safe_load(self, model_id: str):
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

    def unload_model(self, model_id: str):
        """Frees VRAM by unloading the model."""
        if model_id in self._models:
            print(f"---  Unloading model: {model_id} ---")
            # Deleting the Llama object triggers C++ destructor
            del self._models[model_id].model 
            del self._models[model_id]
            gc.collect()
            print(f"--- {model_id} unloaded successfully ---")

    async def warmup_models(self, model_id: str = "gemma-3-1b"):
        print(f"--- Warming up engine: {model_id} ---")
        try:
            brain = await self.get_model(model_id)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: brain.model.create_completion(prompt="Warmup", max_tokens=1)
            )
            print(f"--- {model_id} is hot and ready to serve! ---")
        except Exception as e:
            print(f"Warmup failed: {e}")

    def is_ready(self) -> bool:
        return len(self._models) > 0

# Singleton
manager_instance = ModelManager()

def get_manager() -> ModelManager:
    return manager_instance