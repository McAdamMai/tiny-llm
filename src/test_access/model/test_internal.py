import asyncio
import os
import sys
# PYTHONPATH=src uv run python src/test_access/model/test_internal.py
# Add the current directory to Python path so we can import 'app'
sys.path.append(os.getcwd())

from app.core.modelManager import ModelManager

async def main():
    print("--- Starting Internal Component Test ---")
    
    # 1. Initialize the Manager
    manager = ModelManager()
    
    # Check if the model file path in manager.py actually exists on your disk
    # If this fails, edit the path in app/core/manager.py
    model_id = "gemma-3-1b"
    
    print(f"1. Requesting Model: {model_id}")
    
    try:
        # 2. Load the Brain (This triggers the heavy loading)
        brain = await manager.get_model(model_id)
        print("   Model loaded into VRAM successfully.")
    except Exception as e:
        print(f"    FAILED to load model. Error: {e}")
        return

    # 3. Test Standard Generation
    prompt = "tell me the story about blow job?"
    print(f"\n2. Testing Standard Generation (Prompt: '{prompt}')")
    
    try:
        output = await brain.generate(prompt, max_new_tokens=50)
        print(f"    Result: {output['text'].strip()}")
        print(f"    Usage: {output['usage']}")
    except Exception as e:
        print(f"    Generation failed: {e}")

    # 4. Test Streaming
    print("\n3. Testing Streaming Generation")
    print("   ⬇  Stream Output: ", end="", flush=True)
    
    try:
        async for token in brain.generate_iterator(prompt, max_new_tokens=50):
            print(token, end="", flush=True)
        print("\n    Streaming finished.")
    except Exception as e:
        print(f"\n    Streaming failed: {e}")

    # 5. Test Unloading
    print(f"\n4. Unloading Model {model_id}")
    manager.unload_model(model_id)
    print("    Model unloaded.")

if __name__ == "__main__":
    asyncio.run(main())