# app/core/queue.py
import asyncio
from collections import deque
from typing import Callable, Any, Optional

class InferenceQueue:
    def __init__(self, max_size: int = 100):
        self._queue: deque = deque(maxlen=max_size)
        self._event = asyncio.Event() # Wakes up the worker when new items arrive
        self.processing = False
        self._worker_task = None

    def start_worker(self):
        if self._worker_task is None:
            # pop left from the queue and process
            self.processing = True
            self._worker_task = asyncio.create_task(self._worker())
            print("--- Queue Worker Started ---")

    def shutdown(self):
        """Gracefully shuts down the worker."""
        self.processing = False
        if self._worker_task:
            self._worker_task.cancel()
            print("--- Queue Worker Stopped ---")

    async def enqueue(self, task_fn):
        """
        Enqueue a task and wait for execution.
        Raises: asyncio.QueueFull if queue is full
        """
        if len(self._queue) >= self._queue.maxlen:
            raise asyncio.QueueFull("Inference queue is full")

        future = asyncio.Future() # Future as an empty box or a "claim check."
        self._queue.append((task_fn, future))
        self._event.set()  # Notify the _worker
        # Wait here until the worker fills the future
        return await future
    
    async def enqueue_stream(self, task_fn):
        """
        Special handling for generators. 
        The worker iterates the generator and pushes items to a private queue.
        """
        if len(self._queue) >= self._queue.maxlen:
            raise asyncio.QueueFull("Inference queue is full")

        # Create a channel (mini-queue) for the tokens
        stream_channel = asyncio.Queue()
        loop = asyncio.get_running_loop()
        future = asyncio.Future() 

        # Define the work the background worker will do
        def sync_worker_logic():
            try:
                # A. Execute the blocking model call to get the iterator
                # task_fn is now just "lambda: brain.model.create_completion(...)"
                sync_iterator = task_fn()

                # B. Iterate synchronously (Blocking!)
                for chunk in sync_iterator:
                    # Extract text (Data extraction logic moved here)
                    token = chunk["choices"][0]["text"]
                    
                    # C. Push to Async Queue safely
                    loop.call_soon_threadsafe(stream_channel.put_nowait, token)

            except Exception as e:
                loop.call_soon_threadsafe(stream_channel.put_nowait, e)
            finally:
                # Signal the end
                loop.call_soon_threadsafe(stream_channel.put_nowait, StopAsyncIteration)

        # Add this wrapper to the main queue
        self._queue.append((sync_worker_logic, future))
        self._event.set()

        # 4. Define the generator that the API CALLER will receive
        async def response_generator():
            while True:
                token = await stream_channel.get()
                if token is StopAsyncIteration:
                    break
                if isinstance(token, Exception):
                    raise token
                yield token

        return response_generator()

    # "Sync Core, Async Shell" pattern
    async def enqueue_stream_1(self, task_fn):
        """
        Special handling for generators. 
        The worker iterates the generator and pushes items to a private queue.
        """

        if len(self._queue) >= self._queue.maxlen:
            raise asyncio.QueueFull("Inference queue is full")

        # Create a channel (mini-queue) for the tokens
        stream_channel = asyncio.Queue() # main thread reads from this
        loop = asyncio.get_running_loop() # Capture the Main Loop

        # Worker_wrapper must be a standard def (Sync) so the thread actually executes the code.
        def worker_wrapper():
            print("--- Worker Started ---")
            try:
                # Execute the Blocking Brain logic
                sync_iterator = task_fn()
                print("--- Iterator Created ---")
                # The WORKER iterates, holding the semaphore lock
                for chunk in sync_iterator: # task_fn running on background thread
                    token = chunk["choices"][0]["text"]
                    print(f"Generated: {repr(token)}")
                    loop.call_soon_threadsafe(stream_channel.put_nowait, token)
                    
            except Exception as e:
                # Pass exception to consumer
                print(f"!!! CRASH IN WORKER: {e}")
                loop.call_soon_threadsafe(stream_channel.put_nowait, e)
            finally:
                # Signal "Done"
                print("--- Worker Finished ---")
                loop.call_soon_threadsafe(stream_channel.put_nowait, StopAsyncIteration)
        # Add this wrapper to the main queue
        # We don't use a future here because we return a generator immediately
        self._queue.append(worker_wrapper)
        self._event.set()

        # 4. Consumption (Main Thread)
        while True:
            token = await stream_channel.get()
            if token is StopAsyncIteration: break
            if isinstance(token, Exception): raise token
            yield token

    async def _worker(self):
        """Background worker that processes the queue."""
        while True:
            if not self._queue:
                self._event.clear()
                await self._event.wait()  # Wait until new items arrive (triggered by enqueue)
            
            if not self.processing:
                break  # Exit if shutdown signal is received

            try:
                # Get the oldest task
                task_fn, future = self._queue.popleft()
            except IndexError:
                continue  # Queue was empty, loop back and wait

            if future.cancelled():
                continue  # Skip cancelled tasks
            
            try:
                result = await task_fn()
                if not future.cancelled():
                    future.set_result(result)
            except Exception as e:
                if not future.cancelled():
                    future.set_exception(e)
    
    def depth(self) -> int:
        """Returns current depth of the queue."""
        return len(self._queue)
