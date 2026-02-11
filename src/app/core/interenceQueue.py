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
        future = asyncio.Future() # Future as an empty box or a "claim check."
        # Define the work the background worker will do
        # Oridinarily worker just returns a string but wrapper will push tokens to the channel
        async def worker_wrapper():
            try:
                # The WORKER iterates, holding the semaphore lock
                async for token in task_fn():
                    await stream_channel.put(token) 
            except Exception as e:
                # Pass exception to consumer
                await stream_channel.put(e)
            finally:
                # Signal the end
                await stream_channel.put(StopAsyncIteration)

        # Add this wrapper to the main queue
        # We don't use a future here because we return a generator immediately
        self._queue.append((worker_wrapper, future)) 
        self._event.set()

        # Return a generator that reads from the channel
        while True:
            token = await stream_channel.get()
            if token is StopAsyncIteration:
                break
            if isinstance(token, Exception):
                raise token
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
