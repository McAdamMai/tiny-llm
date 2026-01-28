# app/core/queue.py
import asyncio
from collections import deque
from typing import Dict
class InferenceQueue:
    def __init__(self, max_size: int = 100):
        self._queue: deque = deque(maxlen=max_size)
        self._event = asyncio.Event() # Wakes up the worker when new items arrive
        self.processing = False
        asyncio.create_task(self._worker())


    async def enqueue(self, task_fn):
        """
        Enqueue a task and wait for execution.
        Raises: asyncio.QueueFull if queue is full
        """
        if len(self.queue) >= self._queue.maxlen:
            raise asyncio.QueueFull("Inference queue is full")

        future = asyncio.Future() # Future as an empty box or a "claim check."
        self._queue.append((task_fn, future))
        self._event.set()  # Notify the worker
        # Wait here until the worker fills the future
        return await future

    async def enqueue_stream(self, task_fn):
        """
        Special handling for generators. 
        The worker iterates the generator and pushes items to a private queue.
        """

        if len(self.queue) >= self._queue.maxlen:
            raise asyncio.QueueFull("Inference queue is full")

        # Create a channel (mini-queue) for the tokens
        stream_channel = asyncio.Queue()
        # Define the work the background worker will do
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
        self._queue.append((worker_wrapper, asyncio.Future())) 
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
                await self._event.wait()  # Wait until new items arrive
            
            # Get the oldest task
            task_fn, future = self._queue.popleft()

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