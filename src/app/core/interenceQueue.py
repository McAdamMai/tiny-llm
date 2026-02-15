import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any

class InferenceQueue:
    def __init__(self, max_size: int = 100):
        self._queue = deque(maxlen=max_size)
        self._event = asyncio.Event()
        self.processing = False
        self._worker_task = None
        # We use a ThreadPool to ensure blocking LlamaCPP code never touches the Main Loop
        self._executor = ThreadPoolExecutor(max_workers=1) 

    # --- Lifecycle --- 
    def start_worker(self):
        if self._worker_task is None:
            self.processing = True
            self._worker_task = asyncio.create_task(self._worker_loop())
            print("--- Queue Worker Started ---")
    
    async def shutdown(self):
        """Gracefully shuts down the worker."""
        self.processing = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                print("--- Queue Worker Cancelled Successfully ---")
        # Shut down the ThreadPoolExecutor (Clean up threads)
        self._executor.shutdown(wait=False)
        print("--- Queue Worker Stopped ---")

    # --- Worker Logic --- 
    # Event and future-based queue processing to safely move data from background threads to the main loop.
    async def _worker_loop(self):
        # returns the loop that is currently executing that coroutine (Main Loop)
        loop = asyncio.get_running_loop()
        while self.processing:
            # Clear the queue first, if it's empty, wait until new item arrives. This avoids busy waiting and ensures we only wake up when there's work to do.
            self._event.clear()
            if not self._queue:
                await self._event.wait()
            # Now we have at least one item in the queue
            try:
                task_fn, future = self._queue.popleft()
            except IndexError:
                continue

            if future.cancelled():
                continue

            try:
                # This ensures "for token in iterator" happens in a background thread.
                result = await loop.run_in_executor(self._executor, task_fn)
                
                if not future.done():
                    future.set_result(result) # Task finished successfully
            except Exception as e:
                if not future.done():
                    future.set_exception(e)

    async def enqueue(self, task_fn: Callable[[], Any]):
        """
        Add a task to the queue. The task_fn must be a synchronous function.
        """
        if len(self._queue) >= self._queue.maxlen:
            raise asyncio.QueueFull("Queue full")
        
        # Create a future at PENDING stage
        future = asyncio.Future()
        self._queue.append((task_fn, future))
        # Notify the worker that a new task has arrived
        self._event.set()
        # We don't await future here because for streaming, the future only resolves
        # when the stream is *finished*.
        return future
    
    #---helper---
    def depth(self) -> int:
        """Returns current depth of the queue."""
        return len(self._queue)