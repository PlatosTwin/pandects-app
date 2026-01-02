from __future__ import annotations

import queue
from threading import Event, Thread
from typing import Callable


class AsyncTaskRunner:
    def __init__(self, *, app, max_queue_size: int = 100) -> None:
        self._app = app
        self._queue: queue.Queue[Callable[[], None]] = queue.Queue(max_queue_size)
        self._stop_event = Event()
        self._thread = Thread(target=self._run, name="async-tasks", daemon=True)
        self._thread.start()

    def enqueue(self, task: Callable[[], None]) -> bool:
        if self._stop_event.is_set():
            return False
        try:
            self._queue.put_nowait(task)
        except queue.Full:
            return False
        return True

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._queue.put_nowait(lambda: None)
        except queue.Full:
            pass

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                task = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if task is None:
                continue
            with self._app.app_context():
                try:
                    task()
                except Exception:
                    self._app.logger.exception("Async task failed.")
