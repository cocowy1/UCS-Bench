"""Asynchronous incremental mapping for DirectMe.

This is the deployment-friendly entry point for video / streaming ingest.
It wraps :class:`OfflineMappingEngine` in an async producer/consumer loop
with three properties the v0.3 version lacked:

1. **Backpressure.** The internal queue has a bounded ``maxsize``;
   :meth:`submit` blocks rather than letting the queue grow without limit.
   This matters when frame extraction (cheap) outruns chunk perception
   (expensive) — e.g. when an upstream wearable is push-streaming JPEGs
   and the GPU is saturated.

2. **Chunk-level fault isolation.** A single chunk that crashes inside
   the perception backend (NaN poses, OOM, model bug, corrupt frame) is
   caught, logged into ``failed_chunks``, and the worker keeps running on
   subsequent chunks. The graph is never left in a half-written state
   because :class:`OfflineMappingEngine` only commits a chunk after its
   internal fusion succeeds; a raise during fusion surfaces here.

3. **Resumability.** If the underlying store supports it (SQLite does),
   on restart the worker reads ``last_committed_chunk_id`` and skips chunks
   the previous run already processed. This makes long video ingests
   safe to interrupt.

Concurrency model
-----------------
This module **does not** add multi-reader / multi-writer locking around
the scene graph. Per the v0.4 design decision, the QA path is expected
to read the graph only between chunk commits (or via a separate read
of the saved graph file). Concurrent in-process retrieval during a chunk
write is out of scope. This keeps the implementation small and matches
the deployed reality where ingest and QA usually run as separate processes
talking through the saved graph file.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Iterable

from directme.mapping.offline_engine import MappingEvent, OfflineMappingEngine
from directme.perception.base import VideoFrame

log = logging.getLogger(__name__)


@dataclass
class FailedChunkRecord:
    """Diagnostic record for a chunk whose perception / fusion raised."""

    chunk_id: int
    n_frames: int
    error_type: str
    error_message: str
    first_frame_index: int
    first_frame_timestamp: float


@dataclass
class AsyncIncrementalMapper:
    """Bounded-queue async wrapper around :class:`OfflineMappingEngine`.

    Args:
        engine: a constructed :class:`OfflineMappingEngine`. The mapper
            calls ``engine.process_chunk`` directly; whatever store the
            engine is configured with handles persistence.
        chunk_size: frames per chunk. At 1 FPS, a chunk_size of 10–30
            corresponds to 10–30 seconds of capture and is the sweet
            spot for SCAL3R / DA3-class backends.
        queue_maxsize: bound on the in-memory frame queue. Defaults to
            ``4 * chunk_size``; lower values exert harder backpressure on
            the producer.
        skip_already_committed: if ``True`` (default) and the underlying
            store exposes ``get_last_committed_chunk_id``, chunks with
            ``chunk_id <= last_committed`` are dropped on the floor so a
            restarted run resumes from the next chunk.
        on_chunk_committed: optional async callback invoked after each
            successful chunk commit. Receives ``(chunk_id, events)``.
        swallow_chunk_failures: if ``True`` (default), a chunk that raises
            during processing is logged to ``failed_chunks`` and the
            worker continues. Set to ``False`` to re-raise and stop the
            worker on first failure (useful for tests / debugging).
    """

    engine: OfflineMappingEngine
    chunk_size: int = 10
    queue_maxsize: int | None = None
    skip_already_committed: bool = True
    on_chunk_committed: Callable[[int, list[MappingEvent]], Awaitable[None]] | None = None
    swallow_chunk_failures: bool = True

    # Runtime state
    queue: asyncio.Queue = field(init=False)
    events: list[MappingEvent] = field(default_factory=list, init=False)
    failed_chunks: list[FailedChunkRecord] = field(default_factory=list, init=False)
    n_chunks_committed: int = field(default=0, init=False)
    n_frames_processed: int = field(default=0, init=False)
    _resume_from_chunk_id: int = field(default=0, init=False)
    _stopped: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        maxsize = self.queue_maxsize if self.queue_maxsize is not None else max(2, 4 * self.chunk_size)
        self.queue = asyncio.Queue(maxsize=maxsize)
        # Resume support: query the store for the last committed chunk_id.
        if self.skip_already_committed:
            store = getattr(self.engine, "store", None)
            getter = getattr(store, "get_last_committed_chunk_id", None)
            if callable(getter):
                last = getter()
                if last is not None and last >= 0:
                    self._resume_from_chunk_id = int(last) + 1
                    log.info(
                        "AsyncIncrementalMapper resuming from chunk_id=%d "
                        "(store reports last_committed=%d).",
                        self._resume_from_chunk_id, last,
                    )

    # ------------------------------------------------------------------ #
    # Producer-side API
    # ------------------------------------------------------------------ #
    async def submit(self, frame: VideoFrame) -> None:
        """Enqueue a frame. Blocks if the queue is full (backpressure)."""
        if self._stopped:
            raise RuntimeError("AsyncIncrementalMapper has been closed.")
        await self.queue.put(frame)

    async def submit_iterable(self, frames: Iterable[VideoFrame]) -> None:
        """Convenience: feed an entire generator/list with backpressure."""
        for f in frames:
            await self.submit(f)

    async def close(self) -> None:
        """Signal end-of-stream. Worker will flush a final partial chunk."""
        self._stopped = True
        await self.queue.put(None)

    # ------------------------------------------------------------------ #
    # Consumer-side worker
    # ------------------------------------------------------------------ #
    async def run(self) -> None:
        """Run the consumer loop until ``close()`` is called.

        Pulls frames from the internal queue, batches them into chunks of
        ``chunk_size``, and calls ``engine.process_chunk`` on each batch.
        Failures in a single chunk do not stop the worker (unless
        ``swallow_chunk_failures=False``).
        """
        chunk: list[VideoFrame] = []
        chunk_id = 0
        while True:
            frame = await self.queue.get()
            if frame is None:
                if chunk:
                    await self._process_one_chunk(chunk_id, chunk)
                break
            chunk.append(frame)
            if len(chunk) >= self.chunk_size:
                await self._process_one_chunk(chunk_id, chunk)
                chunk = []
                chunk_id += 1

    async def _process_one_chunk(self, chunk_id: int, frames: list[VideoFrame]) -> None:
        # Resume: skip chunks already committed by a previous run.
        if chunk_id < self._resume_from_chunk_id:
            log.debug(
                "Skipping chunk_id=%d as already committed.", chunk_id
            )
            return

        first = frames[0]
        try:
            t0 = time.monotonic()
            # The actual perception / fusion / save work. This is offloaded
            # to a thread because process_chunk is sync CPU/GPU work and we
            # don't want it to block the event loop's other awaitables
            # (e.g. a progress reporter or the producer's submit() above).
            chunk_events = await asyncio.to_thread(
                self.engine.process_chunk, frames, chunk_id
            )
            elapsed = time.monotonic() - t0
        except Exception as exc:  # noqa: BLE001 — we deliberately catch all
            rec = FailedChunkRecord(
                chunk_id=chunk_id,
                n_frames=len(frames),
                error_type=type(exc).__name__,
                error_message=str(exc),
                first_frame_index=first.index,
                first_frame_timestamp=first.timestamp,
            )
            self.failed_chunks.append(rec)
            log.exception(
                "Chunk %d failed (frames %d-%d): %s",
                chunk_id, first.index, frames[-1].index, exc,
            )
            if not self.swallow_chunk_failures:
                raise
            return

        self.events.extend(chunk_events)
        self.n_chunks_committed += 1
        self.n_frames_processed += len(frames)
        log.info(
            "Chunk %d committed: %d frames, %d events, %.2fs.",
            chunk_id, len(frames), len(chunk_events), elapsed,
        )

        # Persist progress so a future restart can resume here.
        store = getattr(self.engine, "store", None)
        recorder = getattr(store, "record_progress", None)
        if callable(recorder):
            try:
                recorder(chunk_id, frames[-1].timestamp)
            except Exception:  # noqa: BLE001
                log.warning("record_progress failed for chunk %d", chunk_id, exc_info=True)

        if self.on_chunk_committed is not None:
            try:
                await self.on_chunk_committed(chunk_id, chunk_events)
            except Exception:  # noqa: BLE001
                log.warning("on_chunk_committed callback raised", exc_info=True)

    # ------------------------------------------------------------------ #
    # Telemetry
    # ------------------------------------------------------------------ #
    @property
    def stats(self) -> dict:
        """Lightweight dict for logging / observability."""
        return {
            "chunks_committed": self.n_chunks_committed,
            "frames_processed": self.n_frames_processed,
            "queue_size": self.queue.qsize(),
            "failed_chunks": len(self.failed_chunks),
            "resume_from_chunk_id": self._resume_from_chunk_id,
        }


# --------------------------------------------------------------------------- #
# Convenience: drive a synchronous frame iterator end-to-end through the async
# mapper. Most CLI users want this rather than wiring up an event loop manually.
# --------------------------------------------------------------------------- #
async def ingest_frames_async(
    engine: OfflineMappingEngine,
    frames: Iterable[VideoFrame],
    chunk_size: int = 10,
    queue_maxsize: int | None = None,
    swallow_chunk_failures: bool = True,
) -> AsyncIncrementalMapper:
    """Run a one-shot async ingest of ``frames`` and return the mapper.

    The mapper is returned so the caller can inspect ``events``,
    ``failed_chunks``, and ``stats``.
    """
    mapper = AsyncIncrementalMapper(
        engine=engine,
        chunk_size=chunk_size,
        queue_maxsize=queue_maxsize,
        swallow_chunk_failures=swallow_chunk_failures,
    )

    async def _producer() -> None:
        try:
            for f in frames:
                await mapper.submit(f)
        finally:
            await mapper.close()

    await asyncio.gather(_producer(), mapper.run())
    return mapper
