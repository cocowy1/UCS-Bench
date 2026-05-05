"""Tests for AsyncIncrementalMapper (v0.4)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from directme.config import DirectMeConfig
from directme.mapping.async_engine import (
    AsyncIncrementalMapper,
    ingest_frames_async,
)
from directme.mapping.offline_engine import OfflineMappingEngine
from directme.perception.toy import build_living_room_kitchen_demo


def _build_engine(tmp: Path, *, backend: str = "json") -> OfflineMappingEngine:
    cfg = DirectMeConfig()
    cfg.run_dir = str(tmp)
    cfg.storage.backend = backend
    frames, percep_backend = build_living_room_kitchen_demo(tmp / "kf")
    eng = OfflineMappingEngine(backend=percep_backend, config=cfg)
    return eng, frames


def test_async_mapper_processes_all_frames_and_flushes_partial_chunk(tmp_path: Path):
    eng, frames = _build_engine(tmp_path)
    # Toy demo has 4 frames; chunk_size=3 forces a partial trailing chunk.
    mapper = asyncio.run(
        ingest_frames_async(eng, frames, chunk_size=3)
    )
    assert mapper.n_frames_processed == 4
    assert mapper.n_chunks_committed == 2  # 3 + 1
    assert mapper.failed_chunks == []
    assert eng.graph is not None and len(eng.graph.nodes) >= 2


def test_async_mapper_isolates_chunk_failure_and_keeps_running(tmp_path: Path):
    """A backend that raises on a specific chunk_id must not kill the worker.

    We patch process_chunk to raise on chunk_id=0 and succeed otherwise.
    """
    eng, frames = _build_engine(tmp_path)
    real_process = eng.process_chunk
    seen = []

    def flaky(frames_, chunk_id):
        seen.append(chunk_id)
        if chunk_id == 0:
            raise RuntimeError("synthetic perception failure on first chunk")
        return real_process(frames_, chunk_id)

    eng.process_chunk = flaky  # type: ignore[assignment]

    mapper = asyncio.run(
        ingest_frames_async(eng, frames, chunk_size=2, swallow_chunk_failures=True)
    )
    # Two chunks total; first fails, second succeeds.
    assert len(mapper.failed_chunks) == 1
    assert mapper.failed_chunks[0].chunk_id == 0
    assert mapper.failed_chunks[0].error_type == "RuntimeError"
    assert mapper.n_chunks_committed == 1


def test_async_mapper_fail_fast_propagates_first_exception(tmp_path: Path):
    eng, frames = _build_engine(tmp_path)

    def boom(frames_, chunk_id):
        raise RuntimeError("kaboom")

    eng.process_chunk = boom  # type: ignore[assignment]

    raised = False
    try:
        asyncio.run(
            ingest_frames_async(eng, frames, chunk_size=2, swallow_chunk_failures=False)
        )
    except RuntimeError as e:
        raised = True
        assert "kaboom" in str(e)
    assert raised, "swallow_chunk_failures=False must re-raise"


def test_async_mapper_resumes_from_committed_chunk_id(tmp_path: Path):
    """When the SQLite store reports a last_committed_chunk_id, the second
    run must not re-process those chunks."""
    eng, frames = _build_engine(tmp_path, backend="sqlite")

    # First run: process all frames.
    asyncio.run(ingest_frames_async(eng, frames, chunk_size=2))

    # Verify SQLite store recorded progress.
    last = eng.store.get_last_committed_chunk_id()  # type: ignore[union-attr]
    assert last is not None and last >= 0

    # Second run: same store, simulate restart with fresh engine instance.
    eng2, frames2 = _build_engine(tmp_path, backend="sqlite")
    mapper2 = asyncio.run(ingest_frames_async(eng2, frames2, chunk_size=2))
    assert mapper2._resume_from_chunk_id == last + 1
    # All chunks were already committed, so this run processes nothing new.
    # (chunk_id=0,1 are skipped; there are no new ones.)
    assert mapper2.n_chunks_committed == 0


def test_async_mapper_records_drift_telemetry_in_graph(tmp_path: Path):
    eng, frames = _build_engine(tmp_path)
    asyncio.run(ingest_frames_async(eng, frames, chunk_size=2))
    drift = eng.graph.metadata.get("drift_telemetry")  # type: ignore[union-attr]
    assert drift is not None
    assert drift["n_chunks_seen"] >= 1
    assert "warnings" in drift
    assert "cumulative_translation_m" in drift


def test_async_mapper_backpressure_does_not_drop_frames(tmp_path: Path):
    """Even with a tiny queue, every frame submitted must be processed."""
    eng, frames = _build_engine(tmp_path)
    # queue_maxsize=1 forces submit() to await every iteration.

    async def driver():
        m = AsyncIncrementalMapper(engine=eng, chunk_size=2, queue_maxsize=1)

        async def producer():
            for f in frames:
                await m.submit(f)
            await m.close()

        await asyncio.gather(producer(), m.run())
        return m

    mapper = asyncio.run(driver())
    assert mapper.n_frames_processed == 4
