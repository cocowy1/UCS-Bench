# Roadmap

## Done

- ✅ **v0.3** — UCS-Bench 4-dimension evaluation, place induction, ego edges, depth-based reachability, semantic embedding fusion, SQLite incremental store, real-model adapters (DA3, YOLO-World, SAM 2).
- ✅ **v0.4** — Video / frame-stream ingest with `directme ingest`, async incremental mapper with backpressure + chunk fault isolation + SQLite-backed resumability, CLIP-driven keyframe diversity selection, pose-drift telemetry.

## Next (v0.5 candidates, ranked by deployment value)

1. **Loop closure for cross-chunk pose drift.** Keyframe-embedding place
   recognition + a small pose-graph backend (g2o or GTSAM). Closes the
   gap left open by v0.4's measure-don't-correct stance — see
   `docs/algorithm_notes.md` § "Why we measure drift but don't correct it".
2. **Cross-view re-identification for movable objects.** Today the
   distance gate prevents long-range re-ID; a "search far if appearance
   matches" path would let DirectMe answer "where did I last put my
   keys" across a multi-room walk.
3. **Multi-reader / multi-writer concurrency around the scene graph.**
   Required for in-process online QA during ingest. Currently scoped out
   of v0.4 — recommended workaround is between-commit reads or a
   separate QA process. A double-buffered swap design is sketched in
   `directme/mapping/async_engine.py`'s docstring.
4. **Vector search over node embeddings** (FAISS / Annoy / hnswlib) for
   open-ended language queries that don't have a tight label match.
5. **Privacy-preserving graph export** — face / license-plate redaction
   on saved keyframes; per-node access ACLs.
6. **FastAPI streaming service** wrapping `AsyncIncrementalMapper` for
   network-fed wearables.

## Deliberately not on the roadmap

- **Frame-by-frame realtime ingest at 30 FPS.** DirectMe is built for
  1-FPS egocentric capture; raising the input rate doesn't improve
  graph quality enough to be worth the throughput cost.
- **Reconstruction-quality global SfM.** Out of scope; the upstream
  perception backbones (SCAL3R / DA3 / MASt3R) are the place to add
  this.
