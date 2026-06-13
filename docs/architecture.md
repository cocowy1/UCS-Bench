# Architecture

DirectMe is split into two phases that run independently of each other.

## Phase 1 — Offline asynchronous incremental mapping

```text
[VideoFrame stream]
       │
       ▼ (chunk of N frames, e.g. 10s @ 1 FPS)
PerceptionBackend.process_chunk()
       │   • per-frame T_local_from_camera (DA3 or Scal3R)
       │   • dense depth + intrinsics
       │   • open-vocab object detections + masks (YOLO-World + SAM 2)
       │   • optional CLIP/DINO semantic embeddings
       │   • optional appearance histograms
       ▼
ChunkPosePropagator.propagate()
       │   • validates each SE(3) (rejects NaN / non-orthogonal R)
       │   • rejects implausible per-frame translation jumps
       │   • on accept, sets T_world(t) = T_world_end ∘ inv(T_local_start) ∘ T_local(t)
       │   • on reject, leaves the world anchor untouched (ChunkReport logged)
       ▼
For each accepted frame, for each object:
       │   P_world = T_world(t) · P_cam
       │
       ▼
SceneGraph.upsert_object()
       │   1. tracker_id + semantic_label match → merge
       │   2. semantic + color + HSV histogram + (optional CLIP) compatible AND
       │      world distance ≤ per-label threshold → merge
       │   3. otherwise spawn new node
       ▼
Periodic place induction + edge rebuild
       │   • greedy radius clustering of nodes → place_id
       │   • k-NN "near" edges + "in_place" edges
       ▼
JsonSceneGraphStore.save()  /  SqliteSceneGraphStore.save()
       (JSON is full rewrite per chunk; SQLite is per-node incremental)
```

The whole phase is **independent** of any pending question. It can run on a
background process, behind an asyncio queue, or on a separate machine that
ships JSON / SQLite files to the QA host.

## Phase 2 — Online retrieval and QA

```text
question ─► QueryParser
              │   labels, colors, wants_count, wants_location, wants_reachability
              ▼
GraphRetriever.retrieve(question, T_world_from_camera_now)
              │   1. score every node against the parsed intent
              │   2. for top-K survivors, project p_world → p_cam
              │      (p_cam = inv(T_world_from_camera) · p_world)
              │   3. classify into 8 ego relations
              │      (front / behind / left / right / front_left / ...)
              │   4. tag reachable = (||p_cam|| ≤ reachable_radius_m)
              │   5. emit ego_edges = [(ego, node, relation, distance_m, reachable)]
              ▼
RetrievedContext  ───► RuleBasedAnswerGenerator  (or VLM via prompts.py)
                       └── "在您的右前方约 0.5 米处（伸手可及）"
```

Online complexity is `O(|matched nodes|)` per query, **not** `O(video length)`.

## Key invariants

1. The world reference frame is anchored at the **first** frame: `T_world(t=0) = I`.
2. Object world anchors only ever change via `add_observation()`, which uses
   pose-confidence-weighted EMA for static items and motion-aware overwrite
   for movable items.
3. Place IDs are derived from object positions, not assumed; if you delete
   the place_nodes entries, the graph is still self-consistent.
4. **Egocentric state is never persisted.** It is computed per query by
   `directme.retrieval.egocentric.render_egocentric` (a pure function, as
   of v0.3) and lives on the per-query `RetrievedItem` only. The only
   persistent spatial state on a node is `spatial_absolute`. v0.2.x had a
   bug where the rendered snapshot was written back to the node and could
   be persisted; v0.3 closes this by removing
   `spatial_egocentric_dynamic` from `EntityNode` entirely.
5. **Trajectory memory lives in `graph.metadata`, not on nodes.** The
   offline engine appends to `metadata["ego_pose_timeline"]` and
   `metadata["place_visit_timeline"]` after every accepted chunk; this is
   the only signal the v0.3 T&M scorer reads. Nodes themselves remain
   pose-anchored static / dynamic entities.
6. **Async ingest is single-writer, non-locking** *(v0.4)*. The
   `AsyncIncrementalMapper` runs one consumer coroutine and offloads
   `process_chunk` to `asyncio.to_thread`. There is **no** read/write lock
   around the scene graph; concurrent in-process retrieval during a chunk
   commit is unsupported. The supported QA pattern is between-commit reads
   from the saved graph (or an out-of-process reader against the SQLite
   WAL file). This is intentional — locking was scoped out of v0.4 to
   keep the change footprint small. See `docs/ingest.md` for details.
7. **Pose drift is measured, not corrected** *(v0.4)*. After every chunk
   commit `graph.metadata["drift_telemetry"]` records cumulative
   world-frame translation, rejected-chunk count, and threshold-driven
   warnings. The graph contains no loop closure; whatever the backbone
   gives the `ChunkPosePropagator`, the propagator integrates as-is.
   See `docs/algorithm_notes.md` § "Why we measure drift but don't
   correct it" for the SCAL3R / DA3 reasoning.
8. **Keyframe selector state is in-memory only** *(v0.4)*. The
   `KeyframeSelector` candidate pool (paths + bbox areas + embeddings)
   is *not* persisted. Only the selected paths in `EntityNode.keyframes`
   are. On graph load the selector is re-seeded from those paths via
   `adopt_existing()` so newly arriving observations compete against —
   but cannot immediately clobber — the existing selection.

## Async ingest pipeline (v0.4)

```
                ┌──────────────────────┐
   video / ─────▶  iter_frames_from_*  │   1-FPS sampling, chunk-agnostic
   frames-dir   └──────────┬───────────┘
                           │ VideoFrame (one per second of capture)
                           ▼
                ┌──────────────────────┐
                │  AsyncIncrementalMapper  │   bounded queue, backpressure
                └──────────┬───────────┘
                           │ chunks of N frames
                           ▼
                ┌──────────────────────┐
                │  OfflineMappingEngine.process_chunk    │
                │   - PerceptionBackend.process_chunk    │   per-chunk fault
                │   - ChunkPosePropagator.propagate      │   isolation:
                │   - SceneGraph fusion + place induction│   exception → log
                │   - drift_telemetry → graph.metadata   │   to failed_chunks
                │   - SqliteSceneGraphStore.save         │   continue
                │   - SqliteSceneGraphStore.record_progress │
                └──────────────────────┘
```

The dotted boxes inside `process_chunk` either all succeed or all roll back
to the previous chunk's state — the engine never writes a half-fused chunk
to the store, so a failed chunk leaves the graph valid.

## Why the offline / online split matters

UCS-Bench videos can run for tens of minutes to multiple hours. Reprocessing
the entire stream at every question is infeasible. The split also matches the
deployment story:

* The scene graph lives on disk and (optionally) in a long-running
  ingestion process.
* The QA path is stateless, cheap, and can be horizontally scaled.

See `docs/algorithm_notes.md` for the design choices behind specific
thresholds and EMA constants.
