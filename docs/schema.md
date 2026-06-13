# SceneGraph JSON schema

The graph is a JSON file with:

```json
{
  "schema_version": "directme.scene_graph.v2",
  "reference_frame": "Frame_0_World_Origin",
  "metadata": {
    "ego_pose_timeline": [
      {"chunk_id": 0, "timestamp": 5.0, "T_world_from_camera": [...], "translation": [0,0,0], "scene_tag": "living room"}
    ],
    "place_visit_timeline": [
      {"chunk_id": 0, "timestamp": 5.0, "scene_tag": "living room"},
      {"chunk_id": 1, "timestamp": 15.5, "scene_tag": "kitchen"}
    ],
    "drift_telemetry": {
      "cumulative_translation_m": 47.3,
      "n_chunks_seen": 124,
      "n_chunks_rejected": 2,
      "rejected_chunks": [[31, "translation_jump_8.40m"], [97, "invalid_se3_at_index_4"]],
      "per_chunk_jump_m": [[0, 0.21], [1, 0.34]],
      "warnings": []
    }
  },
  "nodes": [],
  "edges": [],
  "place_nodes": {}
}
```

`metadata.ego_pose_timeline` and `metadata.place_visit_timeline` are appended
to by the offline mapping engine after every accepted chunk and are the only
signal the Trajectory & Movement scorer reads. `place_visit_timeline`
is compressed: a new entry is recorded only when the dominant `scene_tag`
changes between adjacent chunks.

`metadata.drift_telemetry` *(v0.4)* is overwritten (not appended) after every
chunk and reflects the **current** state of `ChunkPosePropagator`. The
`warnings` list contains human-readable strings that fire when
`cumulative_translation_m ≥ drift_warning_translation_m` (default 100 m) or
when the rejected-chunk ratio crosses
`drift_warning_rejection_ratio` (default 10 %). v0.4 does not auto-correct
drift; see `docs/algorithm_notes.md` § "Why we measure drift but don't
correct it".

## Node

```json
{
  "node_id": "entity_002",
  "semantic_label": "cup",
  "aliases": [],
  "attributes": {
    "color": "red",
    "color_hsv_histogram": [0.85, 0.1, 0.05],
    "count_contribution": 1,
    "is_movable": true
  },
  "spatial_absolute": {
    "reference_frame": "Frame_0_World_Origin",
    "p_world": [7.3, 0.0, 0.4],
    "observation_count": 1,
    "last_seen_timestamp": 20.0
  },
  "observations": [],
  "keyframes": [],
  "track_ids": [],
  "created_at": 20.0,
  "updated_at": 20.0
}
```

> **Note (v0.3):** `spatial_egocentric_dynamic` is **not** persisted. It is
> recomputed per query by `directme.retrieval.egocentric.render_egocentric`
> from `spatial_absolute` and the current camera pose, and lives on the
> per-query `RetrievedItem` only — never on the persisted node. v0.2.x
> graph files that still contain the field load fine; the field is
> silently dropped on load. See architecture invariants 4 & 5.
