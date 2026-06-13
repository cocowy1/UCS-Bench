# Algorithm notes

This file collects design choices that aren't obvious from reading the code.

## 1. Cluster in world coordinates, not camera coordinates

Camera-frame positions move with the user every frame. Two observations of
the *same* physical cup will have totally different `p_cam` values one
second apart. Therefore:

```text
p_world = T_world_from_camera(t) · p_cam   (during ingest)
cluster on p_world                         (in SceneGraph._find_match)
```

`p_cam_current = inv(T_world_from_camera_now) · p_world_node` is computed
**only at query time** and is never persisted.

## 2. Chunk seam handling

A local pose optimizer (DA3, Scal3R, etc.) may return chunk poses in an
arbitrary local frame whose first pose is not exactly identity. Therefore:

```text
T_world_from_local = T_world_end ∘ inv(T_local_start)
T_world(t) = T_world_from_local ∘ T_local(t)
```

When `T_local_start = I`, this reduces to the textbook
`T_world(t) = T_world_end ∘ T_local(t)`. The general form is one extra
matrix multiply per chunk and avoids seam tearing.

## 3. Identity matching cascade

`SceneGraph._find_match` evaluates these in order:

1. **Same tracker id + compatible label** → instant merge.
2. **Semantic compatibility gate** (label substring / token-set match).
3. **Discrete color name gate** (`red` vs `blue` → reject).
4. **HSV histogram cosine similarity gate** (skipped if either side is missing).
5. **Optional CLIP / DINO embedding cosine gate** (skipped if either side is missing).
6. **Per-label distance threshold** (e.g. `cup`: 0.35 m, `sofa`: 1.50 m).
7. **Composite score** `−distance + 0.20·hist_sim + 0.20·emb_sim` picks the
   single best survivor.

Steps 4 and 5 are optional and additive — supplying both gives the strongest
identity decisions but the system still works with neither.

## 4. Movable vs static updates

`is_movable=true` switches two behaviors:

* The EMA on `spatial_absolute.p_world` uses `dynamic_update_alpha=0.70`
  instead of `static_update_alpha=0.20` — the latest observation dominates.
* If displacement exceeds `motion_overwrite_threshold_m` (default 0.5 m),
  the EMA is bypassed entirely and the new observation overwrites the anchor.

This handles the "phone left on a table, picked up, placed in a bag" case
without producing a phantom node in between.

## 5. Pose-confidence-weighted EMA

The effective EMA alpha is `alpha_base · pose_confidence`. A frame whose
DA3 mean confidence is 0.3 contributes only 30 % of a high-confidence frame.
This prevents brief illumination failures or motion blur from snapping
established anchors to noisy positions.

## 6. Counting

Physical counts equal **node** counts after fusion, not frame counts. Every
node has `count_contribution = 1`. Repeated observations of the same node
(`merge` events) do not increase the count. Two visually identical objects
in different rooms become two nodes because their world distance exceeds
the per-label threshold.

## 7. Reachability is depth-only

`reachable = (||p_cam|| ≤ reachable_radius_m)` where `||·||` is the
Euclidean norm. We deliberately do not factor in obstacles, line-of-sight,
or floor occupancy in v0.2 — that requires an actual occupancy map which
DirectMe does not yet build. Treat reachable as a **conservative spatial
proximity flag**, not a navigation guarantee.

## 8. Online complexity

`GraphRetriever.retrieve()` runs in `O(|nodes|)` because `_score_node` is
linear. For graphs above ~10 K nodes you should add a spatial index
(KD-tree on `p_world`) — this is currently not implemented.

## 9. Long-horizon drift

DirectMe v0.2 has **no loop closure** and no global pose-graph optimization.
Long sessions accumulate drift proportional to the underlying pose
estimator's per-frame error. Mitigations on the roadmap:

* Re-observation drift detection (track a stable landmark and log when its
  re-observed `p_world` differs from the stored anchor by > τ).
* Optional GTSAM / g2o pose-graph optimizer over chunk anchors.
* IMU / dead-reckoning fallback when a chunk is rejected.

None of these are in the current release.

## Why we measure drift but don't correct it (v0.4)

A common question: "Doesn't SCAL3R / DA3 / MASt3R already solve the pose
drift problem, since they jointly optimize depth and camera pose?"

**Partly. They solve it within a chunk; they do not solve it across
chunks.**

Concretely, when DirectMe sends a chunk of N frames to the perception
backbone:

1. The backbone runs whatever internal optimization it does — sparse
   feature matching → local bundle adjustment → multi-view depth
   prediction. The output is N camera poses **in a chunk-local
   coordinate system**, with the first frame typically defined as the
   chunk-local origin (`extrinsics[0] ≈ I`).
2. Within that chunk, the per-frame poses are mutually consistent up to
   the backbone's accuracy. SCAL3R is genuinely strong here — its
   chunk-internal pose error on hand-held egocentric video is
   sub-decimeter on indoor scenes.
3. To produce a single world-frame trajectory across the entire video,
   `ChunkPosePropagator` recursively composes:
   `T_world(chunk_k) = T_world(chunk_{k-1}_end) · T_chunkLocal(chunk_k_start)^-1 · T_chunkLocal(chunk_k_*)`.

It is step 3 that introduces drift. Each chunk-to-chunk junction adds a
small alignment error, and these errors **accumulate monotonically**
along the trajectory — exactly like wheel-odometry drift on a robot.
Backbone strength does not fix this because the backbone never sees the
junction; the chunks are processed independently.

The conventional fix is **loop closure**: when the camera revisits a
place, detect the revisit (e.g. via keyframe-embedding similarity) and
solve a global pose-graph optimization that distributes the accumulated
error around the loop. This is well-understood (ORB-SLAM, Kimera,
COLMAP all do it) but adds substantial machinery: a keyframe database,
a place-recognition module, a pose-graph backend (g2o, GTSAM, Ceres),
and re-derivation of every object's world anchor when the trajectory
shifts.

**v0.4's choice is to not implement this.** Instead, the
`ChunkPosePropagator` records:

* `cumulative_translation_m` — total integrated path length
* `n_chunks_rejected` and `rejected_chunks` — chunks the propagator
  refused to integrate (NaN, non-orthogonal R, or > `max_per_frame_jump_m`)
* `per_chunk_jump_m` — the largest per-frame translation jump in each chunk

…and surfaces threshold-driven warnings into
`graph.metadata["drift_telemetry"]["warnings"]`. This is *honest*: the
user can see "your trajectory is 200 m long with no loop closure;
positions far from the start may be off by a few meters" rather than
seeing a confidently wrong fused graph.

When does this cost real accuracy? Two regimes:

* **Short videos** (< 100 m of cumulative translation, < ~20 minutes of
  capture without large loops): drift is bounded by backbone precision;
  v0.4 is essentially as good as a closure-equipped system.
* **Long videos with revisits** (a 30-min walk through a house that
  returns to the kitchen multiple times): without loop closure, each
  revisit lands at a slightly different world coordinate. Object
  re-identification still works because it uses appearance + label, but
  the *position* of the kitchen drifts. A future v0.5 will close this
  with a keyframe-embedding-based loop-closure module.

In practice for the UCS-Bench scale (several minutes of capture per
clip), v0.4's measure-don't-correct approach is sufficient. For
multi-hour ingests, plan to break the video into ≤ 30-minute segments
or wait for v0.5.
