# Adapter guide

DirectMe core depends on `PerceptionBackend`.

```python
from directme.perception.base import (
    PerceptionBackend,
    VideoFrame,
    ChunkPerception,
    FramePerception,
    ObjectObservation,
)
from directme.geometry.poses import SE3

class MyBackend(PerceptionBackend):
    def process_chunk(self, frames: list[VideoFrame], chunk_id: int) -> ChunkPerception:
        outputs = []
        for frame in frames:
            local_pose = SE3.identity()      # Replace with SCAL3R/SLAM local pose.
            depth = None                     # Replace with DA3 depth.
            intrinsics = None                # 3x3 K.
            objects = [
                ObjectObservation(
                    label="cup",
                    track_id="track_001",
                    p_cam=(0.2, 0.0, 1.1),  # Or provide mask+depth+intrinsics.
                    attributes={"color": "red"},
                    keyframe_path=frame.image_path,
                )
            ]
            outputs.append(FramePerception(frame, local_pose, intrinsics, depth, objects))
        return ChunkPerception(chunk_id=chunk_id, frames=outputs)
```

## Recommended real backend composition

1. Extract frames at 1 FPS or your target FPS.
2. Run chunk-level pose estimation.
3. Run open-vocabulary discovery at a lower frequency.
4. Run segmentation/tracking at a higher frequency.
5. Estimate object 3D centers using mask median depth.
6. Return `FramePerception`.

## Important conventions

- `local_pose` is `T_local_from_camera`, not camera-to-local inverse.
- `ObjectObservation.p_cam` uses camera convention `x=right, y=down, z=forward`.
- The mapping engine converts `p_cam` to `p_world`.
- Graph retrieval converts `p_world` back into current camera coordinates at query time.


## SAM 2 is optional

If you do not have SAM 2 installed, DirectMe can still build a usable graph
from DA3 depth + YOLO-World boxes. In that mode the offline engine estimates
`p_cam` from the bbox center and a robust median depth inside the box. This is
less accurate than mask-based centroids, but it keeps the project easy to run
for community users and CI smoke tests.
