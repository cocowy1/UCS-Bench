import numpy as np

from directme.config import DirectMeConfig
from directme.geometry.poses import SE3
from directme.geometry.unprojection import robust_bbox_depth, unproject_bbox_center
from directme.mapping.offline_engine import OfflineMappingEngine
from directme.perception.base import ChunkPerception, FramePerception, ObjectObservation, VideoFrame
from directme.perception.base import PerceptionBackend


def test_unproject_bbox_center_uses_robust_depth():
    depth = np.ones((8, 8), dtype=np.float32) * 2.0
    depth[0, 0] = 99.0
    intrinsics = np.array([[4.0, 0.0, 4.0], [0.0, 4.0, 4.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    bbox = (2.0, 2.0, 6.0, 6.0)

    z = robust_bbox_depth(bbox, depth)
    assert z == 2.0

    p_cam = unproject_bbox_center(bbox, depth, intrinsics)
    np.testing.assert_allclose(p_cam, np.array([0.0, 0.0, 2.0]), atol=1e-6)


class _BBoxOnlyBackend(PerceptionBackend):
    def process_chunk(self, frames: list[VideoFrame], chunk_id: int) -> ChunkPerception:
        fp = FramePerception(
            frame=frames[0],
            local_pose=SE3.identity(),
            intrinsics=np.array([[4.0, 0.0, 4.0], [0.0, 4.0, 4.0], [0.0, 0.0, 1.0]], dtype=np.float32),
            depth=np.ones((8, 8), dtype=np.float32) * 2.0,
            objects=[
                ObjectObservation(
                    label="cup",
                    bbox_xyxy=(2.0, 2.0, 6.0, 6.0),
                    track_id="track_001",
                )
            ],
        )
        return ChunkPerception(chunk_id=chunk_id, frames=[fp])


def test_offline_engine_accepts_bbox_only_observation(tmp_path):
    cfg = DirectMeConfig()
    cfg.run_dir = str(tmp_path)
    engine = OfflineMappingEngine(backend=_BBoxOnlyBackend(), config=cfg)
    frame = VideoFrame(index=0, timestamp=0.0)

    events = engine.process_frames([frame], chunk_size=1)

    assert len(events) == 1
    assert engine.graph is not None
    assert len(engine.graph.nodes) == 1
    node = next(iter(engine.graph.nodes.values()))
    np.testing.assert_allclose(node.p_world, np.array([0.0, 0.0, 2.0]), atol=1e-6)
