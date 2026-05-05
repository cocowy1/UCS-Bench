from directme.mapping.keyframe_selector import KeyframeSelector, select_keyframes
from directme.mapping.offline_engine import (
    ChunkReport,
    MappingEvent,
    OfflineMappingEngine,
)
from directme.mapping.place_induction import Place, induce_places
from directme.mapping.pose_propagation import (
    ChunkPosePropagator,
    PosePropagationResult,
    is_valid_se3,
    max_translation_jump,
)
from directme.mapping.scene_graph import EntityNode, ObservationRecord, SceneGraph

# Note: ``async_engine`` is intentionally NOT imported eagerly here. It pulls
# in :mod:`asyncio` and would create a circular import via
# ``directme.storage.json_store`` → ``directme.mapping.scene_graph`` →
# ``directme.mapping`` (us) → ``async_engine`` → ``offline_engine`` →
# ``directme.storage.json_store``. Import the async API explicitly:
#
#     from directme.mapping.async_engine import (
#         AsyncIncrementalMapper, FailedChunkRecord, ingest_frames_async,
#     )

__all__ = [
    "ChunkPosePropagator",
    "ChunkReport",
    "EntityNode",
    "KeyframeSelector",
    "MappingEvent",
    "ObservationRecord",
    "OfflineMappingEngine",
    "Place",
    "PosePropagationResult",
    "SceneGraph",
    "induce_places",
    "is_valid_se3",
    "max_translation_jump",
    "select_keyframes",
]
