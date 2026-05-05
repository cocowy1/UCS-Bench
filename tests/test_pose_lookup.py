from directme.geometry.poses import SE3
from directme.mapping.scene_graph import SceneGraph
from directme.retrieval.pose_lookup import pose_from_graph_timeline, pose_record_from_se3


def test_pose_from_graph_timeline_returns_nearest_pose():
    graph = SceneGraph()
    graph.metadata["ego_pose_timeline"] = [
        pose_record_from_se3(SE3.from_translation([1, 0, 0]), timestamp=1.0),
        pose_record_from_se3(SE3.from_translation([5, 0, 0]), timestamp=5.0),
    ]
    pose = pose_from_graph_timeline(graph, timestamp=4.2)
    assert pose.translation.tolist() == [5.0, 0.0, 0.0]


def test_pose_from_graph_timeline_falls_back_to_identity():
    graph = SceneGraph()
    assert pose_from_graph_timeline(graph, timestamp=1.0).translation.tolist() == [0.0, 0.0, 0.0]
