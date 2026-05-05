"""Tests for the lightweight IoU+appearance tracker.

Heavy backbones (YOLO-World, SAM 2) are not exercised here because they
require GPU and large model checkpoints — those are integration tests.
"""

import numpy as np

from directme.perception.adapters.open_vocab_tracking import (
    Detection,
    SimpleIoUAppearanceTracker,
)


def test_tracker_keeps_id_under_iou_overlap():
    tracker = SimpleIoUAppearanceTracker()
    e = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    d1 = Detection(label="cup", bbox_xyxy=(10, 10, 50, 50), score=0.9, embedding=e)
    out1 = tracker.step(0, [d1])
    tid1 = out1[0].track_id

    # Slight shift on next frame: high IoU and same embedding → same id.
    d2 = Detection(label="cup", bbox_xyxy=(12, 12, 52, 52), score=0.9, embedding=e)
    out2 = tracker.step(1, [d2])
    assert out2[0].track_id == tid1


def test_tracker_assigns_new_id_for_distinct_object():
    tracker = SimpleIoUAppearanceTracker()
    a = Detection(label="cup", bbox_xyxy=(10, 10, 50, 50), score=0.9)
    out_a = tracker.step(0, [a])
    b = Detection(label="cup", bbox_xyxy=(500, 500, 540, 540), score=0.9)
    out_b = tracker.step(1, [b])
    assert out_a[0].track_id != out_b[0].track_id


def test_tracker_does_not_match_across_labels():
    tracker = SimpleIoUAppearanceTracker()
    cup = Detection(label="cup", bbox_xyxy=(10, 10, 50, 50), score=0.9)
    out_cup = tracker.step(0, [cup])
    phone = Detection(label="phone", bbox_xyxy=(11, 11, 51, 51), score=0.9)
    out_phone = tracker.step(1, [phone])
    assert out_cup[0].track_id != out_phone[0].track_id
