"""Pytest fixtures (factories live in graph_factories.py)."""

from __future__ import annotations

import pytest


@pytest.fixture
def identity_pose():
    from directme.geometry.poses import SE3

    return SE3.identity()
