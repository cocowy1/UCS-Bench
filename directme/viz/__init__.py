"""Lightweight visualization utilities for DirectMe scene graphs.

Heavy dependencies (matplotlib) are imported lazily so importing
``directme.viz`` is cheap when no rendering is requested.
"""

from directme.viz.topdown import render_topdown_map, save_topdown_map

__all__ = ["render_topdown_map", "save_topdown_map"]
