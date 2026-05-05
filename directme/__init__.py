"""DirectMe 2.0: user-centric continual spatial intelligence for egocentric video streams.

The package root is intentionally side-effect free. Submodules are imported
explicitly by user code, e.g.::

    from directme.config import DirectMeConfig
    from directme.mapping import OfflineMappingEngine, SceneGraph
    from directme.retrieval import GraphRetriever

This avoids circular imports between :mod:`directme.storage` and
:mod:`directme.mapping`.
"""

__version__ = "0.7.0"
__all__ = ["__version__"]
