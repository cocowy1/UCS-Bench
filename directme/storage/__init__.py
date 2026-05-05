"""DirectMe storage backends.

This package provides JSON and SQLite stores for the scene graph. The two
modules are imported lazily via :pep:`562` ``__getattr__`` so that
``from directme.storage.sqlite_store import SqliteSceneGraphStore`` (the
direct path) does not first have to import ``json_store``, which in turn
imports the mapping package and would otherwise trigger a circular import
through :mod:`directme.mapping.offline_engine`.

Backwards compatible — the public names are still ``JsonSceneGraphStore``
and ``SqliteSceneGraphStore``, just now resolved on first attribute access.
"""

__all__ = ["JsonSceneGraphStore", "SqliteSceneGraphStore"]


def __getattr__(name: str):
    if name == "JsonSceneGraphStore":
        from directme.storage.json_store import JsonSceneGraphStore
        return JsonSceneGraphStore
    if name == "SqliteSceneGraphStore":
        from directme.storage.sqlite_store import SqliteSceneGraphStore
        return SqliteSceneGraphStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
