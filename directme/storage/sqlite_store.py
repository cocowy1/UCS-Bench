"""SQLite-backed incremental scene graph persistence.

Designed as a drop-in replacement for :class:`JsonSceneGraphStore` when the
scene graph grows beyond what is comfortable to fully serialize every chunk.

Schema is intentionally simple: we store the graph as JSON blobs partitioned
by node so we can update only the rows that changed in the most recent chunk.
WAL mode is enabled for concurrent reads from the online QA path.

Trade-off:
  * JSON store : trivial, debuggable, but O(graph size) on every save.
  * SQLite store: O(changed nodes), supports concurrent readers.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from directme.mapping.scene_graph import EntityNode, SceneGraph


class SqliteSceneGraphStore:
    """Incremental per-node persistence with WAL-enabled SQLite."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()
        self._dirty_node_ids: set[str] = set()
        self._known_node_ids: set[str] = set()
        self._all_dirty: bool = True
        self._last_save_time: float = 0.0

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                updated_at REAL,
                p_world_x REAL,
                p_world_y REAL,
                p_world_z REAL
            );
            CREATE TABLE IF NOT EXISTS edges (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                relation TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS places (
                place_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            );
            -- v0.4: ingest progress for resumable async runs.
            CREATE TABLE IF NOT EXISTS progress (
                key TEXT PRIMARY KEY,     -- 'last_committed_chunk_id', 'last_committed_timestamp'
                value TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_nodes_position ON nodes (p_world_x, p_world_y, p_world_z);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges (source);
            """
        )
        self._conn.commit()

    # --- progress (v0.4) ----------------------------------------------------

    def record_progress(self, chunk_id: int, timestamp: float | None = None) -> None:
        """Record that ``chunk_id`` has been fully committed.

        Used by :class:`AsyncIncrementalMapper` so a crashed / killed
        process can resume from the next chunk on restart instead of
        re-running perception over already-processed video.
        """
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO progress (key, value) VALUES (?, ?)",
            ("last_committed_chunk_id", str(int(chunk_id))),
        )
        if timestamp is not None:
            cur.execute(
                "INSERT OR REPLACE INTO progress (key, value) VALUES (?, ?)",
                ("last_committed_timestamp", repr(float(timestamp))),
            )
        self._conn.commit()

    def get_last_committed_chunk_id(self) -> int | None:
        """Return the highest chunk_id known to have been committed, or None."""
        cur = self._conn.cursor()
        row = cur.execute(
            "SELECT value FROM progress WHERE key = ?",
            ("last_committed_chunk_id",),
        ).fetchone()
        if not row:
            return None
        try:
            return int(row[0])
        except (TypeError, ValueError):
            return None

    # --- write path ---------------------------------------------------------

    def mark_dirty(self, node_ids: Iterable[str]) -> None:
        for nid in node_ids:
            self._dirty_node_ids.add(nid)

    def save(self, graph: SceneGraph) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("reference_frame", graph.reference_frame),
        )
        cur.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("metadata", json.dumps(graph.metadata, ensure_ascii=False)),
        )
        cur.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (
                "settings",
                json.dumps(graph.to_dict().get("settings", {}), ensure_ascii=False),
            ),
        )

        # Decide which nodes to write.
        #
        # Defensive policy: even if the caller forgot to call mark_dirty(), we
        # *always* detect newly-created and deleted nodes by diffing against
        # the set of node_ids we have already persisted. Only the "in-place
        # mutation of an existing node" case still requires an explicit
        # mark_dirty() call (which the OfflineMappingEngine does for us).
        current_ids = set(graph.nodes.keys())
        new_ids = current_ids - self._known_node_ids
        deleted_ids = self._known_node_ids - current_ids

        if self._all_dirty:
            ids_to_upsert: set[str] = current_ids
        else:
            ids_to_upsert = set(self._dirty_node_ids) | new_ids

        for nid in ids_to_upsert:
            node = graph.nodes.get(nid)
            if node is None:
                cur.execute("DELETE FROM nodes WHERE node_id = ?", (nid,))
                continue
            payload = json.dumps(node.to_dict(), ensure_ascii=False)
            pw = np.asarray(node.p_world, dtype=float).reshape(3)
            cur.execute(
                """
                INSERT OR REPLACE INTO nodes (node_id, payload, updated_at, p_world_x, p_world_y, p_world_z)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (nid, payload, node.updated_at, float(pw[0]), float(pw[1]), float(pw[2])),
            )

        for nid in deleted_ids:
            cur.execute("DELETE FROM nodes WHERE node_id = ?", (nid,))

        self._known_node_ids = set(current_ids)

        # Edges and place_nodes are small; we rewrite them in full each save.
        cur.execute("DELETE FROM edges")
        for e in graph.edges:
            cur.execute(
                "INSERT INTO edges (source, target, relation, payload) VALUES (?, ?, ?, ?)",
                (e["source"], e["target"], e["relation"], json.dumps(e, ensure_ascii=False)),
            )
        cur.execute("DELETE FROM places")
        for pid, p in graph.place_nodes.items():
            cur.execute(
                "INSERT INTO places (place_id, payload) VALUES (?, ?)",
                (pid, json.dumps(p, ensure_ascii=False)),
            )

        self._conn.commit()
        self._dirty_node_ids.clear()
        self._all_dirty = False
        self._last_save_time = time.time()

    # --- read path ----------------------------------------------------------

    def load(self) -> SceneGraph:
        cur = self._conn.cursor()
        cur.execute("SELECT key, value FROM meta")
        meta_rows = dict(cur.fetchall())
        ref = meta_rows.get("reference_frame", "Frame_0_World_Origin")
        metadata = json.loads(meta_rows["metadata"]) if "metadata" in meta_rows else {}

        settings = json.loads(meta_rows["settings"]) if "settings" in meta_rows else {}
        allowed = {
            "merge_threshold_m",
            "max_observations_per_node",
            "keyframes_per_node",
            "dynamic_update_alpha",
            "static_update_alpha",
            "motion_overwrite_threshold_m",
            "color_histogram_min_similarity",
            "semantic_embedding_min_similarity",
            "track_match_max_gap_frames",
            "track_match_max_jump_m",
            "label_merge_thresholds_m",
        }
        graph = SceneGraph(
            reference_frame=ref,
            **{k: v for k, v in settings.items() if k in allowed},
        )
        graph.metadata = metadata

        cur.execute("SELECT payload FROM nodes")
        for (payload,) in cur.fetchall():
            data = json.loads(payload)
            node = EntityNode.from_dict(data)
            graph.nodes[node.node_id] = node

        cur.execute("SELECT payload FROM edges")
        graph.edges = [json.loads(row[0]) for row in cur.fetchall()]

        cur.execute("SELECT place_id, payload FROM places")
        graph.place_nodes = {pid: json.loads(payload) for pid, payload in cur.fetchall()}

        max_id = 0
        for node_id in graph.nodes:
            try:
                max_id = max(max_id, int(node_id.split("_")[-1]))
            except ValueError:
                pass
        graph._next_id = max_id + 1
        return graph

    def exists(self) -> bool:
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM nodes")
        return cur.fetchone()[0] > 0

    def close(self) -> None:
        self._conn.close()
