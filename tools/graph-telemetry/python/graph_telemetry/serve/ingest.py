# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Ingest telemetry JSON into the graph database.

A single parameterized UNWIND path: rows are collected per table and flushed in
dependency order (nodes, then relationships). This handles Kuzu's typed list
columns (tensor shape) and the EAV attribute bag natively, with none of the CSV
escaping hazards a bulk-loader would bring.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from .db import GraphDB

logger = logging.getLogger(__name__)

# Chunk size for UNWIND batches -- large enough to amortize round-trips, small
# enough to keep parameter payloads bounded.
_CHUNK = 5000


def _attr_id(key: str, val: str) -> str:
    """Content-addressed id for an Attr node, so (key, val) pairs are shared."""
    h = hashlib.sha1()
    h.update(key.encode())
    h.update(b"\x00")
    h.update(val.encode())
    return h.hexdigest()


class _Collector:
    """Accumulates rows for every table across one telemetry document."""

    def __init__(self) -> None:
        self.graphs: list[dict] = []
        self.snapshots: list[dict] = []
        self.ops: list[dict] = []
        self.block_args: list[dict] = []
        self.tensors: list[dict] = []
        self.attrs: dict[str, dict] = {}  # id -> {id, key, val}

        self.in_graph: list[dict] = []
        self.in_snap_op: list[dict] = []
        self.in_snap_ba: list[dict] = []
        self.in_snap_tensor: list[dict] = []
        self.contains: list[dict] = []
        self.has_arg: list[dict] = []
        self.produces_op: list[dict] = []
        self.produces_ba: list[dict] = []
        self.consumed_by: list[dict] = []
        self.feeds_op: list[dict] = []
        self.feeds_ba: list[dict] = []
        self.calls: list[dict] = []
        self.refs: list[dict] = []
        self.has_attr_op: list[dict] = []
        self.has_attr_tensor: list[dict] = []

    def add_telemetry(self, doc: dict) -> dict:
        graph = doc["graph"]
        graph_id = graph["graphId"]
        self.graphs.append(
            {
                "graphId": graph_id,
                "modelName": graph.get("modelName", ""),
                "testName": graph.get("testName", ""),
                "gitSha": graph.get("gitSha", ""),
                "runId": graph.get("runId", ""),
                "jobId": graph.get("jobId", ""),
                "jobName": graph.get("jobName", ""),
                "workflowName": graph.get("workflowName", ""),
                "workflowTitle": graph.get("workflowTitle", ""),
                "branch": graph.get("branch", ""),
                "createdAt": graph.get("createdAt", 0),
            }
        )

        source_path = doc.get("sourcePath", "")
        graph_index = doc.get("graphIndex", 0)
        snapshots = doc.get("snapshots", [])
        for snap in snapshots:
            self._collect_snapshot(snap, graph_id, source_path, graph_index)

        return {"graphId": graph_id, "snapshotsIngested": len(snapshots)}

    def _add_attrs(self, owner_id: str, attrs: dict, sink: list[dict]) -> None:
        for key, val in (attrs or {}).items():
            val = "" if val is None else str(val)
            aid = _attr_id(key, val)
            if aid not in self.attrs:
                self.attrs[aid] = {"id": aid, "key": key, "val": val}
            sink.append({"f": owner_id, "t": aid})

    def _collect_snapshot(
        self, snap: dict, graph_id: str, source_path: str, graph_index: int
    ) -> None:
        sid = snap["snapshotId"]
        self.snapshots.append(
            {
                "snapshotId": sid,
                "graphId": graph_id,
                "tag": snap.get("tag", ""),
                "passName": snap.get("passName", ""),
                "snapshotIndex": snap.get("snapshotIndex", 0),
                "timestampUs": snap.get("timestampUs", 0),
                "sourcePath": source_path,
                "graphIndex": graph_index,
                "mlirPath": snap.get("mlirPath", ""),
            }
        )
        self.in_graph.append({"f": sid, "t": graph_id})

        for op in snap.get("ops", []):
            oid = op["id"]
            self.ops.append(
                {
                    "id": oid,
                    "snapshotId": sid,
                    "graphId": graph_id,
                    "parentId": op.get("parentId", ""),
                    "regionIdx": op.get("regionIdx", 0),
                    "blockIdx": op.get("blockIdx", 0),
                    "orderIdx": op.get("order", 0),
                    "opName": op.get("opName", ""),
                    "dialect": op.get("dialect", ""),
                    "location": op.get("location", ""),
                    "mlirLine": op.get("mlirLine", 0),
                    "symName": op.get("symName", ""),
                    "isFunc": bool(op.get("isFunc", False)),
                    "numArgs": op.get("numArgs", 0),
                    "numResults": op.get("numResults", 0),
                    "isTerminator": bool(op.get("isTerminator", False)),
                }
            )
            self.in_snap_op.append({"f": oid, "t": sid})
            if op.get("parentId"):
                self.contains.append(
                    {
                        "f": op["parentId"],
                        "t": oid,
                        "regionIdx": op.get("regionIdx", 0),
                        "blockIdx": op.get("blockIdx", 0),
                        "orderIdx": op.get("order", 0),
                    }
                )
            self._add_attrs(oid, op.get("attrs"), self.has_attr_op)

        for ba in snap.get("blockArgs", []):
            bid = ba["id"]
            self.block_args.append(
                {
                    "id": bid,
                    "snapshotId": sid,
                    "graphId": graph_id,
                    "parentId": ba.get("parentId", ""),
                    "regionIdx": ba.get("regionIdx", 0),
                    "blockIdx": ba.get("blockIdx", 0),
                    "argIdx": ba.get("argIdx", 0),
                }
            )
            self.in_snap_ba.append({"f": bid, "t": sid})
            if ba.get("parentId"):
                self.has_arg.append(
                    {
                        "f": ba["parentId"],
                        "t": bid,
                        "regionIdx": ba.get("regionIdx", 0),
                        "blockIdx": ba.get("blockIdx", 0),
                        "argIdx": ba.get("argIdx", 0),
                    }
                )

        # Index tensors so derived FEEDS edges can resolve producers.
        tensor_by_id: dict[str, dict] = {}
        for t in snap.get("tensors", []):
            tid = t["id"]
            tensor_by_id[tid] = t
            self.tensors.append(
                {
                    "id": tid,
                    "snapshotId": sid,
                    "graphId": graph_id,
                    "producerId": t.get("producerId", ""),
                    "producerType": t.get("producerType", ""),
                    "resultIdx": t.get("resultIdx", 0),
                    "shape": t.get("shape", []),
                    "rank": t.get("rank", 0),
                    "dtype": t.get("dtype", ""),
                }
            )
            self.in_snap_tensor.append({"f": tid, "t": sid})
            pid, ptype = t.get("producerId", ""), t.get("producerType", "")
            if pid and ptype == "op":
                self.produces_op.append(
                    {"f": pid, "t": tid, "resultIdx": t.get("resultIdx", 0)}
                )
            elif pid and ptype == "block_arg":
                self.produces_ba.append(
                    {"f": pid, "t": tid, "resultIdx": t.get("resultIdx", 0)}
                )
            self._add_attrs(tid, t.get("attrs"), self.has_attr_tensor)

        for e in snap.get("edges", []):
            tid, cid = e["tensorId"], e["consumerId"]
            op_idx = e.get("operandIdx", 0)
            self.consumed_by.append({"f": tid, "t": cid, "operandIdx": op_idx})
            # Derived producer -> consumer shortcut.
            t = tensor_by_id.get(tid)
            if not t or not t.get("producerId"):
                continue
            feed = {
                "f": t["producerId"],
                "t": cid,
                "operandIdx": op_idx,
                "resultIdx": t.get("resultIdx", 0),
            }
            if t.get("producerType") == "op":
                self.feeds_op.append(feed)
            elif t.get("producerType") == "block_arg":
                self.feeds_ba.append(feed)

        for c in snap.get("calls", []):
            self.calls.append({"f": c["callerId"], "t": c["calleeId"]})
        for r in snap.get("refs", []):
            self.refs.append(
                {
                    "f": r["userId"],
                    "t": r["targetId"],
                    "attrName": r.get("attrName", ""),
                }
            )

    # -- flush ---------------------------------------------------------------

    def flush(self, db: GraphDB) -> None:
        def unwind(cypher: str, rows: list[dict]) -> None:
            for i in range(0, len(rows), _CHUNK):
                db.execute_write(cypher, {"rows": rows[i : i + _CHUNK]})

        # Nodes. Graph/Attr are MERGEd so re-ingest and shared attrs are idempotent.
        unwind(
            "UNWIND $rows AS r MERGE (n:Graph {graphId: r.graphId}) "
            "ON CREATE SET n.modelName=r.modelName, n.testName=r.testName, "
            "n.gitSha=r.gitSha, n.runId=r.runId, "
            "n.jobId=r.jobId, n.jobName=r.jobName, "
            "n.workflowName=r.workflowName, n.workflowTitle=r.workflowTitle, "
            "n.branch=r.branch, n.createdAt=r.createdAt",
            self.graphs,
        )
        unwind(
            "UNWIND $rows AS r CREATE (n:Snapshot {snapshotId: r.snapshotId, "
            "graphId: r.graphId, tag: r.tag, passName: r.passName, "
            "snapshotIndex: r.snapshotIndex, timestampUs: r.timestampUs, "
            "sourcePath: r.sourcePath, graphIndex: r.graphIndex, "
            "mlirPath: r.mlirPath})",
            self.snapshots,
        )
        unwind(
            "UNWIND $rows AS r CREATE (n:Op {id: r.id, snapshotId: r.snapshotId, "
            "graphId: r.graphId, "
            "parentId: r.parentId, regionIdx: r.regionIdx, blockIdx: r.blockIdx, "
            "orderIdx: r.orderIdx, opName: r.opName, dialect: r.dialect, "
            "location: r.location, mlirLine: r.mlirLine, "
            "symName: r.symName, isFunc: r.isFunc, "
            "numArgs: r.numArgs, numResults: r.numResults, "
            "isTerminator: r.isTerminator})",
            self.ops,
        )
        unwind(
            "UNWIND $rows AS r CREATE (n:BlockArg {id: r.id, "
            "snapshotId: r.snapshotId, graphId: r.graphId, parentId: r.parentId, "
            "regionIdx: r.regionIdx, blockIdx: r.blockIdx, argIdx: r.argIdx})",
            self.block_args,
        )
        unwind(
            # CAST shape to INT64[]: when a whole chunk's shapes are empty
            # lists (e.g. a graph of only scalar/non-tensor values), Kuzu infers
            # the param as STRING[] and rejects it against the INT64[] column.
            "UNWIND $rows AS r CREATE (n:Tensor {id: r.id, "
            "snapshotId: r.snapshotId, graphId: r.graphId, producerId: r.producerId, "
            "producerType: r.producerType, resultIdx: r.resultIdx, "
            "shape: CAST(r.shape AS INT64[]), rank: r.rank, dtype: r.dtype})",
            self.tensors,
        )
        unwind(
            "UNWIND $rows AS r MERGE (n:Attr {id: r.id}) "
            "ON CREATE SET n.key = r.key, n.val = r.val",
            list(self.attrs.values()),
        )

        # Relationships. Bulk COPY resolves endpoints via the primary-key index;
        # UNWIND ... MATCH ... CREATE plans as a node-table scan + cross product
        # (quadratic as the DB grows). Multi-pair rel tables need from/to.
        db.bulk_copy_rel("IN_GRAPH", self.in_graph)
        db.bulk_copy_rel("IN_SNAPSHOT", self.in_snap_op, "Op", "Snapshot")
        db.bulk_copy_rel("IN_SNAPSHOT", self.in_snap_ba, "BlockArg", "Snapshot")
        db.bulk_copy_rel("IN_SNAPSHOT", self.in_snap_tensor, "Tensor", "Snapshot")
        db.bulk_copy_rel("CONTAINS", self.contains)
        db.bulk_copy_rel("HAS_ARG", self.has_arg)
        db.bulk_copy_rel("PRODUCES", self.produces_op, "Op", "Tensor")
        db.bulk_copy_rel("PRODUCES", self.produces_ba, "BlockArg", "Tensor")
        db.bulk_copy_rel("CONSUMED_BY", self.consumed_by)
        db.bulk_copy_rel("FEEDS", self.feeds_op, "Op", "Op")
        db.bulk_copy_rel("FEEDS", self.feeds_ba, "BlockArg", "Op")
        db.bulk_copy_rel("CALLS", self.calls)
        db.bulk_copy_rel("REFERENCES", self.refs)
        db.bulk_copy_rel("HAS_ATTR", self.has_attr_op, "Op", "Attr")
        db.bulk_copy_rel("HAS_ATTR", self.has_attr_tensor, "Tensor", "Attr")

        logger.info(
            "Ingest complete: %d snapshots, %d ops, %d tensors, %d attrs",
            len(self.snapshots),
            len(self.ops),
            len(self.tensors),
            len(self.attrs),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ingest(db: GraphDB, telemetry_json: dict) -> dict:
    """Ingest a telemetry JSON document into the graph DB."""
    db.ensure_schema()
    collector = _Collector()
    summary = collector.add_telemetry(telemetry_json)
    collector.flush(db)
    return summary


def ingest_many(db: GraphDB, docs: list[dict]) -> list[dict]:
    """Ingest several telemetry documents in one batch: a single collector
    accumulates all rows and flushes once, so N graphs cost a handful of
    chunked UNWINDs instead of ~15 small transactions per graph. Shared Attr
    nodes dedupe across the whole batch."""
    db.ensure_schema()
    collector = _Collector()
    summaries = [collector.add_telemetry(doc) for doc in docs]
    collector.flush(db)
    return summaries


def graph_exists(db: GraphDB, graph_id: str) -> bool:
    """True if a Graph with this id is already ingested. Ingestion is not
    idempotent (primary keys are unique), so callers use this to reject a
    re-push cleanly instead of hitting a duplicate-key error."""
    db.ensure_schema()
    rows = db.execute(
        "MATCH (r:Graph {graphId: $graphId}) RETURN r.graphId LIMIT 1",
        {"graphId": graph_id},
    )
    return bool(rows)


def ingest_file(db: GraphDB, path: str) -> dict:
    with open(path) as f:
        return ingest(db, json.load(f))


def ingest_dir(db: GraphDB, dir_path: str) -> list[dict]:
    """Ingest all .json files in a directory in one batch."""
    json_files = sorted(Path(dir_path).glob("*.json"))
    if not json_files:
        logger.warning("No JSON files found in %s", dir_path)
        return []

    db.ensure_schema()
    collector = _Collector()
    results = []
    for jf in json_files:
        logger.info("Collecting %s", jf)
        with open(jf) as f:
            results.append(collector.add_telemetry(json.load(f)))
    collector.flush(db)
    return results
