# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FastAPI application for mlir-graph-serve.

This is a thin HTTP transport over `service.py`, where all query logic lives.
The dashboard, a coding agent (via curl), and the push uploader all talk to
these endpoints; `GET /schema` is self-describing so a client can bootstrap the
node/edge model and example Cypher without prior knowledge.
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .db import GraphDB, connect
from .ingest import ingest as ingest_telemetry, ingest_many, graph_exists
from . import service

logger = logging.getLogger(__name__)

app = FastAPI(title="mlir-graph-serve", version="0.1.0")
app.state.files_root = None


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


def init_db(dsn: str) -> None:
    """Connect to the graph DB and store it on app state."""
    db = connect(dsn)
    db.ensure_schema()
    app.state.db = db


def get_db() -> GraphDB:
    """Get the database from app state."""
    return app.state.db


def init_files_root(path: str | None) -> None:
    """Set the directory the /files route serves snapshot .mlir dumps from.

    This is the telemetry/ingest directory; a Snapshot's mlirPath is relative
    to it.
    """
    app.state.files_root = Path(path).resolve() if path else None


def _handle(fn: Callable, *args, **kwargs):
    """Run a service call, translating service errors to HTTP status codes."""
    try:
        return fn(*args, **kwargs)
    except service.NotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except service.BadRequest as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class CypherRequest(BaseModel):
    query: str
    params: dict[str, Any] | None = None
    limit: int | None = None
    offset: int | None = None


class WorkflowMeta(BaseModel):
    workflowName: str = ""
    workflowTitle: str = ""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/ingest")
async def ingest_endpoint(body: dict[str, Any]):
    """Accept a single telemetry JSON document and load it into the database."""
    graph_id = (body.get("graph") or {}).get("graphId", "")
    if graph_id and graph_exists(get_db(), graph_id):
        raise HTTPException(
            status_code=409, detail=f"graph '{graph_id}' already ingested"
        )
    return ingest_telemetry(get_db(), body)


@app.post("/ingest/archive")
async def ingest_archive_endpoint(request: Request):
    """Ingest a whole run (CI run) as one zip.

    The zip carries the telemetry directory layout: `<graph>.json` documents
    plus their `<graph>/<index>_<tag>.mlir` sidecars (exactly a GitHub Actions
    artifact of the telemetry dir). Every member is extracted under the files
    root (so its mlirPath resolves), then every JSON is ingested -- JSON and
    MLIR arrive together in a single request, for all graphs in the run.
    """
    if app.state.files_root is None:
        raise HTTPException(status_code=400, detail="no files root configured")
    try:
        archive = zipfile.ZipFile(io.BytesIO(await request.body()))
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="body is not a valid zip")

    json_docs: list[bytes] = []
    files_written = 0
    for name in archive.namelist():
        if name.endswith("/"):
            continue
        target = _resolve_in_files_root(name)  # rejects zip-slip paths
        target.parent.mkdir(parents=True, exist_ok=True)
        data = archive.read(name)
        target.write_bytes(data)
        files_written += 1
        if name.endswith(".json"):
            json_docs.append(data)

    docs = []
    for data in json_docs:
        try:
            docs.append(json.loads(data))
        except json.JSONDecodeError:
            continue

    # Pre-flight: reject if any run is already ingested, before writing any
    # rows -- ingestion is not idempotent, so a re-pushed archive fails cleanly
    # with 409 instead of a duplicate-key crash mid-way.
    db = get_db()
    dups = sorted(
        d["graph"]["graphId"]
        for d in docs
        if d.get("graph", {}).get("graphId") and graph_exists(db, d["graph"]["graphId"])
    )
    if dups:
        raise HTTPException(
            status_code=409, detail=f"already ingested: {', '.join(dups)}"
        )

    graphs = ingest_many(db, docs)
    return {
        "graphs": graphs,
        "graphsIngested": len(graphs),
        "filesWritten": files_written,
    }


@app.get("/graphs")
async def list_graphs(run_id: str | None = Query(None, alias="runId")):
    """List graphs.

    With no `runId`, returns every graph. With `runId` present (empty string
    selects the local bucket), returns just that run's graphs -- the dashboard
    uses this to lazily expand one run at a time.
    """
    if run_id is not None:
        return _handle(service.list_graphs_for_run, get_db(), run_id)
    return _handle(service.list_graphs, get_db())


@app.get("/runs")
async def list_runs():
    """Run (CI run) rollup for the dashboard's landing page (one row per run)."""
    return _handle(service.list_runs, get_db())


@app.post("/runs/{run_id}/workflow")
async def set_run_workflow(run_id: str, body: WorkflowMeta):
    """Backfill workflow name/title onto every graph of an existing run."""
    return _handle(
        service.set_run_workflow,
        get_db(),
        run_id,
        body.workflowName,
        body.workflowTitle,
    )


@app.get("/search")
async def search_graphs(q: str = Query("")):
    """Substring-search graphs across their human-meaningful fields."""
    return _handle(service.search_graphs, get_db(), q)


@app.get("/graphs/{graph_id}")
async def get_graph(graph_id: str):
    return _handle(service.get_graph, get_db(), graph_id)


@app.get("/graphs/{graph_id}/snapshots")
async def list_graph_snapshots(graph_id: str):
    return _handle(service.list_graph_snapshots, get_db(), graph_id)


@app.get("/snapshots/{snapshot_id}")
async def get_snapshot(snapshot_id: str):
    return _handle(service.get_snapshot, get_db(), snapshot_id)


@app.get("/diff")
async def diff_snapshots(a: str = Query(...), b: str = Query(...)):
    return _handle(service.diff_snapshots, get_db(), a, b)


@app.post("/cypher")
async def run_cypher(body: CypherRequest):
    return _handle(
        service.run_cypher,
        get_db(),
        body.query,
        body.params,
        body.limit,
        body.offset,
    )


@app.get("/schema")
async def get_schema():
    return _handle(service.get_schema, get_db())


def _resolve_in_files_root(file_path: str):
    """Resolve `file_path` under the configured files root, rejecting traversal."""
    root = app.state.files_root
    if root is None:
        raise HTTPException(status_code=400, detail="no files root configured")
    target = (root / file_path).resolve()
    if root not in target.parents:
        raise HTTPException(status_code=400, detail="invalid path")
    return target


@app.get("/files/{file_path:path}")
async def get_file(file_path: str):
    """Serve a snapshot's MLIR dump by its Snapshot.mlirPath (relative path)."""
    target = _resolve_in_files_root(file_path)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(target, media_type="text/plain")


@app.get("/")
async def root():
    """Redirect to the dashboard."""
    return RedirectResponse(url="/ui/")


# Static dashboard (read-only view over the same service-layer endpoints).
# Mounted last so it never shadows the API routes above.
app.mount(
    "/ui",
    StaticFiles(directory=str(Path(__file__).parent / "static"), html=True),
    name="ui",
)
