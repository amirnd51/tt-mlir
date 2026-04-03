# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Named Cypher query templates and schema for mlir-graph-serve (Kuzu).

The schema is a uniform reflection of MLIR: every operation -- modules and
functions included -- is an `Op` node, nested via `CONTAINS`. Dataflow is
`Op -[PRODUCES]-> Tensor -[CONSUMED_BY]-> Op`, with a derived `FEEDS` shortcut
edge `Op -> Op` for terse pattern matching. Open-ended attributes are modelled
as shared `Attr {key, val}` nodes (EAV), so any attribute is directly
queryable without knowing the schema up front; only universal, typed fields
(tensor shape/rank/dtype, op flags) are top-level columns.
"""

# ---------------------------------------------------------------------------
# Schema creation (Kuzu)
# ---------------------------------------------------------------------------

KUZU_CREATE_NODE_TABLES = [
    """CREATE NODE TABLE IF NOT EXISTS Graph(
        graphId STRING,
        modelName STRING,
        testName STRING,
        gitSha STRING,
        runId STRING,
        jobId STRING,
        jobName STRING,
        workflowName STRING,
        workflowTitle STRING,
        branch STRING,
        createdAt INT64,
        PRIMARY KEY(graphId)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Snapshot(
        snapshotId STRING,
        graphId STRING,
        tag STRING,
        passName STRING,
        snapshotIndex INT64,
        timestampUs INT64,
        sourcePath STRING,
        graphIndex INT64,
        mlirPath STRING,
        PRIMARY KEY(snapshotId)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Op(
        id STRING,
        snapshotId STRING,
        graphId STRING,
        parentId STRING,
        regionIdx INT64,
        blockIdx INT64,
        orderIdx INT64,
        opName STRING,
        dialect STRING,
        location STRING,
        mlirLine INT64,
        symName STRING,
        isFunc BOOL,
        numArgs INT64,
        numResults INT64,
        isTerminator BOOL,
        PRIMARY KEY(id)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS BlockArg(
        id STRING,
        snapshotId STRING,
        graphId STRING,
        parentId STRING,
        regionIdx INT64,
        blockIdx INT64,
        argIdx INT64,
        PRIMARY KEY(id)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Tensor(
        id STRING,
        snapshotId STRING,
        graphId STRING,
        producerId STRING,
        producerType STRING,
        resultIdx INT64,
        shape INT64[],
        rank INT64,
        dtype STRING,
        PRIMARY KEY(id)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Attr(
        id STRING,
        key STRING,
        val STRING,
        PRIMARY KEY(id)
    )""",
]

KUZU_CREATE_REL_TABLES = [
    "CREATE REL TABLE IF NOT EXISTS IN_GRAPH(FROM Snapshot TO Graph)",
    """CREATE REL TABLE IF NOT EXISTS IN_SNAPSHOT(
        FROM Op TO Snapshot, FROM BlockArg TO Snapshot, FROM Tensor TO Snapshot
    )""",
    """CREATE REL TABLE IF NOT EXISTS CONTAINS(
        FROM Op TO Op, regionIdx INT64, blockIdx INT64, orderIdx INT64
    )""",
    """CREATE REL TABLE IF NOT EXISTS HAS_ARG(
        FROM Op TO BlockArg, regionIdx INT64, blockIdx INT64, argIdx INT64
    )""",
    """CREATE REL TABLE IF NOT EXISTS PRODUCES(
        FROM Op TO Tensor, FROM BlockArg TO Tensor, resultIdx INT64
    )""",
    "CREATE REL TABLE IF NOT EXISTS CONSUMED_BY(FROM Tensor TO Op, operandIdx INT64)",
    """CREATE REL TABLE IF NOT EXISTS FEEDS(
        FROM Op TO Op, FROM BlockArg TO Op, operandIdx INT64, resultIdx INT64
    )""",
    "CREATE REL TABLE IF NOT EXISTS CALLS(FROM Op TO Op)",
    "CREATE REL TABLE IF NOT EXISTS REFERENCES(FROM Op TO Op, attrName STRING)",
    "CREATE REL TABLE IF NOT EXISTS HAS_ATTR(FROM Op TO Attr, FROM Tensor TO Attr)",
]

# ---------------------------------------------------------------------------
# Navigation queries
# ---------------------------------------------------------------------------

LIST_GRAPHS = "MATCH (r:Graph) RETURN r ORDER BY r.createdAt DESC"

GET_GRAPH = "MATCH (r:Graph {graphId: $graphId}) RETURN r"

# Run (CI run) rollup for the dashboard's lazy landing page: one row per run, where
# an empty runId is the "local" bucket (graphs ingested without CI
# provenance, i.e. dev pushes). numTests is fetched via RUN_TEST_COUNTS and merged in the
# service -- Kuzu nulls out plain aggregates (max/min) when a DISTINCT-counting
# aggregate shares the same RETURN, so distinct counting is kept separate.
# numTests counts per-job test executions (a test run on n150 + p150 counts
# twice), to match the CI "tests passed" total.
LIST_RUNS = (
    "MATCH (r:Graph) "
    "RETURN coalesce(r.runId, '') AS runId, count(r) AS numGraphs, "
    "max(r.createdAt) AS latestCreatedAt, min(r.branch) AS branch, "
    # max() over the per-run-constant workflow fields: lexicographically the
    # empty string loses, so a populated value wins if any graph carries one.
    "max(r.workflowName) AS workflowName, max(r.workflowTitle) AS workflowTitle "
    "ORDER BY latestCreatedAt DESC"
)

RUN_TEST_COUNTS = (
    "MATCH (r:Graph) "
    "WITH DISTINCT coalesce(r.runId, '') AS runId, r.jobName AS jobName, "
    "r.testName AS testName "
    "RETURN runId, count(testName) AS numTests"
)

# One run.s graphs (empty $runId selects the local bucket), grouped into
# tests on the client. Ordered by test, then ascending chronologically so each
# test's graphs read oldest-first.
LIST_GRAPHS_FOR_RUN = (
    "MATCH (r:Graph) WHERE coalesce(r.runId, '') = $runId "
    "RETURN r ORDER BY r.testName, r.createdAt"
)

# Case-insensitive substring search across the human-meaningful Graph fields.
# $q must be lowercased by the caller; coalesce guards null columns.
SEARCH_GRAPHS = (
    "MATCH (r:Graph) WHERE "
    "lower(coalesce(r.modelName, '')) CONTAINS $q OR "
    "lower(coalesce(r.testName, '')) CONTAINS $q OR "
    "lower(coalesce(r.runId, '')) CONTAINS $q OR "
    "lower(coalesce(r.workflowName, '')) CONTAINS $q OR "
    "lower(coalesce(r.workflowTitle, '')) CONTAINS $q OR "
    "lower(coalesce(r.jobName, '')) CONTAINS $q OR "
    "lower(coalesce(r.branch, '')) CONTAINS $q OR "
    "lower(coalesce(r.gitSha, '')) CONTAINS $q "
    "RETURN r ORDER BY r.createdAt DESC"
)

LIST_SNAPSHOTS_FOR_GRAPH = (
    "MATCH (s:Snapshot)-[:IN_GRAPH]->(r:Graph {graphId: $graphId}) "
    "RETURN s ORDER BY s.snapshotIndex"
)

# Backfill workflow provenance for graphs ingested before it was captured (or
# pushed without it). Targets every graph in a run by its runId.
SET_RUN_WORKFLOW = (
    "MATCH (r:Graph) WHERE coalesce(r.runId, '') = $runId "
    "SET r.workflowName = $workflowName, r.workflowTitle = $workflowTitle "
    "RETURN count(r) AS updated"
)

GET_SNAPSHOT_META = "MATCH (s:Snapshot {snapshotId: $snapshotId}) RETURN s"

GET_SNAPSHOT_OPS = "MATCH (o:Op {snapshotId: $snapshotId}) RETURN o"

GET_SNAPSHOT_BLOCKARGS = "MATCH (b:BlockArg {snapshotId: $snapshotId}) RETURN b"

GET_SNAPSHOT_TENSORS = "MATCH (t:Tensor {snapshotId: $snapshotId}) RETURN t"

GET_SNAPSHOT_EDGES = (
    "MATCH (t:Tensor {snapshotId: $snapshotId})-[e:CONSUMED_BY]->(o:Op) "
    "RETURN t.id AS tensorId, o.id AS consumerId, e.operandIdx AS operandIdx"
)

GET_SNAPSHOT_CALLS = (
    "MATCH (caller:Op {snapshotId: $snapshotId})-[:CALLS]->(callee:Op) "
    "RETURN caller.id AS callerId, callee.id AS calleeId"
)

GET_SNAPSHOT_REFS = (
    "MATCH (u:Op {snapshotId: $snapshotId})-[r:REFERENCES]->(t:Op) "
    "RETURN u.id AS userId, t.id AS targetId, r.attrName AS attrName"
)

# Attribute bags, keyed by owning node id (works for both Op and Tensor).
GET_SNAPSHOT_OP_ATTRS = (
    "MATCH (o:Op {snapshotId: $snapshotId})-[:HAS_ATTR]->(a:Attr) "
    "RETURN o.id AS ownerId, a.key AS key, a.val AS val"
)

GET_SNAPSHOT_TENSOR_ATTRS = (
    "MATCH (t:Tensor {snapshotId: $snapshotId})-[:HAS_ATTR]->(a:Attr) "
    "RETURN t.id AS ownerId, a.key AS key, a.val AS val"
)

# ---------------------------------------------------------------------------
# Stats queries
# ---------------------------------------------------------------------------

SNAPSHOT_OP_COUNT = "MATCH (o:Op {snapshotId: $snapshotId}) RETURN count(o) AS numOps"

SNAPSHOT_TENSOR_COUNT = (
    "MATCH (t:Tensor {snapshotId: $snapshotId}) RETURN count(t) AS numTensors"
)

SNAPSHOT_DIALECT_BREAKDOWN = (
    "MATCH (o:Op {snapshotId: $snapshotId}) "
    "RETURN o.dialect AS dialect, count(o) AS n"
)

# ---------------------------------------------------------------------------
# Schema description (handed to the model alongside the run_cypher tool)
# ---------------------------------------------------------------------------

SCHEMA_DESCRIPTION = {
    "overview": (
        "A uniform MLIR graph. Every operation (modules and functions "
        "included) is an Op node, nested via CONTAINS. Dataflow is "
        "Op-[PRODUCES]->Tensor-[CONSUMED_BY]->Op; FEEDS is a derived Op->Op "
        "shortcut over that pair. Arbitrary op/tensor attributes are shared "
        "Attr{key,val} nodes via HAS_ATTR -- match them directly rather than "
        "parsing strings. Note: Kuzu lists are 1-indexed (t.shape[1] is the "
        "first dim). "
        "SCOPING (important for performance): dataflow edges never cross a "
        "snapshot, so scope traversals with a column filter on the start node "
        "-- WHERE a.graphId = $graphId (a whole graph, across its pass snapshots) "
        "or a.snapshotId = $snapshotId (one pass). Op/Tensor/BlockArg carry "
        "both graphId and snapshotId for this. Filter on those columns rather "
        "than traversing Graph<-Snapshot<-Op to reach start nodes -- the column "
        "filter selects a far smaller start set and is dramatically faster. "
        "Unscoped graph-wide traversals scan the entire store; avoid them."
    ),
    "nodeTypes": {
        "Graph": [
            "graphId (PK)",
            "modelName",
            "testName",
            "gitSha",
            "runId",
            "workflowName",
            "workflowTitle",
            "branch",
            "createdAt",
        ],
        "Snapshot": [
            "snapshotId (PK)",
            "graphId",
            "tag",
            "passName",
            "snapshotIndex",
            "timestampUs",
            "sourcePath",
            "graphIndex",
            "mlirPath",
        ],
        "Op": [
            "id (PK)",
            "snapshotId",
            "graphId",
            "parentId",
            "regionIdx",
            "blockIdx",
            "orderIdx",
            "opName",
            "dialect",
            "location",
            "symName",
            "isFunc",
            "numArgs",
            "numResults",
            "isTerminator",
        ],
        "BlockArg": [
            "id (PK)",
            "snapshotId",
            "graphId",
            "parentId",
            "regionIdx",
            "blockIdx",
            "argIdx",
        ],
        "Tensor": [
            "id (PK)",
            "snapshotId",
            "graphId",
            "producerId",
            "producerType",
            "resultIdx",
            "shape (INT64[])",
            "rank",
            "dtype",
        ],
        "Attr": ["id (PK)", "key", "val"],
    },
    "relationships": [
        {"type": "IN_GRAPH", "from": "Snapshot", "to": "Graph"},
        {"type": "IN_SNAPSHOT", "from": "Op|BlockArg|Tensor", "to": "Snapshot"},
        {
            "type": "CONTAINS",
            "from": "Op",
            "to": "Op",
            "properties": ["regionIdx", "blockIdx", "orderIdx"],
        },
        {
            "type": "HAS_ARG",
            "from": "Op",
            "to": "BlockArg",
            "properties": ["regionIdx", "blockIdx", "argIdx"],
        },
        {
            "type": "PRODUCES",
            "from": "Op|BlockArg",
            "to": "Tensor",
            "properties": ["resultIdx"],
        },
        {
            "type": "CONSUMED_BY",
            "from": "Tensor",
            "to": "Op",
            "properties": ["operandIdx"],
        },
        {
            "type": "FEEDS",
            "from": "Op|BlockArg",
            "to": "Op",
            "properties": ["operandIdx", "resultIdx"],
            "note": "derived producer->consumer shortcut",
        },
        {"type": "CALLS", "from": "Op", "to": "Op"},
        {"type": "REFERENCES", "from": "Op", "to": "Op", "properties": ["attrName"]},
        {"type": "HAS_ATTR", "from": "Op|Tensor", "to": "Attr"},
    ],
    "examples": {
        "reshape->permute->broadcast pattern in one graph (scoped, fast)": (
            "MATCH (a:Op)-[:FEEDS]->(b:Op)-[:FEEDS]->(c:Op) "
            "WHERE a.graphId = $graphId "
            "AND a.opName CONTAINS 'reshape' AND b.opName CONTAINS 'permute' "
            "AND c.opName CONTAINS 'broadcast' "
            "RETURN a.location, b.location, c.location"
        ),
        "mesh_shard feeding matmul in one graph": (
            "MATCH (a:Op)-[:FEEDS]->(b:Op) "
            "WHERE a.graphId = $graphId "
            "AND a.opName CONTAINS 'mesh_shard' AND b.opName CONTAINS 'matmul' "
            "RETURN a.location, b.location"
        ),
        "reshape->broadcast->reshape->permute chain in one graph": (
            "MATCH (r1:Op)-[:FEEDS]->(bc:Op)-[:FEEDS]->(r2:Op)-[:FEEDS]->(p:Op) "
            "WHERE r1.graphId = $graphId "
            "AND r1.opName CONTAINS 'reshape' AND bc.opName CONTAINS 'broadcast' "
            "AND r2.opName CONTAINS 'reshape' AND p.opName CONTAINS 'permute' "
            "RETURN r1.location, p.location"
        ),
        "ops with an L1 operand at a given snapshot": (
            "MATCH (o:Op {snapshotId: $sid})<-[:CONSUMED_BY]-(t:Tensor) "
            "MATCH (t)-[:HAS_ATTR]->(:Attr {key: 'buffer_type', val: 'l1'}) "
            "RETURN DISTINCT o.opName, o.location"
        ),
        "4D tensors": ("MATCH (t:Tensor) WHERE size(t.shape) = 4 RETURN t.id, t.shape"),
    },
}
