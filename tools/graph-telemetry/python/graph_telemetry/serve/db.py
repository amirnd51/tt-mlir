# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""GraphDB abstraction layer backed by embedded Kuzu."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from . import queries as Q

logger = logging.getLogger(__name__)


class GraphDB(ABC):
    """Abstract base class for graph database backends."""

    @abstractmethod
    def execute(self, cypher: str, params: dict | None = None) -> list[dict]:
        """Execute a read query and return rows as list of dicts."""
        ...

    @abstractmethod
    def execute_write(self, cypher: str, params: dict | None = None) -> None:
        """Execute a write query (CREATE, MERGE, etc.)."""
        ...

    @abstractmethod
    def bulk_copy_rel(
        self,
        table: str,
        rows: list[dict],
        from_tbl: str | None = None,
        to_tbl: str | None = None,
    ) -> None:
        """Bulk-create relationships from `rows` (each {f, t, ...props}).

        Far faster than UNWIND ... MATCH ... CREATE for large batches, which
        the planner executes as a full node-table scan + cross product. `from_tbl`
        / `to_tbl` name the endpoint node tables for multi-pair rel tables.
        """
        ...

    @abstractmethod
    def get_schema(self) -> dict:
        """Return the DB schema description."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the database connection."""
        ...

    @abstractmethod
    def ensure_schema(self) -> None:
        """Create node/relationship tables and indexes if they don't exist."""
        ...


class KuzuDB(GraphDB):
    """Embedded Kuzu database for local development."""

    def __init__(self, db_path: str) -> None:
        import kuzu

        self._db = kuzu.Database(db_path)
        self._conn = kuzu.Connection(self._db)
        self._schema_created = False

    def execute(self, cypher: str, params: dict | None = None) -> list[dict]:
        try:
            result = self._conn.execute(cypher, parameters=params or {})
            rows = []
            while result.has_next():
                row = result.get_next()
                column_names = result.get_column_names()
                row_dict = {}
                for i, name in enumerate(column_names):
                    val = row[i]
                    # Kuzu returns node values as dicts already in some cases
                    if hasattr(val, "__dict__"):
                        val = dict(val.__dict__)
                    row_dict[name] = val
                rows.append(row_dict)
            return rows
        except Exception as e:
            logger.error("Kuzu query failed: %s\nQuery: %s", e, cypher)
            raise

    def execute_write(self, cypher: str, params: dict | None = None) -> None:
        try:
            self._conn.execute(cypher, parameters=params or {})
        except Exception as e:
            logger.error("Kuzu write failed: %s\nQuery: %s", e, cypher)
            raise

    def bulk_copy_rel(
        self,
        table: str,
        rows: list[dict],
        from_tbl: str | None = None,
        to_tbl: str | None = None,
    ) -> None:
        if not rows:
            return
        import pandas as pd

        # `df` is resolved by Kuzu's pandas replacement scan from this frame's
        # locals, so the variable must be named `df`. Columns are positional:
        # first two are the FROM/TO primary keys, the rest are rel properties.
        df = pd.DataFrame(rows)  # noqa: F841 - referenced by name in COPY
        spec = f' (from="{from_tbl}", to="{to_tbl}")' if from_tbl else ""
        try:
            self._conn.execute(f"COPY {table} FROM df{spec}")
        except Exception as e:
            logger.error("Kuzu COPY into %s failed: %s", table, e)
            raise

    def ensure_schema(self) -> None:
        if self._schema_created:
            return
        for stmt in Q.KUZU_CREATE_NODE_TABLES:
            self._conn.execute(stmt)
        for stmt in Q.KUZU_CREATE_REL_TABLES:
            self._conn.execute(stmt)
        self._schema_created = True
        logger.info("Kuzu schema created/verified")

    def get_schema(self) -> dict:
        return Q.SCHEMA_DESCRIPTION

    def close(self) -> None:
        # Kuzu Connection doesn't need explicit close; Database does on GC
        pass


def connect(dsn: str) -> GraphDB:
    """Parse a DSN string and return a GraphDB backend.

    Supported formats:
      - kuzu:./path/to/db       (relative path)
      - kuzu:/absolute/path     (absolute path)

    A bare filesystem path (no scheme) is also accepted and treated as Kuzu.
    """
    db_path = dsn[len("kuzu:") :] if dsn.startswith("kuzu:") else dsn
    logger.info("Connecting to Kuzu at %s", db_path)
    return KuzuDB(db_path)
