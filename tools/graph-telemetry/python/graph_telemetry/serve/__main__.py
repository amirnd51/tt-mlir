# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CLI entry point for mlir-graph-serve."""

from __future__ import annotations

import argparse
import logging
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="mlir-graph-serve: Graph telemetry query server"
    )
    parser.add_argument(
        "--db",
        required=True,
        help="Kuzu DSN string, e.g. 'kuzu:./telemetry.kuzu'",
    )
    parser.add_argument(
        "--ingest-dir",
        help="Auto-ingest all JSON files from this directory on startup",
    )
    parser.add_argument(
        "--files-dir",
        help="Directory to serve snapshot .mlir dumps from via /files "
        "(defaults to --ingest-dir)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8321, help="Bind port (default: 8321)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Import here to avoid import-time side effects
    import uvicorn

    from .app import app, init_db, init_files_root

    init_db(args.db)
    # Serve snapshot .mlir dumps from the ingest directory (Snapshot.mlirPath is
    # relative to it).
    init_files_root(args.files_dir or args.ingest_dir)

    if args.ingest_dir:
        from .ingest import ingest_dir

        db = app.state.db
        results = ingest_dir(db, args.ingest_dir)
        for r in results:
            logging.info(
                "Ingested graph=%s snapshots=%d",
                r["graphId"],
                r["snapshotsIngested"],
            )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
