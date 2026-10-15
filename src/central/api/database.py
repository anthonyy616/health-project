"""
Central backend persistence layer (Phase K)
===========================================

SQLAlchemy (SQLite default; MySQL-compatible URL supported via config)
plus a minimal schema_version-based migration runner (no alembic
dependency for the research prototype; migrations are explicit for the
demo scope).

Documents in the repo contain only the schema; no database credentials -
connection URLs come from the environment (`VITALS_DB_URL`), never
committed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "vitals.sqlite3"
SCHEMA_VERSION = 2  # bumped when models change; see run_migrations()


def resolve_db_url(db_url: Optional[str] = None) -> str:
    """Env-overridable SQLite default; keeps credentials out of the repo."""
    url = db_url or os.environ.get("VITALS_DB_URL")
    if url:
        return url
    DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{DEFAULT_DB_PATH}"


def make_engine(db_url: Optional[str] = None) -> Engine:
    engine = create_engine(
        resolve_db_url(db_url), connect_args={"check_same_thread": False}
    )

    if engine.url.get_backend_name() == "sqlite":
        @event.listens_for(engine, "connect")
        def _fk_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def make_session_factory(engine: Engine) -> sessionmaker:
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def run_migrations(engine: Engine) -> int:
    """Versioned schema initialization/upgrade (idempotent).

    Returns the applied schema version. v1 created sessions/measurements;
    v2 adds alerts and a schema_version bookkeeping table. Demo-scope
    upgrades recreate the derived tables; research data stays in the
    version-controlled migrations scripts rather than ad-hoc ALTERs.
    """
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS schema_version ("
            " id INTEGER PRIMARY KEY, version INTEGER NOT NULL)"
        ))
        row = conn.execute(text("SELECT version FROM schema_version ORDER BY id DESC LIMIT 1")).first()
        current = int(row[0]) if row else 0

    from src.central.api.models import Base

    if current < SCHEMA_VERSION:
        # create_all is additive; for demo scope no destructive rewrites.
        Base.metadata.create_all(bind=engine)
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM schema_version"))
            conn.execute(
                text("INSERT INTO schema_version (version) VALUES (:v)"),
                {"v": SCHEMA_VERSION},
            )
            if current > 0:
                logger.info("database upgraded schema %d -> %d", current, SCHEMA_VERSION)
            else:
                logger.info("database initialized at schema v%d", SCHEMA_VERSION)
    else:
        Base.metadata.create_all(bind=engine)
    return SCHEMA_VERSION
