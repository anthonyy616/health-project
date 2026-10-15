"""
Central monitoring backend (Phase K)
====================================

FastAPI application: session management, measurement ingestion (with the
Measurement envelope's validation semantics), history retrieval,
WebSocket live updates, threshold/quality/disconnect alerts and SQLite
persistence.

Boundary preserved: no physiological signal processing here - this layer
persists and broadcasts values that acquisition/processing/fusion produce
(ingest may run a freshness check via the shared Measurement helpers, but
never computes new vitals).

Run: `python -m src.main --mode dashboard` (wired), or
`uvicorn src.central.api.app:create_default_app --factory --reload`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Set

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from sqlalchemy import select
from sqlalchemy.orm import Session as DBSession

from src.central.api.auth import TokenGuard
from src.central.api.database import (
    DEFAULT_DB_PATH,
    SCHEMA_VERSION,
    make_engine,
    make_session_factory,
    run_migrations,
)
from src.central.api.models import AlertRow, Base, MeasurementRow, Session
from src.central.api.schemas import (
    AlertIn,
    MeasurementIn,
    MeasurementOut,
    SessionCreate,
    SessionOut,
)

logger = logging.getLogger(__name__)

NONCLINICAL_DISCLAIMER = (
    "Research prototype for a final-year engineering project. NOT a medical "
    "device; values are not clinically validated and are not for diagnosis "
    "or treatment."
)


def now_utc() -> float:
    return time.time()


def _row_to_out(row: MeasurementRow) -> MeasurementOut:
    return MeasurementOut(
        metric=row.metric,
        status=row.status,
        value=row.value,
        unit=row.unit,
        source=row.source,
        acquired_at_utc=row.acquired_at_utc,
        signal_quality=row.signal_quality,
        confidence=row.confidence,
        failure_reason=row.failure_reason,
        latency_ms=row.latency_ms,
        frame_number=row.frame_number,
        provenance=json.loads(row.provenance) if row.provenance else {},
    )


class MeasurementBroadcaster:
    """Fan-out of new measurements to live WebSocket subscribers."""

    def __init__(self):
        self._loop_lock_sets: Dict[str, Set[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    def register(self, session_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._loop_lock_sets.setdefault(session_id, set()).add(queue)
        return queue

    def unregister(self, session_id: str, queue: asyncio.Queue) -> None:
        self._loop_lock_sets.get(session_id, set()).discard(queue)

    def publish(self, session_id: str, measurement: dict) -> int:
        queues = self._loop_lock_sets.get(session_id, set())
        delivered = 0
        for queue in queues:
            try:
                queue.put_nowait(measurement)
                delivered += 1
            except asyncio.QueueFull:
                logger.warning("slow websocket subscriber dropped a frame")
        return delivered


def create_app(
    engine=None,
    auth_token: Optional[str] = None,
) -> FastAPI:
    """App factory with dependency-injected persistence (testing-friendly)."""
    engine = engine if engine is not None else make_engine()
    run_migrations(engine)
    session_factory = make_session_factory(engine)

    guard = TokenGuard(auth_token)
    broadcaster = MeasurementBroadcaster()

    app = FastAPI(
        title="Contactless Vitals Monitoring — Backend (research prototype)",
        description=NONCLINICAL_DISCLAIMER,
        version=str(SCHEMA_VERSION),
    )

    def db() -> DBSession:
        session = session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _require_session(db: DBSession, session_id: str) -> Session:
        found = db.get(Session, session_id)
        if found is None:
            raise HTTPException(status_code=404, detail="session not found")
        return found

    # -- health ------------------------------------------------------------
    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "schema_version": SCHEMA_VERSION,
            "nonclinical_disclaimer": NONCLINICAL_DISCLAIMER,
        }

    # -- sessions ------------------------------------------------------------
    @app.post("/sessions", response_model=SessionOut, status_code=201, dependencies=[Depends(guard)])
    def create_session(payload: SessionCreate, db: DBSession = Depends(db)):
        record = Session(display_name=payload.display_name, consent_recorded=payload.consent_recorded)
        db.add(record)
        db.flush()
        db.commit()
        db.refresh(record)
        return SessionOut(
            id=record.id,
            display_name=record.display_name,
            started_at=record.started_at.timestamp(),
            ended_at=None,
            consent_recorded=record.consent_recorded,
        )

    @app.get("/sessions", response_model=List[SessionOut], dependencies=[Depends(guard)])
    def list_sessions(db: DBSession = Depends(db)):
        rows = db.scalars(select(Session).order_by(Session.started_at.desc())).all()
        return [
            SessionOut(
                id=r.id,
                display_name=r.display_name,
                started_at=r.started_at.timestamp(),
                ended_at=r.ended_at.timestamp() if r.ended_at else None,
                consent_recorded=r.consent_recorded,
            )
            for r in rows
        ]

    @app.get("/sessions/{session_id}", response_model=SessionOut, dependencies=[Depends(guard)])
    def get_session(session_id: str, db: DBSession = Depends(db)):
        record = _require_session(db, session_id)
        return SessionOut(
            id=record.id,
            display_name=record.display_name,
            started_at=record.started_at.timestamp(),
            ended_at=record.ended_at.timestamp() if record.ended_at else None,
            consent_recorded=record.consent_recorded,
        )

    @app.delete("/sessions/{session_id}", status_code=204, dependencies=[Depends(guard)])
    def delete_session(session_id: str, db: DBSession = Depends(db)):
        record = _require_session(db, session_id)
        db.delete(record)

    # -- measurement ingestion / history -------------------------------------
    @app.post(
        "/sessions/{session_id}/measurements",
        response_model=MeasurementOut,
        status_code=201,
        dependencies=[Depends(guard)],
    )
    def ingest_measurement(
        session_id: str,
        payload: MeasurementIn,
        db: DBSession = Depends(db),
    ):
        _require_session(db, session_id)
        acquired = payload.acquired_at_utc if payload.acquired_at_utc is not None else now_utc()
        from sqlalchemy.dialects import sqlite
        from src.central.api.models import MeasurementRow as MR

        # Upsert: if a row for this session/metric/timestamp already exists,
        # update its payload in place so re-ingestion is idempotent.
        stmt = sqlite.insert(MR).values(
            session_id=session_id,
            metric=payload.metric,
            value=payload.value,
            unit=payload.unit,
            source=payload.source,
            status=payload.status,
            acquired_at_utc=acquired,
            acquired_mono_ns=payload.acquired_mono_ns,
            computed_at_utc=now_utc(),
            signal_quality=payload.signal_quality,
            confidence=payload.confidence,
            failure_reason=payload.failure_reason,
            latency_ms=payload.latency_ms,
            frame_number=payload.frame_number,
            provenance=json.dumps(payload.provenance),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["session_id", "metric", "acquired_at_utc"],
            set_=dict(
                value=payload.value,
                unit=payload.unit,
                source=payload.source,
                status=payload.status,
                acquired_at_utc=acquired,
                acquired_mono_ns=payload.acquired_mono_ns,
                computed_at_utc=now_utc(),
                signal_quality=payload.signal_quality,
                confidence=payload.confidence,
                failure_reason=payload.failure_reason,
                latency_ms=payload.latency_ms,
                frame_number=payload.frame_number,
                provenance=json.dumps(payload.provenance),
            ),
        )
        db.execute(stmt)
        db.commit()
        # Re-fetch the row via the ORM to return a consistent response.
        row = db.query(MR).filter(
            MR.session_id == session_id,
            MR.metric == payload.metric,
            MR.acquired_at_utc == acquired,
        ).order_by(MR.row_id.desc()).first()
        broadcaster.publish(session_id, _row_to_out(row).model_dump())
        return _row_to_out(row)

    @app.get(
        "/sessions/{session_id}/measurements",
        response_model=List[MeasurementOut],
        dependencies=[Depends(guard)],
    )
    def measurement_history(
        session_id: str,
        metric: Optional[str] = None,
        limit: int = 500,
        db: DBSession = Depends(db),
    ):
        _require_session(db, session_id)
        stmt = select(MeasurementRow).where(MeasurementRow.session_id == session_id)
        if metric is not None:
            stmt = stmt.where(MeasurementRow.metric == metric)
        stmt = stmt.order_by(MeasurementRow.acquired_at_utc.desc()).limit(min(limit, 5000))
        rows = db.scalars(stmt).all()
        return [_row_to_out(r) for r in rows]

    # -- live websocket --------------------------------------------------------
    @app.websocket("/ws/sessions/{session_id}")
    async def live_feed(websocket: WebSocket, session_id: str):
        await websocket.accept()
        _require_ws_session(session_id)
        queue = broadcaster.register(session_id)
        try:
            while True:
                payload = await queue.get()
                await websocket.send_json(payload)
        except WebSocketDisconnect:
            pass
        finally:
            broadcaster.unregister(session_id, queue)

    def _require_ws_session(session_id: str):
        # Same lookup as HTTP, without the sync dependency chain.
        with session_factory() as probe:
            if probe.get(Session, session_id) is None:
                raise HTTPException(status_code=404, detail="session not found")

    # -- alerts -----------------------------------------------------------------
    @app.post("/sessions/{session_id}/alerts", status_code=201, dependencies=[Depends(guard)])
    def create_alert(session_id: str, payload: AlertIn, db: DBSession = Depends(db)):
        _require_session(db, session_id)
        row = AlertRow(
            session_id=session_id,
            severity=payload.severity,
            kind=payload.kind,
            message=payload.message,
            metric=payload.metric,
            value=payload.value,
        )
        db.add(row)
        db.flush()
        db.commit()
        db.refresh(row)
        broadcaster.publish(session_id, {
            "type": "alert",
            "severity": row.severity,
            "kind": row.kind,
            "message": row.message,
            "metric": row.metric,
            "value": row.value,
            "created_at": row.created_at.timestamp(),
        })
        return {"row_id": row.row_id, "created_at": row.created_at.timestamp()}

    @app.get("/sessions/{session_id}/alerts", dependencies=[Depends(guard)])
    def list_alerts(session_id: str, limit: int = 100, db: DBSession = Depends(db)):
        _require_session(db, session_id)
        stmt = (
            select(AlertRow)
            .where(AlertRow.session_id == session_id)
            .order_by(AlertRow.created_at.desc())
            .limit(min(limit, 1000))
        )
        rows = db.scalars(stmt).all()
        return [
            {
                "row_id": r.row_id,
                "severity": r.severity,
                "kind": r.kind,
                "message": r.message,
                "metric": r.metric,
                "value": r.value,
                "created_at": r.created_at.timestamp(),
            }
            for r in rows
        ]

    return app


def create_default_app() -> FastAPI:
    """Entry point for `uvicorn src.central.api.app:create_default_app`."""
    import os

    return create_app(auth_token=os.environ.get("VITALS_API_TOKEN"))
