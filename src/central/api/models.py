"""
ORM models for the central backend (Phase K).

Separation of concerns: acquisition/processing (src/contactless,
src/wearable) produce Measurement envelopes; this layer only persists and
retrieves them. No physiological signal processing lives here.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _utcnow():
    return datetime.now(timezone.utc)


class Session(Base):
    """One monitoring session - the boundary each user/device belongs to.

    Human-readable display name only; anonymized by default (no PII
    columns). Consent/access decisions stay out of the ORM layer.
    """

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    display_name: Mapped[str] = mapped_column(String(120), default="Unnamed session")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    # Consent flag is metadata only - enforcement lives at the API surface.
    consent_recorded: Mapped[bool] = mapped_column(Boolean, default=False)

    measurements: Mapped[list["MeasurementRow"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    alerts: Mapped[list["AlertRow"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class MeasurementRow(Base):
    """Persisted Measurement envelope (one row per metric reading)."""

    __tablename__ = "measurements"
    __table_args__ = (
        UniqueConstraint("session_id", "metric", "acquired_at_utc", name="uq_measurement_dedup"),
    )

    row_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), index=True)
    metric: Mapped[str] = mapped_column(String(40), index=True)
    value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    unit: Mapped[str] = mapped_column(String(24))
    source: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(24), index=True)
    # Numeric columns (float seconds for wall clock; int ns for monotonic)
    acquired_at_utc: Mapped[float] = mapped_column(Float, index=True)
    acquired_mono_ns: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    computed_at_utc: Mapped[float] = mapped_column(Float)
    signal_quality: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    failure_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    frame_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    provenance: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON string

    session: Mapped["Session"] = relationship(back_populates="measurements")


class AlertRow(Base):
    """Threshold/disconnect/quality alerts (non-clinical, rule-based)."""

    __tablename__ = "alerts"

    row_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), index=True)
    severity: Mapped[str] = mapped_column(String(16))  # info | warning | critical
    kind: Mapped[str] = mapped_column(String(32))      # threshold | quality | disconnect | stale
    message: Mapped[str] = mapped_column(Text)
    metric: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)

    session: Mapped["Session"] = relationship(back_populates="alerts")
