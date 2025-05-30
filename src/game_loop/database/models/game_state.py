"""SQLAlchemy models for game state entities."""

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BOOLEAN,
    INTEGER,
    TEXT,
    UUID,
    VARCHAR,
    ForeignKey,
    LargeBinary,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin, UUIDMixin


class GameSession(Base, TimestampMixin):
    """Represents a game session."""

    __tablename__ = "game_sessions"

    # Note: Using session_id as primary key to match existing schema
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
        nullable=False,
    )
    player_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("players.id"), nullable=False
    )
    player_state_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    world_state_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    save_name: Mapped[str] = mapped_column(VARCHAR(100), default="New Save")
    # created_at and updated_at are provided by TimestampMixin
    # started_at is specific to the session's logical start,
    # distinct from row creation.
    started_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    game_version: Mapped[str] = mapped_column(VARCHAR(20), default="0.1.0")
    game_time: Mapped[int | None] = mapped_column(INTEGER, nullable=True)
    save_data: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    session_summary: Mapped[str | None] = mapped_column(TEXT, nullable=True)

    # Relationships
    player = relationship("Player")


class WorldRule(Base, UUIDMixin, TimestampMixin):
    """Represents rules governing world behavior."""

    __tablename__ = "world_rules"

    rule_name: Mapped[str] = mapped_column(VARCHAR(100), nullable=False)
    rule_description: Mapped[str | None] = mapped_column(TEXT, nullable=True)
    rule_type: Mapped[str | None] = mapped_column(VARCHAR(30), nullable=True)
    created_by: Mapped[str | None] = mapped_column(VARCHAR(50), nullable=True)
    rule_logic: Mapped[str | None] = mapped_column(TEXT, nullable=True)
    rule_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(384), nullable=True
    )


class EvolutionEvent(Base, UUIDMixin, TimestampMixin):
    """Represents events in the world evolution queue."""

    __tablename__ = "evolution_events"

    event_type: Mapped[str] = mapped_column(VARCHAR(50), nullable=False)
    target_type: Mapped[str | None] = mapped_column(VARCHAR(30), nullable=True)
    target_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    scheduled_time: Mapped[int | None] = mapped_column(INTEGER, nullable=True)
    priority: Mapped[int] = mapped_column(INTEGER, default=0)
    conditions_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    effects_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    is_processed: Mapped[bool] = mapped_column(BOOLEAN, default=False)
