"""SQLAlchemy models for player-related entities."""

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    INTEGER,
    TEXT,
    UUID,
    VARCHAR,
    ForeignKey,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin, TimestampWithTimezone, UUIDMixin


class Player(Base, UUIDMixin, TimestampMixin):
    """Represents a player in the game."""

    __tablename__ = "players"

    name: Mapped[str] = mapped_column(VARCHAR(50), nullable=False)
    username: Mapped[str] = mapped_column(VARCHAR(50), nullable=False)
    last_login: Mapped[datetime | None] = mapped_column(
        TimestampWithTimezone(), nullable=True
    )
    settings_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=True)
    current_location_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("locations.id"), nullable=True
    )
    level: Mapped[int] = mapped_column(
        INTEGER, nullable=False, default=1, server_default="1"
    )
    # Other player attributes (e.g., experience, health)
    # will be addressed if specific TypeErrors arise for them.

    # Relationships
    inventory_items = relationship("PlayerInventory", back_populates="player")
    knowledge_items = relationship("PlayerKnowledge", back_populates="player")
    skills = relationship("PlayerSkill", back_populates="player")
    history_events = relationship("PlayerHistory", back_populates="player")


class PlayerInventory(Base, UUIDMixin, TimestampMixin):
    """Represents items in a player's inventory."""

    __tablename__ = "player_inventories"

    player_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("players.id"), nullable=False
    )
    object_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("objects.id"), nullable=False
    )
    quantity: Mapped[int] = mapped_column(INTEGER, nullable=False, default=1)
    acquired_at: Mapped[datetime] = mapped_column(
        TimestampWithTimezone(), nullable=False
    )
    state_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # Relationships
    player = relationship("Player", back_populates="inventory_items")
    object = relationship("Object")


class PlayerKnowledge(Base, UUIDMixin, TimestampMixin):
    """Represents knowledge acquired by a player."""

    __tablename__ = "player_knowledge"

    player_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("players.id"), nullable=False
    )
    knowledge_key: Mapped[str] = mapped_column(VARCHAR(100), nullable=False)
    knowledge_value: Mapped[str] = mapped_column(TEXT, nullable=False)
    discovered_at: Mapped[datetime] = mapped_column(
        TimestampWithTimezone(), nullable=False
    )
    knowledge_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(384), nullable=True
    )

    # Relationships
    player = relationship("Player", back_populates="knowledge_items")


class PlayerSkill(Base, UUIDMixin, TimestampMixin):
    """Represents a player's skills."""

    __tablename__ = "player_skills"

    player_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("players.id"), nullable=False
    )
    skill_name: Mapped[str] = mapped_column(VARCHAR(50), nullable=False)
    skill_level: Mapped[int] = mapped_column(INTEGER, nullable=False)
    skill_description: Mapped[str | None] = mapped_column(TEXT, nullable=True)
    skill_category: Mapped[str | None] = mapped_column(VARCHAR(50), nullable=True)

    # Relationships
    player = relationship("Player", back_populates="skills")


class PlayerHistory(Base, UUIDMixin, TimestampMixin):
    """Represents a player's action history."""

    __tablename__ = "player_histories"

    player_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("players.id"), nullable=False
    )
    event_timestamp: Mapped[datetime] = mapped_column(
        TimestampWithTimezone(), nullable=False
    )
    event_type: Mapped[str] = mapped_column(VARCHAR(50), nullable=False)
    event_data: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    location_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("locations.id"), nullable=True
    )
    event_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(384), nullable=True
    )

    # Relationships
    player = relationship("Player", back_populates="history_events")
    location = relationship("Location")
