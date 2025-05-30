"""SQLAlchemy models for world-related entities."""

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BOOLEAN,
    TEXT,
    UUID,
    VARCHAR,
    ForeignKey,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin, TimestampWithTimezone, UUIDMixin


class Region(Base, UUIDMixin, TimestampMixin):
    """Represents a region in the game world."""

    __tablename__ = "regions"

    name: Mapped[str] = mapped_column(VARCHAR(100), nullable=False)
    description: Mapped[str | None] = mapped_column(TEXT, nullable=True)
    theme: Mapped[str | None] = mapped_column(VARCHAR(50), nullable=True)
    parent_region_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("regions.id"), nullable=True
    )
    region_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(384), nullable=True
    )

    # Relationships
    locations = relationship("Location", back_populates="region")
    child_regions = relationship("Region")


class Location(Base, UUIDMixin, TimestampMixin):
    """Represents a location in the game world."""

    __tablename__ = "locations"

    name: Mapped[str] = mapped_column(VARCHAR(100), nullable=False)
    short_desc: Mapped[str | None] = mapped_column(TEXT, nullable=True)
    full_desc: Mapped[str | None] = mapped_column(TEXT, nullable=True)
    region_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("regions.id"), nullable=True
    )
    location_type: Mapped[str | None] = mapped_column(VARCHAR(50), nullable=True)
    is_dynamic: Mapped[bool] = mapped_column(BOOLEAN, default=False)
    discovery_timestamp: Mapped[datetime | None] = mapped_column(
        TimestampWithTimezone(), nullable=True
    )
    created_by: Mapped[str | None] = mapped_column(VARCHAR(50), nullable=True)
    state_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    location_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(384), nullable=True
    )

    # Relationships
    region = relationship("Region", back_populates="locations")
    objects = relationship("Object", back_populates="location")
    npcs = relationship("NPC", back_populates="location")
    connections_from = relationship(
        "LocationConnection",
        foreign_keys="LocationConnection.from_location_id",
        back_populates="from_location",
    )
    connections_to = relationship(
        "LocationConnection",
        foreign_keys="LocationConnection.to_location_id",
        back_populates="to_location",
    )


class Object(Base, UUIDMixin, TimestampMixin):
    """Represents an object in the game world."""

    __tablename__ = "objects"

    name: Mapped[str] = mapped_column(VARCHAR(100), nullable=False)
    short_desc: Mapped[str | None] = mapped_column(TEXT, nullable=True)
    full_desc: Mapped[str | None] = mapped_column(TEXT, nullable=True)
    object_type: Mapped[str | None] = mapped_column(VARCHAR(50), nullable=True)
    properties_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    is_takeable: Mapped[bool] = mapped_column(BOOLEAN, default=False)
    location_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("locations.id"), nullable=True
    )
    state_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    object_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(384), nullable=True
    )

    # Relationships
    location = relationship("Location", back_populates="objects")


class NPC(Base, UUIDMixin, TimestampMixin):
    """Represents a non-player character."""

    __tablename__ = "npcs"

    name: Mapped[str] = mapped_column(VARCHAR(100), nullable=False)
    short_desc: Mapped[str | None] = mapped_column(TEXT, nullable=True)
    full_desc: Mapped[str | None] = mapped_column(TEXT, nullable=True)
    npc_type: Mapped[str | None] = mapped_column(VARCHAR(50), nullable=True)
    personality_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )
    knowledge_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    location_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("locations.id"), nullable=True
    )
    state_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    npc_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(384), nullable=True
    )

    # Relationships
    location = relationship("Location", back_populates="npcs")


class Quest(Base, UUIDMixin, TimestampMixin):
    """Represents a quest in the game."""

    __tablename__ = "quests"

    title: Mapped[str] = mapped_column(VARCHAR(100), nullable=False)
    description: Mapped[str | None] = mapped_column(TEXT, nullable=True)
    quest_type: Mapped[str | None] = mapped_column(VARCHAR(50), nullable=True)
    status: Mapped[str | None] = mapped_column(VARCHAR(30), nullable=True)
    requirements_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )
    steps_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    rewards_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    is_dynamic: Mapped[bool] = mapped_column(BOOLEAN, default=False)
    quest_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(384), nullable=True
    )


class LocationConnection(Base, UUIDMixin, TimestampMixin):
    """Represents connections between locations."""

    __tablename__ = "location_connections"

    from_location_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("locations.id"), nullable=False
    )
    to_location_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("locations.id"), nullable=False
    )
    direction: Mapped[str | None] = mapped_column(VARCHAR(50), nullable=True)
    description: Mapped[str | None] = mapped_column(TEXT, nullable=True)
    is_visible: Mapped[bool] = mapped_column(BOOLEAN, default=True)
    requirements_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )

    # Relationships
    from_location = relationship(
        "Location", foreign_keys=[from_location_id], back_populates="connections_from"
    )
    to_location = relationship(
        "Location", foreign_keys=[to_location_id], back_populates="connections_to"
    )
