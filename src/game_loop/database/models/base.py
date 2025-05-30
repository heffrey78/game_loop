"""Base SQLAlchemy model with common functionality."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import UUID, TypeDecorator, text
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class TimestampWithTimezone(TypeDecorator):
    """Custom type to ensure TIMESTAMP WITH TIME ZONE in PostgreSQL."""

    impl = TIMESTAMP
    cache_ok = True

    def load_dialect_impl(self, dialect: Any) -> Any:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(TIMESTAMP(timezone=True))
        return dialect.type_descriptor(TIMESTAMP)


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    type_annotation_map = {
        dict[str, Any]: JSONB,
        uuid.UUID: UUID(as_uuid=True),
        datetime: TimestampWithTimezone(),
    }


class TimestampMixin:
    """Mixin for models that need created_at/updated_at timestamps."""

    created_at: Mapped[datetime] = mapped_column(
        TimestampWithTimezone(),
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        TimestampWithTimezone(),
        server_default=text("CURRENT_TIMESTAMP"),
        onupdate=text("clock_timestamp()"),
        nullable=False,
    )


class UUIDMixin:
    """Mixin for models that use UUID primary keys."""

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
        nullable=False,
    )


class DatabaseError(Exception):
    """Base exception for database operations."""

    pass


class EntityNotFoundError(DatabaseError):
    """Raised when an entity is not found."""

    pass


class ValidationError(DatabaseError):
    """Raised when data validation fails."""

    pass
