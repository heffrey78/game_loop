"""Basic ORM integration tests."""

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from game_loop.database.models.base import Base


@pytest.mark.asyncio
async def test_database_connection(db_session: AsyncSession) -> None:
    """Test basic database connectivity."""
    result = await db_session.execute(text("SELECT 1"))
    assert result.scalar() == 1


@pytest.mark.asyncio
async def test_extensions_available(db_session: AsyncSession) -> None:
    """Test that required PostgreSQL extensions are available."""
    # Check for pgvector extension
    result = await db_session.execute(
        text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
    )
    has_vector = result.scalar()
    # The extension should be installed
    assert has_vector == 1, "pgvector extension should be available"


@pytest.mark.asyncio
async def test_create_tables(db_session: AsyncSession) -> None:
    """Test that we can create tables from the ORM models."""
    # This will test if the models are properly defined
    # The actual table creation happens in the database setup
    # Here we just verify we can access the metadata
    assert Base.metadata is not None
    assert len(Base.metadata.tables) > 0
