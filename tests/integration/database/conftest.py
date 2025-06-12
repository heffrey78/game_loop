"""
Configuration for database integration tests.
"""

import os
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from game_loop.config.models import DatabaseConfig
from game_loop.database.session_factory import DatabaseSessionFactory


@pytest.fixture(scope="session")
def database_config() -> DatabaseConfig:
    """Create database configuration for testing."""
    return DatabaseConfig(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        username=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
        database=os.environ.get("POSTGRES_DB", "game_loop"),
        echo=True,  # Enable SQL logging for tests
    )


@pytest_asyncio.fixture(scope="function")
async def session_factory(
    database_config: DatabaseConfig,
) -> AsyncGenerator[DatabaseSessionFactory, None]:
    """Create a session factory for testing."""
    factory = DatabaseSessionFactory(database_config)
    await factory.initialize()
    yield factory
    await factory.close()


@pytest_asyncio.fixture
async def db_session(
    session_factory: DatabaseSessionFactory,
) -> AsyncGenerator[AsyncSession, None]:
    """Create a database session for testing with transaction rollback."""
    # Create session directly to avoid auto-commit behavior
    session = session_factory._session_factory()
    try:
        # Start a nested transaction for test isolation
        trans = await session.begin()
        try:
            yield session
        finally:
            # Rollback if transaction is still active (not already closed by an error)
            if trans.is_active:
                await trans.rollback()
    finally:
        # Close the session
        await session.close()


@pytest.fixture
def skip_if_no_db():
    """Skip test if database is not available."""
    try:
        import importlib.util

        # Check if asyncpg is available
        if importlib.util.find_spec("asyncpg") is None:
            pytest.skip("asyncpg not available")

        # Could add additional database connectivity check here
        return False
    except ImportError:
        pytest.skip("Database integration tests require asyncpg")
