"""SQLAlchemy async session factory."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from game_loop.config.models import DatabaseConfig

logger = logging.getLogger(__name__)


class DatabaseSessionFactory:
    """Factory for creating database sessions."""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._engine = None
        self._session_factory = None

    async def initialize(self) -> None:
        """Initialize the database engine and session factory."""
        database_url = (
            f"postgresql+asyncpg://{self.config.username}:"
            f"{self.config.password}@{self.config.host}:"
            f"{self.config.port}/{self.config.database}"
        )

        engine_kwargs = {
            "echo": self.config.echo,
            "pool_recycle": self.config.pool_recycle,
        }

        # Only add pool configuration for QueuePool (default), not StaticPool
        if self.config.echo:
            # For testing with StaticPool, don't add pool size parameters
            engine_kwargs["poolclass"] = StaticPool
        else:
            # For production with QueuePool
            engine_kwargs.update(
                {
                    "pool_size": self.config.pool_size,
                    "max_overflow": self.config.max_overflow,
                    "pool_timeout": self.config.pool_timeout,
                }
            )

        self._engine = create_async_engine(database_url, **engine_kwargs)

        self._session_factory = async_sessionmaker(
            bind=self._engine, class_=AsyncSession, expire_on_commit=False
        )

        logger.info("Database session factory initialized")

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an async database session."""
        if not self._session_factory:
            raise RuntimeError("Session factory not initialized")

        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def close(self) -> None:
        """Close the database engine."""
        if self._engine:
            await self._engine.dispose()
            logger.info("Database session factory closed")
