"""
Database connection utilities for the game loop application.
This module provides functions for creating database connections and connection pools.

This component is part of the Data Layer as shown in the architecture diagram
(docs/architecture-diagram.mmd) and implements the PostgreSQL connection handling
as specified in the tech stack (docs/tech-stack.md).

The connection pool pattern used here supports the async I/O patterns
mentioned in the tech stack documentation, and provides efficient
database access for the GameStateManager component.
"""

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import asyncpg
from asyncpg.pool import Pool

logger = logging.getLogger(__name__)

# Default connection parameters from environment variables
# as specified in the tech stack document
DEFAULT_DB_CONFIG = {
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "port": int(os.environ.get("POSTGRES_PORT", "5432")),
    "user": os.environ.get("POSTGRES_USER", "postgres"),
    "password": os.environ.get("POSTGRES_PASSWORD", "postgres"),
    "database": os.environ.get("POSTGRES_DB", "game_loop"),
}

# Connection pool - single instance for the application
# following the connection pooling best practice from tech stack
_pool: Pool | None = None


async def create_pool(config: dict[str, Any] | None = None) -> Pool:
    """
    Create and return a connection pool.

    Creates a PostgreSQL connection pool with optimized settings for
    our game loop application as per the tech stack document.

    Args:
        config: Database connection parameters. Defaults to environment variables.

    Returns:
        asyncpg connection pool
    """
    global _pool

    if _pool is not None:
        return _pool

    db_config = DEFAULT_DB_CONFIG.copy()
    if config:
        db_config.update(config)

    logger.info(
        f"Creating database connection pool to "
        f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )

    try:
        # Create pool with settings optimized for our application pattern
        # min_size and max_size configured for typical game loop operation
        _pool = await asyncpg.create_pool(
            host=db_config["host"],
            port=db_config["port"],
            user=db_config["user"],
            password=db_config["password"],
            database=db_config["database"],
            min_size=5,  # Minimum connections kept ready
            max_size=20,  # Maximum connections during peak load
            command_timeout=60.0,  # Timeout for long-running queries
        )

        # Test the connection to make sure it works
        async with _pool.acquire() as conn:
            version = await conn.fetchval("SELECT version()")
            logger.info(f"Connected to PostgreSQL: {version}")

            # Check for pgvector extension which is required for embedding storage
            # as described in the embedding_pipeline.md document
            has_vector = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
            )
            if not has_vector:
                logger.warning("pgvector extension is not installed in the database")

        return _pool
    except Exception as e:
        logger.error(f"Failed to create connection pool: {e}")
        raise


@asynccontextmanager
async def get_connection() -> AsyncGenerator[asyncpg.Connection, None]:
    """
    Context manager for getting a connection from the pool.

    Provides a safe, async-compatible way to acquire and release database
    connections. This pattern supports the async I/O patterns mentioned
    in the tech stack document.

    Yields:
        Database connection from the pool

    Example:
        ```python
        async with get_connection() as conn:
            result = await conn.fetch("SELECT * FROM players")
        ```
    """
    if _pool is None:
        await create_pool()

    if _pool is None:
        raise RuntimeError("Could not create database connection pool")

    try:
        async with _pool.acquire() as connection:
            yield connection
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise


async def close_pool() -> None:
    """
    Close the connection pool.

    Safely shuts down the database connection pool when the application
    is terminating. This ensures proper cleanup of database resources.
    """
    global _pool

    if _pool is not None:
        logger.info("Closing database connection pool")
        await _pool.close()
        _pool = None


async def execute_query(query: str, *args: Any) -> Any:
    """
    Execute a query and return the result.

    Executes a SQL query with proper connection management.

    Args:
        query: SQL query string
        args: Query parameters

    Returns:
        Query result
    """
    async with get_connection() as conn:
        return await conn.execute(query, *args)


async def fetch_all(query: str, *args: Any) -> list[Any]:
    """
    Fetch all rows from a query.

    Retrieves all results from a SQL query.
    Useful for querying collections of game entities.

    Args:
        query: SQL query string
        args: Query parameters

    Returns:
        List of query results
    """
    async with get_connection() as conn:
        result = await conn.fetch(query, *args)
        return list(result)


async def fetch_one(query: str, *args: Any) -> Any:
    """
    Fetch a single row from a query.

    Retrieves a single entity from the database.
    Useful for looking up specific game objects by ID.

    Args:
        query: SQL query string
        args: Query parameters

    Returns:
        Single row result or None
    """
    async with get_connection() as conn:
        return await conn.fetchrow(query, *args)


async def fetch_val(query: str, *args: Any) -> Any:
    """
    Fetch a single value from a query.

    Retrieves a single value from the database.
    Useful for COUNT, EXISTS, or single-value lookups.

    Args:
        query: SQL query string
        args: Query parameters

    Returns:
        Single value result or None
    """
    async with get_connection() as conn:
        return await conn.fetchval(query, *args)
