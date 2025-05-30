#!/usr/bin/env python3
"""
Test script to verify that our custom TimestampWithTimezone type generates
correct SQL.
"""

import asyncio
import os

import pytest
from sqlalchemy import Column, Connection, MetaData, Table, text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.asyncio import create_async_engine

from game_loop.database.models.base import TimestampWithTimezone

# Build database URL from environment or defaults
db_user = os.getenv("POSTGRES_USER", "postgres")
db_pass = os.getenv("POSTGRES_PASSWORD", "postgres")
db_host = os.getenv("POSTGRES_HOST", "localhost")
db_port = os.getenv("POSTGRES_PORT", "5432")
db_name = os.getenv("POSTGRES_DB", "game_loop")
db_url = f"postgresql+asyncpg://{db_user}:{db_pass}" f"@{db_host}:{db_port}/{db_name}"


@pytest.mark.asyncio
async def test_timestamp_generation() -> None:
    """Verify that our custom timestamp type generates proper SQL."""
    engine = create_async_engine(db_url)

    # Prepare metadata and table schema
    metadata = MetaData()
    Table(
        "test_timestamps",
        metadata,
        Column("id", PG_UUID(as_uuid=True), primary_key=True),
        Column("created_at", TimestampWithTimezone(), nullable=False),
        Column("updated_at", TimestampWithTimezone(), nullable=False),
    )

    async with engine.begin() as conn:
        # Drop existing table if any
        await conn.execute(text("DROP TABLE IF EXISTS test_timestamps"))

        # Create table
        def create(sync_conn: Connection) -> None:
            metadata.create_all(sync_conn)

        await conn.run_sync(create)

        # Retrieve generated types
        query = (
            "SELECT column_name, data_type "
            "FROM information_schema.columns "
            "WHERE table_name='test_timestamps' "
            "ORDER BY column_name"
        )
        result = await conn.execute(text(query))

        print("=== TIMESTAMP TYPE TEST ===")
        for col, dtype in result.fetchall():
            print(f"Column: {col}, Generated Type: {dtype}")
        print("=" * 20)
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(test_timestamp_generation())
