"""
Basic tests for SQLAlchemy models and database creation.
"""

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from game_loop.database.models.base import Base


@pytest.mark.asyncio
async def test_create_tables(db_session: AsyncSession) -> None:
    """Test that we can create all tables from our SQLAlchemy models."""
    # Create all tables
    async with db_session.bind.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Verify tables exist by querying the information schema
    result = await db_session.execute(
        text(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """
        )
    )

    tables = [row[0] for row in result.fetchall()]

    # Check that we have our main tables
    expected_tables = [
        "players",
        "player_inventories",
        "player_knowledge",
        "player_skills",
        "player_histories",
        "regions",
        "locations",
        "objects",
        "npcs",
        "quests",
        "location_connections",
        "game_sessions",
        "world_rules",
        "evolution_events",
    ]

    for table in expected_tables:
        assert table in tables, f"Expected table '{table}' not found in database"

    print(f"Successfully created {len(tables)} tables: {tables}")


@pytest.mark.asyncio
async def test_vector_extension_integration(db_session: AsyncSession) -> None:
    """Test that pgvector extension works with our models."""
    # Create all tables first
    async with db_session.bind.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Test vector operations
    await db_session.execute(text("SELECT ARRAY[1,2,3]::vector(3)"))

    # Test that we can create vectors with our embedding dimensions (384)
    await db_session.execute(
        text("SELECT ARRAY[" + ",".join(["0.1"] * 384) + "]::vector(384)")
    )

    print("Vector extension integration successful")


@pytest.mark.asyncio
async def test_model_constraints(db_session: AsyncSession) -> None:
    """Test that model constraints are properly applied."""
    # Drop and recreate schema to handle foreign key constraints
    await db_session.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
    await db_session.execute(text("CREATE SCHEMA public"))
    await db_session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    await db_session.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
    await db_session.commit()

    # Import and use SQLAlchemy models
    from src.game_loop.database.models.base import Base

    # Create all tables using a separate engine connection
    engine = db_session.bind
    async with engine.begin() as conn:

        def sync_create_all(sync_conn):
            Base.metadata.create_all(sync_conn)

        await conn.run_sync(sync_create_all)

    # Test UUID and timestamp constraints by checking column definitions
    result = await db_session.execute(
        text(
            """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = 'players'
            ORDER BY column_name
        """
        )
    )

    columns = {
        row[0]: {"type": row[1], "nullable": row[2] == "YES", "default": row[3]}
        for row in result.fetchall()
    }

    # Debug: Print all columns
    print("\n=== ACTUAL COLUMNS IN PLAYERS TABLE ===")
    for col_name, col_info in columns.items():
        nullable = col_info["nullable"]
        default = col_info["default"]
        print(
            f"Column: {col_name}, Type: {col_info['type']}, "
            f"Nullable: {nullable}, Default: {default}"
        )
    print("=" * 37)

    # Check that we have the expected columns with proper types
    assert "id" in columns
    assert columns["id"]["type"] == "uuid"
    assert not columns["id"]["nullable"]

    assert "created_at" in columns
    expected_type = "timestamp with time zone"
    assert columns["created_at"]["type"] == expected_type
    assert not columns["created_at"]["nullable"]

    assert "updated_at" in columns
    assert columns["updated_at"]["type"] == expected_type
    assert not columns["updated_at"]["nullable"]

    print("Model constraints properly applied")
