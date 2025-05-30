"""Test CRUD operations on SQLAlchemy models."""

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from game_loop.database.models import Base, Player


@pytest.mark.asyncio
async def test_models_imported_correctly(db_session: AsyncSession) -> None:
    """Test that all models are properly imported and registered."""
    tables = Base.metadata.tables
    assert len(tables) >= 14  # Check if at least the core tables are there

    expected_tables = {
        "game_sessions",
        "world_rules",
        "evolution_events",
        "players",
        "player_inventories",
        "player_knowledge",
        "player_skills",
        "player_histories",  # Corrected from player_history based on model
        "regions",
        "locations",
        "objects",
        "npcs",
        "quests",
        "location_connections",
    }

    actual_tables = set(tables.keys())
    assert expected_tables.issubset(actual_tables)


@pytest.mark.asyncio
async def test_create_tables_in_database(db_session: AsyncSession) -> None:
    """Test that we can create the tables in the database."""
    bind = db_session.bind
    if not isinstance(bind, AsyncEngine):
        pytest.fail(
            "db_session.get_bind() did not return an AsyncEngine instance. "
            f"Got: {type(bind)}"
        )

    async with bind.begin() as conn:  # engine.begin() returns an AsyncTransaction
        await conn.run_sync(Base.metadata.create_all)

    # Verify a few key tables exist
    query = text(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' "
        "AND table_name IN ('players', 'game_sessions', 'regions')"
    )
    result = await db_session.execute(query)

    tables_in_db = [row[0] for row in result.fetchall()]
    assert "players" in tables_in_db
    assert "game_sessions" in tables_in_db
    assert "regions" in tables_in_db


@pytest.mark.asyncio
async def test_basic_model_crud(db_session: AsyncSession) -> None:
    """Test basic CRUD operations on models."""
    bind = db_session.bind
    if not isinstance(bind, AsyncEngine):
        pytest.fail(
            "db_session.get_bind() did not return an AsyncEngine instance. "
            f"Got: {type(bind)}"
        )

    async with bind.begin() as conn:  # engine.begin() returns an AsyncTransaction
        await conn.run_sync(Base.metadata.create_all)

    player = Player(name="Test Player", username="crud_test_user", level=1)

    db_session.add(player)
    await db_session.commit()

    # Test reading the player back
    result = await db_session.execute(
        text("SELECT name FROM players WHERE name = :name"), {"name": "Test Player"}
    )
    fetched_name = result.scalar()
    assert fetched_name == "Test Player"

    # Test updating the player
    await db_session.execute(
        text("UPDATE players SET level = 2 WHERE name = :name"),
        {"name": "Test Player"},
    )
    await db_session.commit()

    # Verify update
    result = await db_session.execute(
        text("SELECT level FROM players WHERE name = :name"), {"name": "Test Player"}
    )
    level = result.scalar()
    assert level == 2

    # Clean up
    await db_session.execute(
        text("DELETE FROM players WHERE name = :name"), {"name": "Test Player"}
    )
    await db_session.commit()
