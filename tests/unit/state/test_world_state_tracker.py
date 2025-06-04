import json
from collections.abc import AsyncGenerator
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from asyncpg import Pool
from asyncpg.transaction import Transaction

from game_loop.state.models import (
    ActionResult,
    EvolutionEvent,
    Location,
    NonPlayerCharacter,
    WorldObject,
    WorldState,
)
from game_loop.state.world_state import WorldStateTracker


@pytest_asyncio.fixture
async def mock_db_pool() -> AsyncGenerator[tuple[Pool, MagicMock], None]:
    """Fixture for a mock database pool and connection."""
    mock_pool = MagicMock(spec=Pool)
    mock_conn = MagicMock()  # Simplified mock connection for most tests

    async def mock_acquire_aenter(*args, **kwargs) -> MagicMock:
        return mock_conn

    async def mock_acquire_aexit(*args, **kwargs) -> None:
        pass

    mock_pool.acquire.return_value.__aenter__ = mock_acquire_aenter
    mock_pool.acquire.return_value.__aexit__ = mock_acquire_aexit

    # For transaction mocking
    mock_transaction = MagicMock(spec=Transaction)

    async def mock_transaction_aenter(*args, **kwargs) -> MagicMock:
        return mock_transaction

    async def mock_transaction_aexit(*args, **kwargs) -> None:
        pass

    mock_conn.transaction.return_value.__aenter__ = mock_transaction_aenter
    mock_conn.transaction.return_value.__aexit__ = mock_transaction_aexit

    mock_conn.fetchval = AsyncMock()
    mock_conn.fetchrow = AsyncMock()
    mock_conn.execute = AsyncMock()

    yield mock_pool, mock_conn


@pytest_asyncio.fixture
async def world_tracker(
    mock_db_pool: tuple[Pool, MagicMock],
) -> AsyncGenerator[WorldStateTracker, None]:
    """Fixture for WorldStateTracker with a mocked db_pool."""
    pool, _ = mock_db_pool
    tracker = WorldStateTracker(db_pool=pool)
    return tracker


@pytest_asyncio.fixture
async def initialized_tracker(
    world_tracker: WorldStateTracker, mock_db_pool: tuple[Pool, MagicMock]
) -> AsyncGenerator[WorldStateTracker, None]:
    """Fixture for an initialized WorldStateTracker."""
    _, mock_conn = mock_db_pool
    world_id = uuid4()
    initial_state = WorldState(
        world_id=world_id,
        global_flags={"initial": True},
        locations={
            uuid4(): Location(name="Test Location 1", description="Desc 1"),
            uuid4(): Location(name="Test Location 2", description="Desc 2"),
        },
    )
    world_tracker._current_state = initial_state
    world_tracker._current_session_id = uuid4()  # Ensure session_id is set

    # Mock save_state for tests that might trigger it indirectly
    # Use patch.object for instance methods
    with patch.object(world_tracker, "save_state", new_callable=AsyncMock) as _:
        yield world_tracker


@pytest.mark.asyncio
async def test_world_state_tracker_initialization(
    world_tracker: WorldStateTracker,
) -> None:
    """Test that the WorldStateTracker initializes with no current state."""
    assert world_tracker._current_state is None
    assert world_tracker._current_session_id is None
    assert world_tracker.db_pool is not None


@pytest.mark.asyncio
async def test_create_new_world(
    world_tracker: WorldStateTracker, mock_db_pool: tuple[Pool, MagicMock]
) -> None:
    """Test creating a new world."""
    pool, mock_conn = mock_db_pool
    session_id = uuid4()

    # Set session ID first
    world_tracker._current_session_id = session_id

    mock_conn.fetchval.side_effect = [uuid4(), session_id]

    # Patch save_state for this specific test
    with patch.object(world_tracker, "save_state", new_callable=AsyncMock):
        new_world = await world_tracker.create_new_world(
            global_flags={"initial_flag": False}
        )

        assert new_world is not None
        assert world_tracker._current_state is not None
        # Don't assert specific world_id since implementation generates its own
        assert isinstance(world_tracker._current_state.world_id, UUID)
        assert world_tracker._current_state.global_flags["initial_flag"] is False


@pytest.mark.asyncio
async def test_create_new_world_no_session_id(world_tracker: WorldStateTracker) -> None:
    """Test creating a new world when no session_id is set (should raise)."""
    # Don't set session_id, it should be None by default
    with pytest.raises(
        ValueError, match="Session ID must be set before creating a new world."
    ):
        await world_tracker.create_new_world()


@pytest.mark.asyncio
async def test_load_state_success(
    world_tracker: WorldStateTracker, mock_db_pool: tuple[Pool, MagicMock]
) -> None:
    """Test loading world state successfully."""
    pool, mock_conn = mock_db_pool
    world_id = uuid4()
    session_id = uuid4()
    expected_world_state_dict = {
        "world_id": str(world_id),
        "global_flags": {"loaded": True},
        "locations": {},
        "evolution_queue": [],
    }
    # Mock fetchrow to return a row with state_data field
    mock_conn.fetchrow.return_value = {
        "state_data": json.dumps(expected_world_state_dict)
    }

    await world_tracker.load_state(world_id, session_id)

    assert world_tracker._current_state is not None
    assert world_tracker._current_state.world_id == world_id
    assert world_tracker._current_state.global_flags == {"loaded": True}
    assert world_tracker._current_session_id == session_id
    mock_conn.fetchrow.assert_called_once_with(
        """
                SELECT state_data FROM world_states
                WHERE world_id = $1 AND session_id = $2
                ORDER BY updated_at DESC LIMIT 1
                """,
        world_id,
        session_id,
    )


@pytest.mark.asyncio
async def test_save_state_success(
    world_tracker: WorldStateTracker, mock_db_pool: tuple[Pool, MagicMock]
) -> None:
    """Test saving world state successfully."""
    pool, mock_conn = mock_db_pool
    world_id = uuid4()
    session_id = uuid4()
    current_world_state = WorldState(world_id=world_id, global_flags={"saved": True})
    world_tracker._current_state = current_world_state
    world_tracker._current_session_id = session_id

    await world_tracker.save_state()

    mock_conn.execute.assert_called_once_with(
        """
                        INSERT INTO world_states (
                            world_id,
                            session_id,
                            state_data,
                            updated_at
                            )
                        VALUES ($1, $2, $3, NOW())
                        ON CONFLICT (world_id, session_id) DO UPDATE
                        SET state_data = EXCLUDED.state_data,
                            updated_at = NOW();
                        """,
        world_id,
        session_id,
        current_world_state.model_dump_json(exclude_none=True),
    )


@pytest.mark.asyncio
async def test_save_state_no_current_state(world_tracker: WorldStateTracker) -> None:
    """Test save_state when no current state exists (should not error, no op)."""
    world_tracker._current_session_id = uuid4()
    # Patch to ensure execute is not called
    with patch.object(
        world_tracker.db_pool, "acquire", new_callable=AsyncMock
    ) as mock_acquire:
        await world_tracker.save_state()
        mock_acquire.assert_not_called()


@pytest.mark.asyncio
async def test_save_state_no_session_id(world_tracker: WorldStateTracker) -> None:
    """Test save_state when no session_id exists (should not error, no op)."""
    world_tracker._current_state = WorldState(world_id=uuid4())
    with patch.object(
        world_tracker.db_pool, "acquire", new_callable=AsyncMock
    ) as mock_acquire:
        await world_tracker.save_state()
        mock_acquire.assert_not_called()


@pytest.mark.asyncio
async def test_shutdown_does_not_error(
    world_tracker: WorldStateTracker, mock_db_pool: tuple[Pool, MagicMock]
) -> None:
    """Test that shutdown calls save_state and completes without error."""
    world_id = uuid4()
    session_id = uuid4()
    world_tracker._current_state = WorldState(world_id=world_id)
    world_tracker._current_session_id = session_id

    # Mock save_state for this test
    with patch.object(world_tracker, "save_state", new_callable=AsyncMock) as mock_save:
        await world_tracker.shutdown()
        mock_save.assert_called_once()


@pytest.mark.asyncio
async def test_update_from_action_no_current_state(
    world_tracker: WorldStateTracker,
) -> None:
    """Test update_from_action when no current_state, should log warning and return."""
    action_result = ActionResult(
        object_changes=[{"object_id": uuid4(), "updates": {"name": "new name"}}]
    )
    # Should not raise exception, just log warning and return
    await world_tracker.update_from_action(action_result, current_location_id=uuid4())


@pytest.mark.asyncio
async def test_update_object_state_in_action(
    initialized_tracker: WorldStateTracker,
) -> None:
    """Test updating an object's state via ActionResult."""
    tracker = initialized_tracker
    assert tracker._current_state is not None
    loc_id = list(tracker._current_state.locations.keys())[0]
    obj_id = uuid4()
    tracker._current_state.locations[loc_id].objects[obj_id] = WorldObject(
        name="Old Chest", description="An old wooden chest."
    )

    action_result = ActionResult(
        object_changes=[{"object_id": str(obj_id), "updates": {"name": "Shiny Chest"}}]
    )
    await tracker.update_from_action(action_result, current_location_id=loc_id)

    updated_object = tracker._current_state.locations[loc_id].objects[obj_id]
    assert updated_object.name == "Shiny Chest"
    # save_state should have been called by update_from_action
    # The mock is on initialized_tracker instance
    cast(AsyncMock, tracker.save_state).assert_called_once()


@pytest.mark.asyncio
async def test_update_npc_state_in_action(
    initialized_tracker: WorldStateTracker,
) -> None:
    """Test updating an NPC's state via ActionResult."""
    tracker = initialized_tracker
    assert tracker._current_state is not None
    loc_id = list(tracker._current_state.locations.keys())[0]
    npc_id = uuid4()
    initial_npc = NonPlayerCharacter(name="Guard", description="A watchful guard.")
    tracker._current_state.locations[loc_id].npcs[npc_id] = initial_npc

    action_result = ActionResult(
        npc_changes=[
            {
                "npc_id": str(npc_id),
                "updates": {
                    "dialogue_state": "hostile",
                    "current_behavior": "attacking",
                },
            }
        ]
    )
    await tracker.update_from_action(action_result, current_location_id=loc_id)

    updated_npc = tracker._current_state.locations[loc_id].npcs[npc_id]
    assert updated_npc.dialogue_state == "hostile"
    assert updated_npc.current_behavior == "attacking"
    cast(AsyncMock, tracker.save_state).assert_called_once()


@pytest.mark.asyncio
async def test_update_location_state_flags_in_action(
    initialized_tracker: WorldStateTracker,
) -> None:
    """Test updating a location's state_flags via ActionResult."""
    tracker = initialized_tracker
    assert tracker._current_state is not None
    loc_id = list(tracker._current_state.locations.keys())[0]
    tracker._current_state.locations[loc_id].state_flags = {"weather": "cloudy"}

    action_result = ActionResult(
        location_state_changes={"state_flags": {"weather": "sunny"}}
    )
    await tracker.update_from_action(action_result, current_location_id=loc_id)

    updated_location = tracker._current_state.locations[loc_id]
    assert updated_location.state_flags["weather"] == "sunny"
    cast(AsyncMock, tracker.save_state).assert_called_once()


@pytest.mark.asyncio
async def test_update_from_action_full(
    initialized_tracker: WorldStateTracker, mock_db_pool: tuple[Pool, MagicMock]
) -> None:
    """Test a full update_from_action with multiple changes."""
    tracker = initialized_tracker
    _pool, _mock_conn = mock_db_pool

    assert tracker._current_state is not None
    loc_id = list(tracker._current_state.locations.keys())[0]
    obj_id = uuid4()
    npc_id = uuid4()

    tracker._current_state.locations[loc_id].objects[obj_id] = WorldObject(
        name="Old Chest", description="A dusty old chest."
    )
    tracker._current_state.locations[loc_id].npcs[npc_id] = NonPlayerCharacter(
        name="Old Guard", description="A sleepy old guard."
    )
    tracker._current_state.locations[loc_id].state_flags = {"time": "night"}
    # Don't preset global_flags to avoid conflicts

    action_result = ActionResult(
        object_changes=[{"object_id": str(obj_id), "updates": {"name": "Shiny Chest"}}],
        npc_changes=[{"npc_id": str(npc_id), "updates": {"name": "Friendly Guard"}}],
        location_state_changes={"state_flags": {"weather": "sunny"}},
        global_flag_changes={"event_started": True},
        triggers_evolution=True,
        evolution_trigger="test_event",
        evolution_data={"detail": "triggered"},
    )

    await tracker.update_from_action(action_result, current_location_id=loc_id)

    assert (
        tracker._current_state.locations[loc_id].objects[obj_id].name == "Shiny Chest"
    )
    assert (
        tracker._current_state.locations[loc_id].npcs[npc_id].name == "Friendly Guard"
    )
    assert tracker._current_state.locations[loc_id].state_flags["weather"] == "sunny"
    # Check that the global flag was actually set
    assert tracker._current_state.global_flags.get("event_started") is True


# --- Getter Tests ---
@pytest.mark.asyncio
async def test_get_world_id(initialized_tracker: WorldStateTracker) -> None:
    """Test getting the world ID."""
    tracker = initialized_tracker
    assert tracker._current_state is not None
    world_id = await tracker.get_world_id()
    assert world_id == tracker._current_state.world_id


@pytest.mark.asyncio
async def test_get_location_details(initialized_tracker: WorldStateTracker) -> None:
    """Test getting details for a specific location."""
    tracker = initialized_tracker
    assert tracker._current_state is not None
    loc_id = list(tracker._current_state.locations.keys())[0]
    # Changed to get_location_details
    location = await tracker.get_location_details(loc_id)
    assert location is not None
    assert location.name == tracker._current_state.locations[loc_id].name


@pytest.mark.asyncio
async def test_get_all_locations(initialized_tracker: WorldStateTracker) -> None:
    """Test getting all locations."""
    tracker = initialized_tracker
    assert tracker._current_state is not None
    # This is now an async method
    locations = await tracker.get_all_locations()
    # Based on initialized_tracker fixture
    assert len(locations) == 2
    assert isinstance(locations, dict)


@pytest.mark.asyncio
async def test_get_global_flags(initialized_tracker: WorldStateTracker) -> None:
    """Test getting global flags."""
    tracker = initialized_tracker
    assert tracker._current_state is not None
    # This is now an async method
    flags = await tracker.get_global_flags()
    assert flags == {"initial": True}


@pytest.mark.asyncio
async def test_get_evolution_queue(initialized_tracker: WorldStateTracker) -> None:
    """Test getting the evolution queue."""
    tracker = initialized_tracker
    assert tracker._current_state is not None
    tracker._current_state.evolution_queue.append(
        EvolutionEvent(trigger="test", data={})
    )
    queue = await tracker.get_evolution_queue()
    assert len(queue) == 1
