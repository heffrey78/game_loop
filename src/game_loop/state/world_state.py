"""Manages the state of the game world, including persistence and evolution."""

import json
import logging
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import asyncpg
from pydantic import BaseModel, ValidationError

from .models import ActionResult, Location, NonPlayerCharacter, WorldObject, WorldState

logger = logging.getLogger(__name__)


class WorldStateTracker:
    """Tracks and persists the world's state."""

    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self._current_state: WorldState | None = None
        self._current_session_id: UUID | None = None

    async def initialize(
        self, session_id: UUID, world_state_id: UUID | None = None
    ) -> None:
        """Initialize the tracker for a given session,
        loading existing state if ID provided."""
        self._current_session_id = session_id
        if world_state_id:
            await self.load_state(world_state_id, session_id)
        else:
            # This case might be handled by create_new_world or needs clarification
            logger.warning(
                "WorldStateTracker initialized without a world_state_id to load."
            )
            self._current_state = None  # Ensure state is clear

    async def create_new_world(
        self, initial_locations: dict[UUID, Location] | None = None
    ) -> WorldState:
        """Creates a new world state in memory and database."""
        if not self._current_session_id:
            raise ValueError("Session ID must be set before creating a new world.")

        # Create a new world state with a generated UUID
        new_world_id = uuid4()
        new_world_state = WorldState(
            world_id=new_world_id,  # Explicitly set the world_id
            locations=initial_locations or {},
            evolution_queue=[],  # Initialize with empty evolution queue
            # Initialize other fields as needed
        )
        self._current_state = new_world_state

        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                try:
                    # Check if the table has the expected structure
                    table_exists = await conn.fetchval(
                        """
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_name = 'world_states'
                        )
                        """
                    )

                    if not table_exists:
                        logger.error("world_states table does not exist")
                        raise RuntimeError("world_states table does not exist")

                    # Insert the new world state
                    await conn.execute(
                        """
                        INSERT INTO world_states (world_id, session_id, state_data)
                        VALUES ($1, $2, $3)
                        """,
                        self._current_state.world_id,
                        self._current_session_id,
                        self._current_state.model_dump_json(exclude_none=True),
                    )
                    logger.info(
                        f"Created new world state {self._current_state.world_id} "
                        f"for session {self._current_session_id}"
                    )
                except Exception as e:
                    logger.error(f"Error creating new world state: {e}")
                    self._current_state = None  # Rollback in-memory state
                    raise
        return self._current_state

    async def load_state(self, world_id: UUID, session_id: UUID) -> WorldState | None:
        """Loads world state from the database."""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT state_data FROM world_states
                WHERE world_id = $1 AND session_id = $2
                ORDER BY updated_at DESC LIMIT 1
                """,
                world_id,
                session_id,
            )
            if row and row["state_data"]:
                try:
                    self._current_state = WorldState.model_validate_json(
                        row["state_data"]
                    )
                    self._current_session_id = (
                        session_id  # Ensure session ID is updated
                    )
                    logger.info(
                        f"Loaded world state {world_id} for session {session_id}"
                    )
                    return self._current_state
                except (ValidationError, json.JSONDecodeError) as e:
                    logger.error(
                        f"Error validating/parsing world state {world_id}: {e}"
                    )
                    self._current_state = None
                    return None
            else:
                logger.warning(
                    f"No world state found for world {world_id} "
                    f"and session {session_id}"
                )
                self._current_state = None
                return None

    async def save_state(self) -> None:
        """Saves the current world state to the database."""
        if not self._current_state or not self._current_session_id:
            logger.warning("No current state or session ID is loaded.")
            return

        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                try:
                    await conn.execute(
                        """
                        INSERT INTO world_states (world_id,
                        session_id,
                        state_data,
                        updated_at
                        )
                        VALUES ($1, $2, $3, NOW())
                        ON CONFLICT (world_id, session_id) DO UPDATE
                        SET state_data = EXCLUDED.state_data,
                            updated_at = NOW();
                        """,
                        self._current_state.world_id,
                        self._current_session_id,
                        self._current_state.model_dump_json(exclude_none=True),
                    )
                    logger.info(
                        f"Saved world state {self._current_state.world_id} for "
                        f"session {self._current_session_id}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error saving world state {self._current_state.world_id}: {e}"
                    )
                    raise

    def get_state(self) -> WorldState | None:
        """Returns the current in-memory world state."""
        return self._current_state

    async def update_from_action(
        self, action_result: ActionResult, current_location_id: UUID | None
    ) -> None:
        """Updates the world state based on the results of an action."""
        if not self._current_state or not current_location_id:
            logger.warning(
                "Cannot update world state from action: "
                "No world state or current location ID."
            )
            return

        state_changed = False
        current_location = self._current_state.locations.get(current_location_id)
        if not current_location:
            logger.warning(
                f"Current location {current_location_id} not found in world state."
            )
            return

        # Update World Objects in the current location
        if action_result.object_changes:
            for change in action_result.object_changes:
                # Handle object removal
                if change.get("action") == "remove":
                    obj_id = change.get("object_id")
                    location_id = change.get("location_id", current_location_id)
                    # Make sure we have the right location
                    target_location = self._current_state.locations.get(location_id)
                    if target_location and obj_id in target_location.objects:
                        del target_location.objects[obj_id]
                        logger.debug(
                            f"Removed object {obj_id} from location {location_id}"
                        )
                        state_changed = True
                    else:
                        logger.warning(
                            f"Could not remove object {obj_id} from location "
                            f"{location_id}: not found"
                        )

                # Handle object addition
                elif change.get("action") == "add":
                    obj_data = change.get("object")
                    location_id = change.get("location_id", current_location_id)
                    # Make sure we have the right location
                    target_location = self._current_state.locations.get(location_id)
                    if target_location and obj_data:
                        # Create WorldObject from data
                        from game_loop.state.models import WorldObject

                        try:
                            if isinstance(obj_data, dict):
                                obj = WorldObject(**obj_data)
                                # Use the object_id from the data if available,
                                # otherwise generate one
                                obj_id = (
                                    obj.object_id
                                    if hasattr(obj, "object_id") and obj.object_id
                                    else uuid4()
                                )
                                target_location.objects[obj_id] = obj
                                logger.debug(
                                    f"Added object {obj_id} to location {location_id}"
                                )
                                state_changed = True
                            else:
                                logger.warning(
                                    f"Invalid object data for addition: {obj_data}"
                                )
                        except Exception as e:
                            logger.error(
                                f"Failed to add object to location {location_id}: {e}"
                            )

                # Handle object updates (existing code)
                elif change.get("object_id") and change.get("update"):
                    obj_id = change.get("object_id")
                    updates = change.get("update")  # e.g., {"state.locked": False}
                    if updates is None:
                        logger.warning(f"Update data is None for object {obj_id}")
                        continue
                    # Convert obj_id to UUID or skip if invalid
                    try:
                        if isinstance(obj_id, str):
                            obj_id = UUID(obj_id)
                        if obj_id is None:
                            logger.warning("Object ID is None")
                            continue
                        world_obj = current_location.objects.get(obj_id)
                    except ValueError as e:
                        logger.warning(f"Invalid UUID format for object_id: {e}")
                        continue

                    if world_obj is not None and isinstance(updates, dict):
                        for key_path, value in updates.items():
                            # Basic nested update handling (e.g., "state.locked")
                            parts = key_path.split(".")
                            target = world_obj
                            try:
                                for i, part in enumerate(parts):
                                    if i == len(parts) - 1:
                                        if isinstance(target, BaseModel):
                                            setattr(target, part, value)
                                        elif isinstance(target, dict):
                                            target[part] = value
                                        else:
                                            raise TypeError(
                                                "Cannot set attribute on "
                                                f"{type(target)}"
                                            )
                                    else:
                                        if isinstance(target, BaseModel):
                                            target = getattr(target, part)
                                        elif isinstance(target, dict):
                                            target = target[part]
                                        else:
                                            raise TypeError(
                                                "Cannot get attribute from "
                                                f"{type(target)}"
                                            )
                                logger.debug(
                                    f"Updated object {obj_id}: set {key_path} "
                                    f"to {value}"
                                )
                                state_changed = True
                            except (
                                AttributeError,
                                KeyError,
                                IndexError,
                                TypeError,
                            ) as e:
                                logger.warning(
                                    f"Failed to apply update '{key_path}={value}' "
                                    f"to object {obj_id}: {e}"
                                )
                    else:
                        logger.warning(
                            f"Object {obj_id} not found in current location "
                            f"{current_location_id} for update."
                        )

                # If none of the expected actions match
                else:
                    logger.warning(f"Unrecognized object change action: {change}")

        # Update NPCs in the current location
        if action_result.npc_changes:
            for change in action_result.npc_changes:
                npc_id = change.get("npc_id")
                updates = change.get("update")  # e.g., {"state.hostile": True}
                if npc_id and updates is not None and isinstance(updates, dict):
                    npc = current_location.npcs.get(npc_id)
                    if npc:
                        # Similar update logic as for objects
                        for key_path, value in updates.items():
                            parts = key_path.split(".")
                            target = npc
                            try:
                                for i, part in enumerate(parts):
                                    if i == len(parts) - 1:
                                        if isinstance(target, BaseModel):
                                            setattr(target, part, value)
                                        elif isinstance(target, dict):
                                            target[part] = value
                                        else:
                                            raise TypeError(
                                                "Cannot set attribute on "
                                                f"{type(target)}"
                                            )
                                    else:
                                        if isinstance(target, BaseModel):
                                            target = getattr(target, part)
                                        elif isinstance(target, dict):
                                            target = target[part]
                                        else:
                                            raise TypeError(
                                                "Cannot get attribute from "
                                                f"{type(target)}"
                                            )
                                logger.debug(
                                    f"Updated NPC {npc_id}: set {key_path} to {value}"
                                )
                                state_changed = True
                            except (
                                AttributeError,
                                KeyError,
                                IndexError,
                                TypeError,
                            ) as e:
                                logger.warning(
                                    f"Failed to apply update '{key_path}={value}' to "
                                    f"NPC {npc_id}: {e}"
                                )
                    else:
                        logger.warning(
                            f"NPC {npc_id} not found in current location "
                            f"{current_location_id} for update."
                        )

        # Update Location State Flags
        if action_result.location_state_changes:
            for flag, value in action_result.location_state_changes.items():
                current_location.state_flags[flag] = value
                logger.debug(
                    f"Updated location {current_location_id} flag '{flag}' to {value}"
                )
                state_changed = True

        # Queue Evolution Event if triggered
        if action_result.triggers_evolution and action_result.evolution_trigger:
            await self.queue_evolution_event(
                trigger=action_result.evolution_trigger,
                data=action_result.evolution_data or {},
                priority=action_result.priority,
                timestamp=action_result.timestamp,
            )
            state_changed = True  # Queuing is a state change

        # Persist changes if any occurred (similar consideration as PlayerStateTracker)
        if state_changed:
            await self.save_state()

    async def get_location_description(self, location_id: UUID) -> str | None:
        """Retrieves the description for a given location ID."""
        if not self._current_state:
            return None
        location = self._current_state.locations.get(location_id)
        return location.description if location else None

    async def get_location_details(self, location_id: UUID) -> Location | None:
        """Retrieves the full Location model for a given ID."""
        if not self._current_state:
            return None
        return self._current_state.locations.get(location_id)

    async def get_location_objects(self, location_id: UUID) -> dict[UUID, WorldObject]:
        """Retrieves the objects present in a given location."""
        location = await self.get_location_details(location_id)
        return location.objects if location else {}

    async def get_location_npcs(
        self, location_id: UUID
    ) -> dict[UUID, NonPlayerCharacter]:
        """Retrieves the NPCs present in a given location."""
        location = await self.get_location_details(location_id)
        return location.npcs if location else {}

    async def queue_evolution_event(
        self,
        trigger: str,
        data: dict[str, Any],
        priority: int = 5,
        timestamp: datetime | None = None,
    ) -> None:
        """Adds an event to the world evolution queue."""
        if not self._current_state:
            return
        event = {
            "trigger": trigger,
            "data": data,
            "priority": priority,
            "timestamp": timestamp or datetime.now(),
        }
        # Simple append for now, could implement priority queue later
        self._current_state.evolution_queue.append(event)
        logger.debug(f"Queued evolution event: {trigger}")
        # Consider saving state here or marking as dirty

    async def process_evolution_queue(self) -> list[dict[str, Any]]:
        """Processes events in the evolution queue (basic implementation)."""
        if not self._current_state:
            return []

        processed_events = []
        # Simple FIFO processing for now
        # A real implementation would likely involve more complex logic,
        # potentially calling other services or modifying world state directly.
        while self._current_state.evolution_queue:
            event = self._current_state.evolution_queue.pop(0)
            logger.info(
                f"Processing evolution event: {event['trigger']} with "
                f"data {event['data']}"
            )
            # TODO: Implement actual event processing logic here
            # This might involve changing NPC states, object states,
            # location flags, etc.
            # For now, just log and collect the processed events.
            processed_events.append(event)
            # Mark state as changed if processing modifies anything
            # await self.save_state() # Or defer saving

        return processed_events

    async def get_world_id(self) -> UUID | None:
        """Returns the ID of the currently loaded world."""
        return self._current_state.world_id if self._current_state else None

    async def shutdown(self) -> None:
        """Perform any cleanup, like ensuring the final state is saved."""
        logger.info("Shutting down WorldStateTracker...")
        # Decide if auto-save on shutdown is desired
        # await self.save_state()
        logger.info("WorldStateTracker shut down.")
