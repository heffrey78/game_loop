"""Manages the state of the game world, including persistence and evolution."""

import json
import logging
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import asyncpg
from pydantic import BaseModel, ValidationError

from .models import (
    ActionResult,
    EvolutionEvent,  # Added EvolutionEvent
    Location,
    NonPlayerCharacter,
    WorldObject,
    WorldState,
)

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
            # This case might be handled by create_new_world or needs
            # clarification
            logger.warning(
                "WorldStateTracker initialized without a " "world_state_id to load."
            )
            self._current_state = None  # Ensure state is clear

    async def create_new_world(
        self,
        initial_locations: dict[UUID, Location] | None = None,
        global_flags: dict[str, Any] | None = None,
    ) -> WorldState:
        """Creates a new world state in memory and database."""
        if not self._current_session_id:
            raise ValueError("Session ID must be set before creating a new world.")

        new_world_id = uuid4()
        new_world_state = WorldState(
            world_id=new_world_id,
            locations=initial_locations or {},
            global_flags=global_flags or {},
            evolution_queue=[],
        )
        self._current_state = new_world_state

        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                try:
                    await conn.execute(
                        """
                        INSERT INTO world_states (
                            world_id, session_id, state_data
                        )
                        VALUES ($1, $2, $3)
                        """,
                        self._current_state.world_id,
                        self._current_session_id,
                        self._current_state.model_dump_json(exclude_none=True),
                    )
                    logger.info(
                        "Created new world state %s for session %s",
                        self._current_state.world_id,
                        self._current_session_id,
                    )
                except Exception as e:
                    logger.error("Error creating new world state: %s", e)
                    self._current_state = None
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
                    self._current_session_id = session_id
                    logger.info(
                        "Loaded world state %s for session %s",
                        world_id,
                        session_id,
                    )
                    return self._current_state
                except (ValidationError, json.JSONDecodeError) as e:
                    logger.error(
                        "Error validating/parsing world state %s: %s",
                        world_id,
                        e,
                    )
                    self._current_state = None
                    return None
            else:
                logger.warning(
                    "No world state found for world %s and session %s",
                    world_id,
                    session_id,
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
                        self._current_state.world_id,
                        self._current_session_id,
                        self._current_state.model_dump_json(exclude_none=True),
                    )
                    logger.info(
                        "Saved world state %s for session %s",
                        self._current_state.world_id,
                        self._current_session_id,
                    )
                except Exception as e:
                    logger.error(
                        "Error saving world state %s: %s",
                        self._current_state.world_id,
                        e,
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
                "Current location %s not found in world state.",
                current_location_id,
            )
            return

        # Update World Objects in the current location
        if action_result.object_changes:
            for change in action_result.object_changes:
                obj_id_any = change.get("object_id")

                if change.get("action") == "remove":
                    location_id_any = change.get("location_id", current_location_id)
                    location_id = (
                        UUID(str(location_id_any))
                        if isinstance(location_id_any, str | UUID)
                        else current_location_id  # Fallback
                    )
                    target_location = self._current_state.locations.get(location_id)
                    if target_location and obj_id_any:
                        obj_id = (
                            UUID(str(obj_id_any))
                            if isinstance(obj_id_any, str | UUID)
                            else None
                        )
                        if obj_id and obj_id in target_location.objects:
                            del target_location.objects[obj_id]
                            logger.debug(
                                "Removed object %s from location %s",
                                obj_id,
                                location_id,
                            )
                            state_changed = True
                        else:
                            logger.warning(
                                "Could not remove object %s from "
                                "location %s: not found or invalid ID",
                                obj_id_any,
                                location_id,
                            )
                    else:
                        logger.warning(
                            "Could not remove object: "
                            "target_location or obj_id_any missing. "
                            "Location: %s, Obj ID: %s",
                            location_id,
                            obj_id_any,
                        )

                elif change.get("action") == "add":
                    obj_data = change.get("object")
                    location_id_any = change.get("location_id", current_location_id)
                    location_id = (
                        UUID(str(location_id_any))
                        if isinstance(location_id_any, str | UUID)
                        else current_location_id
                    )

                    target_location = self._current_state.locations.get(location_id)
                    if target_location and obj_data:
                        try:
                            if isinstance(obj_data, dict):
                                obj = WorldObject(**obj_data)
                                obj_id_to_add = (
                                    obj.object_id
                                    if hasattr(obj, "object_id") and obj.object_id
                                    else uuid4()
                                )
                                target_location.objects[obj_id_to_add] = obj
                                logger.debug(
                                    "Added object %s to location %s",
                                    obj_id_to_add,
                                    location_id,
                                )
                                state_changed = True
                            else:
                                logger.warning(
                                    "Invalid object data for addition: %s",
                                    obj_data,
                                )
                        except Exception as e:
                            logger.error(
                                "Failed to add object to location %s: %s",
                                location_id,
                                e,
                            )

                elif obj_id_any and change.get("updates"):
                    updates = change.get("updates")
                    if updates is None:
                        logger.warning("Update data is None for object %s", obj_id_any)
                        continue
                    try:
                        obj_id = (
                            UUID(str(obj_id_any))
                            if isinstance(obj_id_any, str | UUID)
                            else None
                        )
                        if obj_id is None:
                            logger.warning(
                                "Object ID is None or invalid: %s", obj_id_any
                            )
                            continue
                        world_obj = current_location.objects.get(obj_id)
                    except ValueError as e:
                        logger.warning("Invalid UUID format for object_id: %s", e)
                        continue
                    except KeyError:
                        logger.warning(
                            "Object %s not found in current location %s " "for update.",
                            obj_id_any,  # Use obj_id_any
                            current_location_id,
                        )
                        continue

                    if world_obj is not None and isinstance(updates, dict):
                        for key_path, value in updates.items():
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
                                    "Updated object %s: set %s to %s",
                                    obj_id,
                                    key_path,
                                    value,
                                )
                                state_changed = True
                            except (
                                AttributeError,
                                KeyError,
                                IndexError,
                                TypeError,
                            ) as e:
                                logger.warning(
                                    "Failed to apply update '%s=%s' "
                                    "to object %s: %s",
                                    key_path,
                                    value,
                                    obj_id,
                                    e,
                                )
                    elif world_obj is None:
                        logger.warning(
                            "Object %s not found in current location %s "
                            "for update (re-check).",
                            obj_id_any,
                            current_location_id,
                        )
                else:
                    logger.warning("Unrecognized object change action: %s", change)

        # Update NPCs in the current location
        if action_result.npc_changes:
            for change in action_result.npc_changes:
                npc_id_any = change.get("npc_id")
                updates = change.get("updates")
                if npc_id_any and updates is not None and isinstance(updates, dict):
                    try:
                        npc_id = (
                            UUID(str(npc_id_any))
                            if isinstance(npc_id_any, str | UUID)
                            else None
                        )
                        if npc_id is None:
                            logger.warning("NPC ID is None or invalid: %s", npc_id_any)
                            continue
                        npc = current_location.npcs.get(npc_id)
                    except ValueError as e:
                        logger.warning("Invalid UUID format for npc_id: %s", e)
                        continue
                    except KeyError:
                        logger.warning(
                            "NPC %s not found in current location %s " "for update.",
                            npc_id_any,
                            current_location_id,
                        )
                        continue

                    if npc:
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
                                    "Updated NPC %s: set %s to %s",
                                    npc_id,
                                    key_path,
                                    value,
                                )
                                state_changed = True
                            except (
                                AttributeError,
                                KeyError,
                                IndexError,
                                TypeError,
                            ) as e:
                                logger.warning(
                                    "Failed to apply update '%s=%s' to " "NPC %s: %s",
                                    key_path,
                                    value,
                                    npc_id,
                                    e,
                                )
                    else:
                        logger.warning(
                            "NPC %s not found in current location %s "
                            "for update (re-check).",
                            npc_id_any,
                            current_location_id,
                        )

        # Update Location State Flags
        if action_result.location_state_changes:
            for key, value in action_result.location_state_changes.items():
                if key == "state_flags" and isinstance(value, dict):
                    # Handle nested state_flags updates
                    for flag, flag_value in value.items():
                        current_location.state_flags[flag] = flag_value
                        logger.debug(
                            "Updated location %s flag '%s' to %s",
                            current_location_id,
                            flag,
                            flag_value,
                        )
                        state_changed = True
                else:
                    # Handle direct flag updates
                    current_location.state_flags[key] = value
                    logger.debug(
                        "Updated location %s flag '%s' to %s",
                        current_location_id,
                        key,
                        value,
                    )
                    state_changed = True

        # Update Global Flags
        if action_result.global_flag_changes:
            for flag, value in action_result.global_flag_changes.items():
                self._current_state.global_flags[flag] = value
                logger.debug(
                    "Updated global flag '%s' to %s",
                    flag,
                    value,
                )
                state_changed = True

        # Queue Evolution Event if triggered
        if action_result.triggers_evolution and action_result.evolution_trigger:
            # Assuming evolution_trigger is not None if triggers_evolution
            # is True
            await self.queue_evolution_event(
                trigger=action_result.evolution_trigger,
                data=action_result.evolution_data or {},
            )
            state_changed = True

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
        """Retrieves all objects for a given location ID."""
        if not self._current_state:
            return {}
        location = self._current_state.locations.get(location_id)
        return location.objects if location else {}

    async def get_location_npcs(
        self, location_id: UUID
    ) -> dict[UUID, NonPlayerCharacter]:
        """Retrieves all NPCs for a given location ID."""
        if not self._current_state:
            return {}
        location = self._current_state.locations.get(location_id)
        return location.npcs if location else {}

    async def get_world_id(self) -> UUID | None:
        """Returns the ID of the current world state."""
        return self._current_state.world_id if self._current_state else None

    async def get_all_locations(self) -> dict[UUID, Location]:
        """Returns all locations in the current world state."""
        return self._current_state.locations if self._current_state else {}

    async def get_global_flags(self) -> dict[str, Any]:
        """Returns the global flags of the current world state."""
        return self._current_state.global_flags if self._current_state else {}

    async def queue_evolution_event(
        self,
        trigger: str,
        data: dict[str, Any] | None = None,
        priority: int = 0,
        timestamp: datetime | None = None,
    ) -> None:
        """Adds an event to the world evolution queue."""
        if not self._current_state:
            logger.warning("Cannot queue evolution event: No world state loaded.")
            return

        event = EvolutionEvent(
            trigger=trigger,
            data=data or {},
            priority=priority,
            timestamp=timestamp or datetime.utcnow(),
        )
        self._current_state.evolution_queue.append(event)
        self._current_state.evolution_queue.sort(
            key=lambda e: (-e.priority, e.timestamp)
        )
        logger.debug("Queued evolution event: %s", trigger)

    async def process_next_evolution_event(self) -> bool:
        """Processes the next event in the evolution queue."""
        if not self._current_state or not self._current_state.evolution_queue:
            return False

        event = self._current_state.evolution_queue.pop(0)
        logger.info(
            "Processing evolution event: %s with data %s",
            event.trigger,
            event.data,
        )

        # Placeholder for actual event processing logic
        # This would involve calling specific handlers based on event.trigger

        await self.save_state()  # Save state after processing an event
        return True

    async def get_evolution_queue(self) -> list[EvolutionEvent]:
        """Get the current evolution queue."""
        if not self._current_state:
            return []
        return self._current_state.evolution_queue.copy()

    async def shutdown(self) -> None:
        """Perform any cleanup before the tracker is destroyed."""
        if self._current_state and self._current_session_id:
            logger.info(
                "Shutting down WorldStateTracker for world %s, session %s. "
                "Attempting final save.",
                self._current_state.world_id,
                self._current_session_id,
            )
            try:
                await self.save_state()
            except Exception as e:
                logger.error("Error during final save on shutdown: %s", e)
        else:
            logger.info("Shutting down WorldStateTracker (no active state to save).")
        self._current_state = None
        self._current_session_id = None
