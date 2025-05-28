"""Manages the state of the player character, including persistence."""

import json
import logging
from typing import Any
from uuid import UUID

import asyncpg
from pydantic import ValidationError

from .models import (
    ActionResult,
    InventoryItem,
    PlayerKnowledge,
    PlayerState,
    PlayerStats,
)

logger = logging.getLogger(__name__)


class PlayerStateTracker:
    """Tracks and persists the player's state."""

    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self._current_state: PlayerState | None = None
        self._current_session_id: UUID | None = None

    async def initialize(
        self, session_id: UUID, player_state_id: UUID | None = None
    ) -> None:
        """
        Initialize the tracker for a session,
        with existing state if ID provided.
        """
        self._current_session_id = session_id
        if player_state_id:
            await self.load_state(player_state_id, session_id)
        else:
            logger.warning(
                "PlayerStateTracker initialized without a player_state_id to " "load."
            )
            self._current_state = None

    async def create_new_player(
        self,
        player_name: str = "Player",
        starting_location_id: UUID | None = None,
    ) -> PlayerState:  # Ensure return type is met or exception raised
        """Creates a new player state in memory and database."""
        if not self._current_session_id:
            raise ValueError("Session ID must be set before creating a new player.")

        new_player_state = PlayerState(
            name=player_name,
            current_location_id=starting_location_id,
        )
        self._current_state = new_player_state

        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                try:
                    await conn.execute(
                        """
                        INSERT INTO player_states (
                            player_id, session_id, state_data
                        )
                        VALUES ($1, $2, $3)
                        """,
                        self._current_state.player_id,
                        self._current_session_id,
                        self._current_state.model_dump_json(exclude_none=True),
                    )
                    logger.info(
                        f"Created new player state "
                        f"{self._current_state.player_id} for session "
                        f"{self._current_session_id}"
                    )
                    return self._current_state
                except Exception as e:
                    logger.error(f"Error creating new player state: {e}")
                    self._current_state = None  # Rollback in-memory state
                    raise

    async def load_state(self, player_id: UUID, session_id: UUID) -> PlayerState | None:
        """Loads player state from the database."""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT state_data FROM player_states
                WHERE player_id = $1 AND session_id = $2
                ORDER BY updated_at DESC LIMIT 1
                """,
                player_id,
                session_id,
            )
            if row and row["state_data"]:
                try:
                    # Pydantic v2 uses model_validate_json
                    self._current_state = PlayerState.model_validate_json(
                        row["state_data"]
                    )
                    self._current_session_id = (
                        session_id  # Ensure session ID is updated on load
                    )
                    logger.info(
                        f"Loaded player state {player_id} for session " f"{session_id}"
                    )
                    return self._current_state
                except (ValidationError, json.JSONDecodeError) as e:
                    logger.error(
                        f"Error validating/parsing player state {player_id}: " f"{e}"
                    )
                    self._current_state = None
                    return None
            else:
                logger.warning(
                    f"No player state found for player {player_id} "
                    f"and session {session_id}"
                )
                self._current_state = None
                return None

    async def save_state(self) -> None:
        """Saves the current player state to the database."""
        if not self._current_state or not self._current_session_id:
            logger.warning(
                "Attempted to save state, but no current state or session ID "
                "is loaded."
            )
            return

        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                try:
                    # Use INSERT ... ON CONFLICT DO UPDATE (UPSERT)
                    await conn.execute(
                        """
                        INSERT INTO player_states (
                            player_id,
                            session_id,
                            state_data,
                            updated_at
                            )
                        VALUES ($1, $2, $3, NOW())
                        ON CONFLICT (player_id, session_id) DO UPDATE
                        SET state_data = EXCLUDED.state_data,
                            updated_at = NOW();
                        """,
                        self._current_state.player_id,
                        self._current_session_id,
                        self._current_state.model_dump_json(exclude_none=True),
                    )
                    logger.info(
                        f"Saved player state {self._current_state.player_id} "
                        f"for session {self._current_session_id}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error saving player state "
                        f"{self._current_state.player_id}: {e}"
                    )
                    raise  # Re-raise after logging

    def get_state(self) -> PlayerState | None:
        """Returns the current in-memory player state."""
        return self._current_state

    async def update_from_action(self, action_result: ActionResult) -> None:
        """Updates the player state based on the results of an action."""
        if not self._current_state:
            logger.warning("Cannot update state from action: No player state loaded.")
            return

        # Update Location
        if action_result.location_change and action_result.new_location_id:
            await self.update_location(action_result.new_location_id)

        # Update Inventory
        if action_result.inventory_changes:
            for change in action_result.inventory_changes:
                action = change.get("action")
                if action == "add":
                    item_data = change.get("item")
                    if isinstance(item_data, dict):
                        item = InventoryItem(**item_data)
                    elif isinstance(item_data, InventoryItem):
                        item = item_data
                    else:
                        logger.warning(
                            "Invalid item data format in inventory change: "
                            f"{item_data}"
                        )
                        continue
                    await self.add_inventory_item(item)
                elif action == "remove":
                    item_id = change.get("item_id")
                    quantity = change.get("quantity", 1)
                    if item_id:
                        # Convert string UUID to UUID object if needed
                        if isinstance(item_id, str):
                            from uuid import UUID

                            item_id = UUID(item_id)
                        await self.remove_inventory_item(item_id, quantity)
                elif action == "update":
                    item_id = change.get("item_id")
                    updates = change.get("updates")
                    if item_id and updates:
                        # Convert string UUID to UUID object if needed
                        if isinstance(item_id, str):
                            from uuid import UUID

                            item_id = UUID(item_id)
                        await self.update_inventory_item(item_id, updates)

        # Update Knowledge
        if action_result.knowledge_updates:
            for knowledge_data in action_result.knowledge_updates:
                if isinstance(knowledge_data, dict):
                    knowledge = PlayerKnowledge(**knowledge_data)
                elif isinstance(knowledge_data, PlayerKnowledge):
                    knowledge = knowledge_data
                else:
                    logger.warning(f"Invalid knowledge data format: {knowledge_data}")
                    continue
                await self.add_knowledge(knowledge)

        # Update Stats
        if action_result.stat_changes:
            await self.update_stats(action_result.stat_changes)

        # Update Progress (Quests, Flags)
        if action_result.progress_updates:
            await self.update_progress(action_result.progress_updates)

        # Persist changes if any occurred
        # Consider if saving should happen here
        # or be controlled externally (e.g., by GameStateManager)
        # If saving here:
        # if state_changed:
        #     await self.save_state()

    async def update_location(self, new_location_id: UUID) -> None:
        """Updates the player's current location and visited list."""
        if not self._current_state:
            return
        if self._current_state.current_location_id != new_location_id:
            self._current_state.current_location_id = new_location_id
            if new_location_id not in self._current_state.visited_locations:
                self._current_state.visited_locations.append(new_location_id)
            logger.debug(
                f"Player {self._current_state.player_id} "
                f"moved to location {new_location_id}"
            )
            # Consider saving state here or marking as dirty

    async def add_inventory_item(self, item: InventoryItem) -> None:
        """Adds an item to the player's inventory."""
        if not self._current_state:
            return
        # Check if item already exists
        # (by item_id or maybe name for stackable?)
        existing_item = next(
            (i for i in self._current_state.inventory if i.item_id == item.item_id),
            None,
        )
        if existing_item:
            # Handle stacking or specific logic if needed
            existing_item.quantity += item.quantity
            logger.debug(
                f"Updated quantity for item {item.name} ({item.item_id}) "
                f"to {existing_item.quantity}"
            )
        else:
            self._current_state.inventory.append(item)
            logger.debug(f"Added item {item.name} ({item.item_id}) to inventory")
        # Consider saving state here or marking as dirty

    async def remove_inventory_item(self, item_id: UUID, quantity: int = 1) -> None:
        """Removes an item (or quantity) from the player's inventory."""
        if not self._current_state:
            return
        item_to_remove = next(
            (i for i in self._current_state.inventory if i.item_id == item_id),
            None,
        )
        if item_to_remove:
            if item_to_remove.quantity > quantity:
                item_to_remove.quantity -= quantity
                logger.debug(
                    f"Decreased quantity for item {item_to_remove.name} "
                    f"({item_id}) to {item_to_remove.quantity}"
                )
            else:
                self._current_state.inventory.remove(item_to_remove)
                logger.debug(
                    f"Removed item {item_to_remove.name} ({item_id}) " f"from inventory"
                )
            # Consider saving state here or marking as dirty
        else:
            logger.warning(
                f"Attempted to remove item {item_id} not found in inventory."
            )

    async def update_inventory_item(
        self, item_id: UUID, updates: dict[str, Any]
    ) -> None:
        """Updates attributes of an existing inventory item."""
        if not self._current_state:
            return
        item_to_update = next(
            (i for i in self._current_state.inventory if i.item_id == item_id),
            None,
        )
        if item_to_update:
            for key, value in updates.items():
                if hasattr(item_to_update, key):
                    setattr(item_to_update, key, value)
                else:
                    # Add or update custom attributes in the attributes dict
                    item_to_update.attributes[key] = value
            logger.debug(
                f"Updated item {item_to_update.name} ({item_id}) with: " f"{updates}"
            )

        else:
            logger.warning(
                f"Attempted to update item {item_id} not found in inventory."
            )

    async def add_knowledge(self, knowledge: PlayerKnowledge) -> None:
        """Adds a piece of knowledge to the player's state."""
        if not self._current_state:
            return
        # Avoid duplicates based on topic or a more robust check?
        if not any(k.topic == knowledge.topic for k in self._current_state.knowledge):
            self._current_state.knowledge.append(knowledge)
            logger.debug(f"Added knowledge: {knowledge.topic}")
            # Consider saving state here or marking as dirty
        else:
            logger.debug(f"Knowledge topic '{knowledge.topic}' already exists.")

    async def update_stats(self, stat_changes: dict[str, int | float]) -> None:
        """
        Updates player stats, ensuring values stay within bounds
        (e.g., health).
        """
        if not self._current_state:
            return
        stats = self._current_state.stats
        for stat, change in stat_changes.items():
            if hasattr(stats, stat):
                current_value = getattr(stats, stat)
                new_value = current_value + change

                # Apply constraints (e.g., health clamping)
                if stat == "health":
                    new_value = max(0, min(stats.max_health, new_value))
                elif stat == "mana":
                    new_value = max(0, min(stats.max_mana, new_value))
                # Add other constraints as needed

                setattr(stats, stat, new_value)
                logger.debug(f"Updated stat '{stat}' to {new_value}")
            else:
                logger.warning(f"Attempted to update unknown stat: {stat}")
        # Consider saving state here or marking as dirty

    async def update_progress(self, progress_updates: dict[str, Any]) -> None:
        """Updates player progress (quests, flags)."""
        if not self._current_state:
            return
        progress = self._current_state.progress

        # Example: Update quest state
        if "quest_update" in progress_updates:
            quest_id_str = progress_updates["quest_update"].get("quest_id")
            quest_state = progress_updates["quest_update"].get("state")
            if quest_id_str and quest_state:
                try:
                    quest_id = UUID(quest_id_str)
                    progress.active_quests[quest_id] = quest_state
                    logger.debug(f"Updated quest {quest_id} state: {quest_state}")
                except ValueError:
                    logger.warning(f"Invalid UUID format for quest_id: {quest_id_str}")

        # Example: Set/unset flags
        if "flag_update" in progress_updates:
            flag_name = progress_updates["flag_update"].get("name")
            flag_value = progress_updates["flag_update"].get("value")
            if flag_name is not None and isinstance(flag_value, bool):
                progress.flags[flag_name] = flag_value
                logger.debug(f"Updated flag '{flag_name}' to {flag_value}")

        # Example: Add completed quest
        if "quest_complete" in progress_updates:
            quest_id_str = progress_updates["quest_complete"].get("quest_id")
            if quest_id_str:
                try:
                    quest_id = UUID(quest_id_str)
                    if quest_id in progress.active_quests:
                        del progress.active_quests[quest_id]
                    if quest_id not in progress.completed_quests:
                        progress.completed_quests.append(quest_id)
                        logger.debug(f"Marked quest {quest_id} as completed.")
                except ValueError:
                    logger.warning(f"Invalid UUID format for quest_id: {quest_id_str}")

        # Add more specific progress update logic as needed
        # Consider saving state here or marking as dirty

    async def get_player_id(self) -> UUID | None:
        """Returns the ID of the currently loaded player."""
        return self._current_state.player_id if self._current_state else None

    async def get_current_location_id(self) -> UUID | None:
        """Returns the ID of the player's current location."""
        return self._current_state.current_location_id if self._current_state else None

    async def get_inventory(self) -> list[InventoryItem]:
        """Returns the player's current inventory."""
        return self._current_state.inventory if self._current_state else []

    async def get_knowledge(self) -> list[PlayerKnowledge]:
        """Returns the player's acquired knowledge."""
        return self._current_state.knowledge if self._current_state else []

    async def get_stats(self) -> PlayerStats | None:
        """Returns the player's current stats model."""
        return self._current_state.stats if self._current_state else None

    async def shutdown(self) -> None:
        """Perform any cleanup, like ensuring the final state is saved."""
        logger.info("Shutting down PlayerStateTracker...")
        # Decide if auto-save on shutdown is desired
        # await self.save_state()
        logger.info("PlayerStateTracker shut down.")
