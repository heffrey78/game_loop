"""
Target usage handler implementation.
"""

import logging

from game_loop.state.models import ActionResult, InventoryItem, Location, PlayerState

from .base import UsageHandler

logger = logging.getLogger(__name__)


class TargetUsageHandler(UsageHandler):
    """
    Handler for "use X on Y" usage scenarios.
    Handles using an item on another object or NPC.
    """

    async def validate(
        self,
        item_to_use: InventoryItem,
        player_state: PlayerState,
        current_location: Location,
    ) -> bool:
        """Validate if the target usage is possible."""
        # Validation would be moved here
        # For now, just return True as validation is done in handle
        return True

    async def handle(
        self,
        item_to_use: InventoryItem,
        target_name: str | None,
        player_state: PlayerState,
        current_location: Location,
    ) -> ActionResult:
        """Handle using an item on another object or NPC."""
        if not target_name:
            return ActionResult(
                success=False,
                feedback_message="What do you want to use this on?",
            )

        normalized_target_name = target_name.lower()
        item_name = item_to_use.name.lower()

        # Handle writing tools on writable surfaces
        if self._is_writing_tool(item_name):
            return await self._handle_writing_interaction(
                item_to_use, normalized_target_name, player_state, current_location
            )

        # Handle keys on lockable objects
        elif self._is_key_tool(item_name):
            return await self._handle_key_interaction(
                item_to_use, normalized_target_name, player_state, current_location
            )

        # Handle tools on repairable objects
        elif self._is_repair_tool(item_name):
            return await self._handle_repair_interaction(
                item_to_use, normalized_target_name, player_state, current_location
            )

        # Handle cleaning supplies
        elif self._is_cleaning_tool(item_name):
            return await self._handle_cleaning_interaction(
                item_to_use, normalized_target_name, player_state, current_location
            )

        # Generic tool usage
        else:
            return await self._handle_generic_tool_usage(
                item_to_use, normalized_target_name, player_state, current_location
            )

    def _is_writing_tool(self, item_name: str) -> bool:
        """Check if item is a writing tool."""
        writing_tools = ["pen", "pencil", "marker", "quill", "chalk", "stylus"]
        return any(tool in item_name for tool in writing_tools)

    def _is_key_tool(self, item_name: str) -> bool:
        """Check if item is a key or unlocking tool."""
        key_tools = ["key", "keycard", "lockpick", "crowbar"]
        return any(tool in item_name for tool in key_tools)

    def _is_repair_tool(self, item_name: str) -> bool:
        """Check if item is a repair tool."""
        repair_tools = ["screwdriver", "wrench", "hammer", "tool kit", "pliers"]
        return any(tool in item_name for tool in repair_tools)

    def _is_cleaning_tool(self, item_name: str) -> bool:
        """Check if item is for cleaning."""
        cleaning_tools = ["cloth", "rag", "brush", "cleaner", "soap"]
        return any(tool in item_name for tool in cleaning_tools)

    async def _handle_writing_interaction(
        self,
        item_to_use: InventoryItem,
        target_name: str,
        player_state: PlayerState,
        current_location: Location,
    ) -> ActionResult:
        """Handle writing tool on writable surface."""
        # Check if target is writable
        writable_surfaces = [
            "paper",
            "letter",
            "note",
            "wall",
            "book",
            "document",
            "resignation letter",
        ]

        if not any(surface in target_name for surface in writable_surfaces):
            return ActionResult(
                success=False,
                feedback_message=f"You can't write on the {target_name} with the {item_to_use.name}.",
            )

        # Check if target exists in location
        if not await self._target_exists_in_location(target_name, current_location):
            return ActionResult(
                success=False,
                feedback_message=f"I don't see '{target_name}' here to write on.",
            )

        # Successful writing interaction
        return ActionResult(
            success=True,
            feedback_message=f"You use the {item_to_use.name} to write on the {target_name}. "
            f"What would you like to write? (Try: write [text] on {target_name})",
            object_changes=[
                {
                    "interaction_type": "writing_setup",
                    "tool_used": item_to_use.name,
                    "target": target_name,
                }
            ],
        )

    async def _handle_key_interaction(
        self,
        item_to_use: InventoryItem,
        target_name: str,
        player_state: PlayerState,
        current_location: Location,
    ) -> ActionResult:
        """Handle key or unlocking tool on lockable object."""
        lockable_objects = ["door", "chest", "box", "safe", "cabinet", "locker"]

        if not any(obj in target_name for obj in lockable_objects):
            return ActionResult(
                success=False,
                feedback_message=f"The {item_to_use.name} doesn't seem useful on the {target_name}.",
            )

        if not await self._target_exists_in_location(target_name, current_location):
            return ActionResult(
                success=False, feedback_message=f"I don't see '{target_name}' here."
            )

        return ActionResult(
            success=True,
            feedback_message=f"You use the {item_to_use.name} on the {target_name}. "
            f"It opens with a satisfying click, revealing its contents.",
            object_changes=[
                {
                    "interaction_type": "unlock",
                    "tool_used": item_to_use.name,
                    "target": target_name,
                    "state_change": "opened",
                }
            ],
        )

    async def _handle_repair_interaction(
        self,
        item_to_use: InventoryItem,
        target_name: str,
        player_state: PlayerState,
        current_location: Location,
    ) -> ActionResult:
        """Handle repair tool on broken object."""
        repairable_objects = [
            "machine",
            "device",
            "equipment",
            "instrument",
            "engine",
            "computer",
        ]

        if not any(obj in target_name for obj in repairable_objects):
            return ActionResult(
                success=False,
                feedback_message=f"The {target_name} doesn't appear to need repair with the {item_to_use.name}.",
            )

        if not await self._target_exists_in_location(target_name, current_location):
            return ActionResult(
                success=False,
                feedback_message=f"I don't see '{target_name}' here to repair.",
            )

        return ActionResult(
            success=True,
            feedback_message=f"You carefully use the {item_to_use.name} to repair the {target_name}. "
            f"It now looks much better and seems to function properly.",
            object_changes=[
                {
                    "interaction_type": "repair",
                    "tool_used": item_to_use.name,
                    "target": target_name,
                    "state_change": "repaired",
                }
            ],
        )

    async def _handle_cleaning_interaction(
        self,
        item_to_use: InventoryItem,
        target_name: str,
        player_state: PlayerState,
        current_location: Location,
    ) -> ActionResult:
        """Handle cleaning tool on dirty object."""
        cleanable_objects = [
            "surface",
            "mirror",
            "glass",
            "window",
            "equipment",
            "screen",
        ]

        if not any(obj in target_name for obj in cleanable_objects):
            return ActionResult(
                success=False,
                feedback_message=f"You can't clean the {target_name} with the {item_to_use.name}.",
            )

        if not await self._target_exists_in_location(target_name, current_location):
            return ActionResult(
                success=False,
                feedback_message=f"I don't see '{target_name}' here to clean.",
            )

        return ActionResult(
            success=True,
            feedback_message=f"You clean the {target_name} with the {item_to_use.name}. "
            f"It's now spotless and gleaming.",
            object_changes=[
                {
                    "interaction_type": "clean",
                    "tool_used": item_to_use.name,
                    "target": target_name,
                    "state_change": "cleaned",
                }
            ],
        )

    async def _handle_generic_tool_usage(
        self,
        item_to_use: InventoryItem,
        target_name: str,
        player_state: PlayerState,
        current_location: Location,
    ) -> ActionResult:
        """Handle generic tool usage on objects."""
        if not await self._target_exists_in_location(target_name, current_location):
            return ActionResult(
                success=False, feedback_message=f"I don't see '{target_name}' here."
            )

        # Provide helpful feedback for unimplemented interactions
        return ActionResult(
            success=False,
            feedback_message=f"You try to use the {item_to_use.name} on the {target_name}, "
            f"but you're not sure how they would work together. "
            f"Try examining them more closely first.",
            object_changes=[
                {
                    "interaction_type": "attempted_generic",
                    "tool_used": item_to_use.name,
                    "target": target_name,
                }
            ],
        )

    async def _target_exists_in_location(
        self, target_name: str, location: Location
    ) -> bool:
        """Check if target exists in the current location."""
        # Check location description
        location_desc = getattr(location, "description", "").lower()
        if target_name in location_desc:
            return True

        # Check location objects
        objects = getattr(location, "objects", [])
        for obj in objects:
            obj_name = getattr(obj, "name", "").lower()
            if target_name in obj_name or obj_name in target_name:
                return True

        # Check for partial matches
        target_words = target_name.split()
        for word in target_words:
            if len(word) > 3 and word in location_desc:
                return True

        return False
