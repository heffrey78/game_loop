"""
Inventory command handler for the Game Loop.
Handles INVENTORY, TAKE, and DROP commands for inventory management.
"""

import logging
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

from game_loop.core.command_handlers.base_handler import CommandHandler
from game_loop.core.input_processor import ParsedCommand
from game_loop.state.models import ActionResult

if TYPE_CHECKING:
    from game_loop.state.manager import GameStateManager
    from game_loop.state.models import Location, PlayerState, WorldObject

logger = logging.getLogger(__name__)


class InventoryCommandHandler(CommandHandler):
    """
    Handler for inventory commands (INVENTORY, TAKE, DROP) in the Game Loop.

    Handles comprehensive inventory management with capacity tracking,
    item organization, and rich displays.
    """

    def __init__(self, console: Console, state_manager: "GameStateManager"):
        """
        Initialize the inventory handler.

        Args:
            console: Rich console for output
            state_manager: Game state manager for accessing and updating game state
        """
        super().__init__(console, state_manager)

        # Command mappings
        self.inventory_commands = ["inventory", "i", "items", "check inventory"]
        self.take_commands = ["take", "get", "pick up", "grab", "collect"]
        self.drop_commands = ["drop", "discard", "put down", "leave", "release"]

        # Default inventory limits
        self.default_max_items = 10
        self.default_max_weight = 100.0

    async def handle(self, command: ParsedCommand) -> ActionResult:
        """
        Handle an inventory command and return the result.

        Args:
            command: The parsed inventory command to handle

        Returns:
            ActionResult describing the inventory operation outcome
        """
        # Get required game state
        player_state, current_location, world_state = await self.get_required_state()

        # Basic validation
        if not player_state:
            return ActionResult(
                success=False, feedback_message="Error: Cannot access player state."
            )

        # Validate location for take/drop commands
        if not current_location and (
            self._is_take_command(command.action.lower())
            or self._is_drop_command(command.action.lower())
        ):
            return ActionResult(
                success=False,
                feedback_message="Error: Cannot determine current location.",
            )

        # Determine command type and handle accordingly
        action_lower = command.action.lower()

        if self._is_inventory_command(action_lower):
            return await self._handle_inventory_display(player_state)
        elif self._is_take_command(action_lower):
            return await self._handle_take_item(command, current_location, player_state)
        elif self._is_drop_command(action_lower):
            return await self._handle_drop_item(command, current_location, player_state)
        else:
            return ActionResult(
                success=False,
                feedback_message="I don't understand that inventory command.",
            )

    def _is_inventory_command(self, action: str) -> bool:
        """Check if this is an inventory display command."""
        return action in self.inventory_commands

    def _is_take_command(self, action: str) -> bool:
        """Check if this is a take/get command."""
        return any(take_cmd in action for take_cmd in self.take_commands)

    def _is_drop_command(self, action: str) -> bool:
        """Check if this is a drop command."""
        return any(drop_cmd in action for drop_cmd in self.drop_commands)

    async def _handle_inventory_display(
        self, player_state: "PlayerState"
    ) -> ActionResult:
        """
        Handle inventory display command.

        Args:
            player_state: Current player state

        Returns:
            ActionResult with inventory display
        """
        try:
            # Get inventory
            inventory = getattr(player_state, "inventory", {})

            if not inventory:
                return ActionResult(
                    success=True, feedback_message="[dim]Your inventory is empty.[/dim]"
                )

            # Create rich inventory display
            display = self._create_inventory_display(inventory, player_state)

            return ActionResult(success=True, feedback_message=display)

        except Exception as e:
            logger.error(f"Error displaying inventory: {e}")
            return ActionResult(
                success=False,
                feedback_message="You have trouble checking your inventory.",
            )

    async def _handle_take_item(
        self,
        command: ParsedCommand,
        current_location: "Location | None",
        player_state: "PlayerState",
    ) -> ActionResult:
        """
        Handle taking an item from the location.

        Args:
            command: The parsed take command
            current_location: Current location
            player_state: Current player state

        Returns:
            ActionResult describing the take operation
        """
        if not current_location:
            return ActionResult(
                success=False,
                feedback_message="Error: Cannot determine current location.",
            )

        # Extract item name
        item_name = command.subject or command.target
        if not item_name:
            return ActionResult(
                success=False, feedback_message="What would you like to take?"
            )

        normalized_name = self.normalize_name(item_name)

        try:
            # Find the item in the location
            item_to_take = await self._find_location_item(
                normalized_name, current_location
            )

            if not item_to_take:
                return ActionResult(
                    success=False,
                    feedback_message=f"You don't see any '{item_name}' here to take.",
                )

            # Check if item can be taken
            if not self._can_take_item(item_to_take):
                reason = getattr(
                    item_to_take, "take_restriction_reason", "You can't take that."
                )
                return ActionResult(success=False, feedback_message=reason)

            # Check inventory capacity
            capacity_check = self._check_inventory_capacity(item_to_take, player_state)
            if not capacity_check["success"]:
                return ActionResult(
                    success=False, feedback_message=capacity_check["message"]
                )

            # Perform the take operation
            take_result = await self._perform_take_item(
                item_to_take, current_location, player_state
            )

            return take_result

        except Exception as e:
            logger.error(f"Error taking item '{item_name}': {e}")
            return ActionResult(
                success=False,
                feedback_message=f"You have trouble taking the {item_name}.",
            )

    async def _handle_drop_item(
        self,
        command: ParsedCommand,
        current_location: "Location | None",
        player_state: "PlayerState",
    ) -> ActionResult:
        """
        Handle dropping an item from inventory.

        Args:
            command: The parsed drop command
            current_location: Current location
            player_state: Current player state

        Returns:
            ActionResult describing the drop operation
        """
        if not current_location:
            return ActionResult(
                success=False,
                feedback_message="Error: Cannot determine current location.",
            )

        # Extract item name
        item_name = command.subject or command.target
        if not item_name:
            return ActionResult(
                success=False, feedback_message="What would you like to drop?"
            )

        normalized_name = self.normalize_name(item_name)

        try:
            # Find the item in inventory
            item_to_drop = await self._find_inventory_item(
                normalized_name, player_state
            )

            if not item_to_drop:
                return ActionResult(
                    success=False,
                    feedback_message=f"You don't have any '{item_name}' to drop.",
                )

            # Check if item can be dropped
            if not self._can_drop_item(item_to_drop):
                reason = getattr(
                    item_to_drop, "drop_restriction_reason", "You can't drop that here."
                )
                return ActionResult(success=False, feedback_message=reason)

            # Perform the drop operation
            drop_result = await self._perform_drop_item(
                item_to_drop, current_location, player_state
            )

            return drop_result

        except Exception as e:
            logger.error(f"Error dropping item '{item_name}': {e}")
            return ActionResult(
                success=False,
                feedback_message=f"You have trouble dropping the {item_name}.",
            )

    def _create_inventory_display(
        self, inventory: dict, player_state: "PlayerState"
    ) -> str:
        """
        Create a rich display of the inventory.

        Args:
            inventory: Player's inventory
            player_state: Current player state

        Returns:
            Formatted inventory display string
        """
        try:
            display = "[bold]Inventory:[/bold]\n"

            if not inventory:
                return display + "[dim]Empty[/dim]"

            # Create table for better formatting
            table = Table(show_header=True, header_style="bold")
            table.add_column("Item", style="cyan")
            table.add_column("Quantity", style="yellow", justify="right")
            table.add_column("Condition", style="green")

            total_weight = 0.0
            total_items = 0

            # Add items to table
            for item_id, item in inventory.items():
                if isinstance(item, dict):
                    name = item.get("name", "Unknown Item")
                    quantity = item.get("quantity", 1)
                    condition = item.get("condition", "good")
                    weight = item.get("weight", 1.0)
                else:
                    name = getattr(item, "name", "Unknown Item")
                    quantity = getattr(item, "quantity", 1)
                    condition = getattr(item, "condition", "good")
                    weight = getattr(item, "weight", 1.0)

                table.add_row(name, str(quantity), condition)
                total_weight += weight * quantity
                total_items += quantity

            # Convert table to string (simplified for text output)
            items_list = []
            for item_id, item in inventory.items():
                if isinstance(item, dict):
                    name = item.get("name", "Unknown Item")
                    quantity = item.get("quantity", 1)
                    condition = item.get("condition", "good")
                else:
                    name = getattr(item, "name", "Unknown Item")
                    quantity = getattr(item, "quantity", 1)
                    condition = getattr(item, "condition", "good")

                if quantity > 1:
                    items_list.append(f"[cyan]{name}[/cyan] (x{quantity})")
                else:
                    items_list.append(f"[cyan]{name}[/cyan]")

                if condition != "good":
                    items_list[-1] += f" [dim]({condition})[/dim]"

            display += "\n".join(f"  • {item}" for item in items_list)

            # Add capacity information
            max_items = getattr(
                player_state, "max_inventory_items", self.default_max_items
            )
            max_weight = getattr(
                player_state, "max_inventory_weight", self.default_max_weight
            )

            display += f"\n\n[dim]Carrying: {total_items}/{max_items} items"
            if max_weight > 0:
                display += f", {total_weight:.1f}/{max_weight:.1f} weight"
            display += "[/dim]"

            return display

        except Exception as e:
            logger.error(f"Error creating inventory display: {e}")
            return "[bold]Inventory:[/bold]\n[dim]Error displaying inventory[/dim]"

    async def _find_location_item(
        self, item_name: str, location: "Location"
    ) -> "WorldObject | None":
        """
        Find an item in the current location.

        Args:
            item_name: Normalized name of item to find
            location: Location to search

        Returns:
            The item object if found, None otherwise
        """
        try:
            if not hasattr(location, "objects") or not location.objects:
                return None

            for obj in location.objects:
                obj_name = self.normalize_name(getattr(obj, "name", ""))
                if obj_name and item_name in obj_name:
                    return obj

            return None

        except Exception as e:
            logger.error(f"Error finding location item: {e}")
            return None

    async def _find_inventory_item(
        self, item_name: str, player_state: "PlayerState"
    ) -> dict | None:
        """
        Find an item in the player's inventory.

        Args:
            item_name: Normalized name of item to find
            player_state: Player state to search

        Returns:
            The item data if found, None otherwise
        """
        try:
            inventory = getattr(player_state, "inventory", {})
            if not inventory:
                return None

            for item_id, item in inventory.items():
                if isinstance(item, dict):
                    name = self.normalize_name(item.get("name", ""))
                else:
                    name = self.normalize_name(getattr(item, "name", ""))

                if name and item_name in name:
                    return {"id": item_id, "data": item}

            return None

        except Exception as e:
            logger.error(f"Error finding inventory item: {e}")
            return None

    def _can_take_item(self, item: "WorldObject") -> bool:
        """
        Check if an item can be taken.

        Args:
            item: Item to check

        Returns:
            True if item can be taken
        """
        try:
            # Check if item is takeable
            if hasattr(item, "takeable"):
                return getattr(item, "takeable", True)

            # Check for specific restrictions
            if hasattr(item, "fixed") and item.fixed:
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking if item can be taken: {e}")
            return False

    def _can_drop_item(self, item: dict) -> bool:
        """
        Check if an item can be dropped.

        Args:
            item: Item to check

        Returns:
            True if item can be dropped
        """
        try:
            item_data = item.get("data", {})

            # Check for drop restrictions
            if isinstance(item_data, dict):
                if item_data.get("no_drop", False):
                    return False
                if item_data.get("quest_item", False):
                    return False
            else:
                if getattr(item_data, "no_drop", False):
                    return False
                if getattr(item_data, "quest_item", False):
                    return False

            return True

        except Exception as e:
            logger.error(f"Error checking if item can be dropped: {e}")
            return True

    def _check_inventory_capacity(
        self, item: "WorldObject", player_state: "PlayerState"
    ) -> dict:
        """
        Check if there's capacity for the item in inventory.

        Args:
            item: Item to check capacity for
            player_state: Current player state

        Returns:
            Dict with 'success' boolean and 'message' string
        """
        try:
            inventory = getattr(player_state, "inventory", {})
            max_items = getattr(
                player_state, "max_inventory_items", self.default_max_items
            )
            max_weight = getattr(
                player_state, "max_inventory_weight", self.default_max_weight
            )

            # Check item count
            current_items = len(inventory)
            if current_items >= max_items:
                return {
                    "success": False,
                    "message": f"Your inventory is full ({current_items}/{max_items} items). Drop something first.",
                }

            # Check weight if applicable
            if max_weight > 0:
                current_weight = sum(
                    (
                        item.get("weight", 1.0) * item.get("quantity", 1)
                        if isinstance(item, dict)
                        else getattr(item, "weight", 1.0) * getattr(item, "quantity", 1)
                    )
                    for item in inventory.values()
                )

                item_weight = getattr(item, "weight", 1.0)
                if current_weight + item_weight > max_weight:
                    return {
                        "success": False,
                        "message": f"That would be too heavy ({current_weight + item_weight:.1f}/{max_weight:.1f}). Drop something first.",
                    }

            return {"success": True, "message": ""}

        except Exception as e:
            logger.error(f"Error checking inventory capacity: {e}")
            return {"success": True, "message": ""}  # Default to allowing

    async def _perform_take_item(
        self, item: "WorldObject", location: "Location", player_state: "PlayerState"
    ) -> ActionResult:
        """
        Perform the actual take operation.

        Args:
            item: Item to take
            location: Current location
            player_state: Current player state

        Returns:
            ActionResult describing the outcome
        """
        try:
            # Create item data for inventory
            item_data = {
                "name": getattr(item, "name", "Unknown Item"),
                "description": getattr(item, "description", "An item."),
                "weight": getattr(item, "weight", 1.0),
                "quantity": 1,
                "condition": getattr(item, "condition", "good"),
                "type": getattr(item, "type", "misc"),
                "value": getattr(item, "value", 0),
            }

            # Add to inventory
            inventory = getattr(player_state, "inventory", {})
            if not inventory:
                inventory = {}
                player_state.inventory = inventory

            # Generate unique ID for item
            item_id = f"item_{len(inventory) + 1}"
            inventory[item_id] = item_data

            # Remove from location
            if hasattr(location, "objects") and location.objects:
                location.objects = [obj for obj in location.objects if obj != item]

            # Update state
            await self.state_manager.player_tracker.update_state(player_state)
            await self.state_manager.update_location_details(location)

            return ActionResult(
                success=True,
                feedback_message=f"You take the [cyan]{item_data['name']}[/cyan].",
            )

        except Exception as e:
            logger.error(f"Error performing take operation: {e}")
            return ActionResult(
                success=False,
                feedback_message="Something went wrong while taking the item.",
            )

    async def _perform_drop_item(
        self, item: dict, location: "Location", player_state: "PlayerState"
    ) -> ActionResult:
        """
        Perform the actual drop operation.

        Args:
            item: Item to drop (with 'id' and 'data' keys)
            location: Current location
            player_state: Current player state

        Returns:
            ActionResult describing the outcome
        """
        try:
            item_id = item["id"]
            item_data = item["data"]

            # Remove from inventory
            inventory = getattr(player_state, "inventory", {})
            if item_id in inventory:
                del inventory[item_id]

            # Create world object for location
            # This is a simplified approach - in a full implementation,
            # you'd create a proper WorldObject instance
            location_objects = getattr(location, "objects", [])
            if not location_objects:
                location_objects = []
                location.objects = location_objects

            # For now, we'll add a basic representation
            # In a full implementation, this would be a proper WorldObject
            dropped_object = type(
                "DroppedObject",
                (),
                {
                    "name": (
                        item_data.get("name", "Unknown Item")
                        if isinstance(item_data, dict)
                        else getattr(item_data, "name", "Unknown Item")
                    ),
                    "description": (
                        item_data.get("description", "An item.")
                        if isinstance(item_data, dict)
                        else getattr(item_data, "description", "An item.")
                    ),
                    "takeable": True,
                    "weight": (
                        item_data.get("weight", 1.0)
                        if isinstance(item_data, dict)
                        else getattr(item_data, "weight", 1.0)
                    ),
                },
            )()

            location_objects.append(dropped_object)

            # Update state
            await self.state_manager.player_tracker.update_state(player_state)
            await self.state_manager.update_location_details(location)

            item_name = (
                item_data.get("name", "item")
                if isinstance(item_data, dict)
                else getattr(item_data, "name", "item")
            )

            return ActionResult(
                success=True, feedback_message=f"You drop the [cyan]{item_name}[/cyan]."
            )

        except Exception as e:
            logger.error(f"Error performing drop operation: {e}")
            return ActionResult(
                success=False,
                feedback_message="Something went wrong while dropping the item.",
            )
