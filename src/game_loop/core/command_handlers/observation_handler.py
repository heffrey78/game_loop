"""
Observation command handler for the Game Loop.
Handles LOOK and EXAMINE commands for observing the game world.
"""

import logging
from typing import TYPE_CHECKING

from rich.console import Console

from game_loop.core.command_handlers.base_handler import CommandHandler
from game_loop.core.input_processor import ParsedCommand
from game_loop.state.models import ActionResult

if TYPE_CHECKING:
    from game_loop.state.manager import GameStateManager
    from game_loop.state.models import Location, PlayerState

logger = logging.getLogger(__name__)


class ObservationCommandHandler(CommandHandler):
    """
    Handler for observation commands (LOOK, EXAMINE) in the Game Loop.

    Handles detailed examination of locations, objects, and NPCs with
    rich descriptions and contextual information.
    """

    def __init__(self, console: Console, state_manager: "GameStateManager"):
        """
        Initialize the observation handler.

        Args:
            console: Rich console for output
            state_manager: Game state manager for accessing game state
        """
        super().__init__(console, state_manager)

        # Commands that trigger general location observation
        self.look_commands = ["look", "l", "look around", "examine surroundings"]

        # Commands that trigger specific object examination
        self.examine_commands = ["examine", "inspect", "look at", "check", "study"]

    async def handle(self, command: ParsedCommand) -> ActionResult:
        """
        Handle an observation command and return the result.

        Args:
            command: The parsed observation command to handle

        Returns:
            ActionResult describing what was observed
        """
        # Get required game state
        player_state, current_location, world_state = await self.get_required_state()

        # Basic validation
        if not current_location:
            return ActionResult(
                success=False,
                feedback_message="Error: Cannot determine current location.",
            )

        # Validate player state
        if not player_state:
            return ActionResult(
                success=False, feedback_message="Error: Cannot access player state."
            )

        # Determine what type of observation this is
        if self._is_general_look(command):
            return await self._handle_general_look(current_location, player_state)
        else:
            return await self._handle_specific_examine(
                command, current_location, player_state
            )

    def _is_general_look(self, command: ParsedCommand) -> bool:
        """
        Determine if this is a general look command (no specific target).

        Args:
            command: The parsed command

        Returns:
            True if this is a general look command
        """
        action_lower = command.action.lower()

        # Check if it's a general look command
        if action_lower in self.look_commands:
            return True

        # Check if it's "look" or "examine" without a specific target
        if action_lower in ["look", "examine"] and not command.subject:
            return True

        return False

    async def _handle_general_look(
        self, current_location: "Location", player_state: "PlayerState"
    ) -> ActionResult:
        """
        Handle a general look around command.

        Args:
            current_location: Current location to describe
            player_state: Current player state

        Returns:
            ActionResult with location description
        """
        try:
            # Generate comprehensive location description
            description = await self._generate_location_description(current_location)

            # Add exits information
            exits_info = self._generate_exits_description(current_location)
            if exits_info:
                description += f"\n\n{exits_info}"

            # Add objects information
            objects_info = await self._generate_objects_description(current_location)
            if objects_info:
                description += f"\n\n{objects_info}"

            # Add NPCs information
            npcs_info = await self._generate_npcs_description(current_location)
            if npcs_info:
                description += f"\n\n{npcs_info}"

            # Add atmospheric details
            atmosphere_info = self._generate_atmosphere_description(current_location)
            if atmosphere_info:
                description += f"\n\n{atmosphere_info}"

            return ActionResult(success=True, feedback_message=description)

        except Exception as e:
            logger.error(f"Error generating location description: {e}")
            return ActionResult(
                success=False,
                feedback_message="You have trouble focusing on your surroundings.",
            )

    async def _handle_specific_examine(
        self,
        command: ParsedCommand,
        current_location: "Location",
        player_state: "PlayerState",
    ) -> ActionResult:
        """
        Handle examination of a specific object or entity.

        Args:
            command: The parsed examine command
            current_location: Current location
            player_state: Current player state

        Returns:
            ActionResult with examination details
        """
        target = command.subject or command.target
        if not target:
            return ActionResult(
                success=False, feedback_message="What would you like to examine?"
            )

        # Normalize the target name
        normalized_target = self.normalize_name(target)
        if not normalized_target:
            return ActionResult(
                success=False,
                feedback_message="I don't understand what you want to examine.",
            )

        try:
            # Try to find the target in various places
            examination_result = await self._find_and_examine_target(
                normalized_target, current_location, player_state
            )

            if examination_result:
                return examination_result

            # If not found, provide helpful feedback
            return ActionResult(
                success=False,
                feedback_message=f"You don't see any '{target}' here to examine.",
            )

        except Exception as e:
            logger.error(f"Error examining target '{target}': {e}")
            return ActionResult(
                success=False,
                feedback_message=f"You have trouble examining the {target}.",
            )

    async def _find_and_examine_target(
        self,
        target_name: str,
        current_location: "Location",
        player_state: "PlayerState",
    ) -> ActionResult | None:
        """
        Find and examine a specific target.

        Args:
            target_name: Normalized name of target to examine
            current_location: Current location
            player_state: Current player state

        Returns:
            ActionResult if target found and examined, None otherwise
        """
        # Check player inventory first
        inventory_result = await self._examine_inventory_item(target_name, player_state)
        if inventory_result:
            return inventory_result

        # Check location objects
        location_result = await self._examine_location_object(
            target_name, current_location
        )
        if location_result:
            return location_result

        # Check NPCs
        npc_result = await self._examine_npc(target_name, current_location)
        if npc_result:
            return npc_result

        # Check location features (doors, exits, etc.)
        feature_result = await self._examine_location_feature(
            target_name, current_location
        )
        if feature_result:
            return feature_result

        return None

    async def _examine_inventory_item(
        self, target_name: str, player_state: "PlayerState"
    ) -> ActionResult | None:
        """
        Examine an item in the player's inventory.

        Args:
            target_name: Name of item to examine
            player_state: Current player state

        Returns:
            ActionResult if item found, None otherwise
        """
        try:
            if not hasattr(player_state, "inventory") or not player_state.inventory:
                return None

            # Search through inventory
            for item_id, item in player_state.inventory.items():
                if isinstance(item, dict):
                    item_name = self.normalize_name(item.get("name", ""))
                else:
                    item_name = self.normalize_name(getattr(item, "name", ""))

                if item_name and target_name in item_name:
                    description = await self._generate_item_description(item)
                    return ActionResult(
                        success=True,
                        feedback_message=f"[bold]{item.get('name', 'Item')}[/bold]\n{description}",
                    )

            return None

        except Exception as e:
            logger.error(f"Error examining inventory item: {e}")
            return None

    async def _examine_location_object(
        self, target_name: str, current_location: "Location"
    ) -> ActionResult | None:
        """
        Examine an object in the current location.

        Args:
            target_name: Name of object to examine
            current_location: Current location

        Returns:
            ActionResult if object found, None otherwise
        """
        try:
            if not hasattr(current_location, "objects") or not current_location.objects:
                return None

            # Search through location objects
            for obj in current_location.objects:
                obj_name = self.normalize_name(getattr(obj, "name", ""))
                if obj_name and target_name in obj_name:
                    description = await self._generate_object_description(obj)
                    return ActionResult(
                        success=True,
                        feedback_message=f"[bold]{getattr(obj, 'name', 'Object')}[/bold]\n{description}",
                    )

            return None

        except Exception as e:
            logger.error(f"Error examining location object: {e}")
            return None

    async def _examine_npc(
        self, target_name: str, current_location: "Location"
    ) -> ActionResult | None:
        """
        Examine an NPC in the current location.

        Args:
            target_name: Name of NPC to examine
            current_location: Current location

        Returns:
            ActionResult if NPC found, None otherwise
        """
        try:
            if not hasattr(current_location, "npcs") or not current_location.npcs:
                return None

            # Search through NPCs
            for npc in current_location.npcs:
                npc_name = self.normalize_name(getattr(npc, "name", ""))
                if npc_name and target_name in npc_name:
                    description = await self._generate_npc_description(npc)
                    return ActionResult(
                        success=True,
                        feedback_message=f"[bold]{getattr(npc, 'name', 'Someone')}[/bold]\n{description}",
                    )

            return None

        except Exception as e:
            logger.error(f"Error examining NPC: {e}")
            return None

    async def _examine_location_feature(
        self, target_name: str, current_location: "Location"
    ) -> ActionResult | None:
        """
        Examine a location feature (exit, door, etc.).

        Args:
            target_name: Name of feature to examine
            current_location: Current location

        Returns:
            ActionResult if feature found, None otherwise
        """
        try:
            # Check exits/connections
            if (
                hasattr(current_location, "connections")
                and current_location.connections
            ):
                for direction, connection in current_location.connections.items():
                    if target_name in direction.lower():
                        description = f"There is an exit leading {direction}."
                        if (
                            hasattr(connection, "description")
                            and connection.description
                        ):
                            description += f" {connection.description}"

                        return ActionResult(success=True, feedback_message=description)

            # Check for common location features
            feature_descriptions = {
                "floor": "The floor is solid beneath your feet.",
                "ceiling": "You look up at the ceiling above.",
                "wall": "The walls surround you.",
                "walls": "The walls surround you.",
                "room": f"You are in {current_location.name if hasattr(current_location, 'name') else 'a room'}.",
            }

            if target_name in feature_descriptions:
                return ActionResult(
                    success=True, feedback_message=feature_descriptions[target_name]
                )

            return None

        except Exception as e:
            logger.error(f"Error examining location feature: {e}")
            return None

    async def _generate_location_description(self, location: "Location") -> str:
        """
        Generate a rich description of the location.

        Args:
            location: Location to describe

        Returns:
            Formatted location description
        """
        description = ""

        # Location name
        if hasattr(location, "name") and location.name:
            description += f"[bold]{location.name}[/bold]\n"

        # Main description
        if hasattr(location, "description") and location.description:
            description += location.description
        elif hasattr(location, "short_description") and location.short_description:
            description += location.short_description
        else:
            description += "You find yourself in an unremarkable location."

        return description

    def _generate_exits_description(self, location: "Location") -> str:
        """
        Generate description of available exits.

        Args:
            location: Location to check for exits

        Returns:
            Formatted exits description
        """
        try:
            if not hasattr(location, "connections") or not location.connections:
                return ""

            exits = list(location.connections.keys())
            if not exits:
                return ""

            if len(exits) == 1:
                return f"[dim]There is an exit to the {exits[0]}.[/dim]"
            elif len(exits) == 2:
                return f"[dim]There are exits to the {exits[0]} and {exits[1]}.[/dim]"
            else:
                exit_list = ", ".join(exits[:-1])
                return (
                    f"[dim]There are exits to the {exit_list}, and {exits[-1]}.[/dim]"
                )

        except Exception as e:
            logger.error(f"Error generating exits description: {e}")
            return ""

    async def _generate_objects_description(self, location: "Location") -> str:
        """
        Generate description of objects in the location.

        Args:
            location: Location to check for objects

        Returns:
            Formatted objects description
        """
        try:
            if not hasattr(location, "objects") or not location.objects:
                return ""

            visible_objects = [
                obj for obj in location.objects if not getattr(obj, "hidden", False)
            ]

            if not visible_objects:
                return ""

            if len(visible_objects) == 1:
                obj = visible_objects[0]
                name = getattr(obj, "name", "something")
                return f"[cyan]You see {name} here.[/cyan]"
            else:
                object_names = [
                    getattr(obj, "name", "something") for obj in visible_objects
                ]
                return f"[cyan]You see: {', '.join(object_names)}.[/cyan]"

        except Exception as e:
            logger.error(f"Error generating objects description: {e}")
            return ""

    async def _generate_npcs_description(self, location: "Location") -> str:
        """
        Generate description of NPCs in the location.

        Args:
            location: Location to check for NPCs

        Returns:
            Formatted NPCs description
        """
        try:
            if not hasattr(location, "npcs") or not location.npcs:
                return ""

            if len(location.npcs) == 1:
                npc = location.npcs[0]
                name = getattr(npc, "name", "someone")
                return f"[magenta]{name} is here.[/magenta]"
            else:
                npc_names = [getattr(npc, "name", "someone") for npc in location.npcs]
                return f"[magenta]You see: {', '.join(npc_names)}.[/magenta]"

        except Exception as e:
            logger.error(f"Error generating NPCs description: {e}")
            return ""

    def _generate_atmosphere_description(self, location: "Location") -> str:
        """
        Generate atmospheric details for the location.

        Args:
            location: Location to describe

        Returns:
            Atmospheric description
        """
        try:
            # For now, return empty - this could be enhanced with
            # time-of-day, weather, mood, etc.
            return ""

        except Exception as e:
            logger.error(f"Error generating atmosphere description: {e}")
            return ""

    async def _generate_item_description(self, item) -> str:
        """
        Generate detailed description of an item.

        Args:
            item: Item to describe

        Returns:
            Item description
        """
        try:
            if isinstance(item, dict):
                description = item.get("description", "An ordinary item.")
                condition = item.get("condition", "good")
                if condition != "good":
                    description += f" It appears to be in {condition} condition."
                return description
            else:
                description = getattr(item, "description", "An ordinary item.")
                if hasattr(item, "condition"):
                    condition = getattr(item, "condition", "good")
                    if condition != "good":
                        description += f" It appears to be in {condition} condition."
                return description

        except Exception as e:
            logger.error(f"Error generating item description: {e}")
            return "You can't make out much detail."

    async def _generate_object_description(self, obj) -> str:
        """
        Generate detailed description of a location object.

        Args:
            obj: Object to describe

        Returns:
            Object description
        """
        try:
            description = getattr(obj, "description", "An unremarkable object.")

            # Add additional details if available
            if hasattr(obj, "material"):
                material = obj.material
                description += f" It appears to be made of {material}."

            if hasattr(obj, "condition"):
                condition = obj.condition
                if condition != "good":
                    description += f" It looks {condition}."

            return description

        except Exception as e:
            logger.error(f"Error generating object description: {e}")
            return "You can't make out much detail about this object."

    async def _generate_npc_description(self, npc) -> str:
        """
        Generate detailed description of an NPC.

        Args:
            npc: NPC to describe

        Returns:
            NPC description
        """
        try:
            description = getattr(npc, "description", "An ordinary person.")

            # Add mood/state information if available
            if hasattr(npc, "mood"):
                mood = npc.mood
                if mood:
                    description += f" They seem {mood}."

            return description

        except Exception as e:
            logger.error(f"Error generating NPC description: {e}")
            return "You can't make out much detail about this person."
