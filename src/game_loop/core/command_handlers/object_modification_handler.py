"""
Object modification handler for all forms of object state changes and modifications.

This handler supports writing, opening, adjusting, combining, and other object modifications
that create persistent state changes in the game world.
"""

import logging
from typing import TYPE_CHECKING, Any

from rich.console import Console

from game_loop.core.command_handlers.base_handler import CommandHandler
from game_loop.core.input_processor import ParsedCommand
from game_loop.state.models import ActionResult

if TYPE_CHECKING:
    from game_loop.state.manager import GameStateManager
    from game_loop.state.models import Location, PlayerState

logger = logging.getLogger(__name__)


class ObjectModificationHandler(CommandHandler):
    """
    Handle object modifications including writing, opening, adjusting, combining.

    Supports persistent object state changes that affect the game world
    and can be observed by other players or systems.
    """

    def __init__(self, console: Console, state_manager: "GameStateManager"):
        super().__init__(console, state_manager)

        # Define modification types and their associated verbs
        self.modification_types = {
            "write": {
                "verbs": [
                    "write",
                    "inscribe",
                    "mark",
                    "sign",
                    "scribble",
                    "draw",
                    "carve",
                ],
                "required_tools": [
                    "pen",
                    "pencil",
                    "marker",
                    "quill",
                    "chalk",
                    "knife",
                ],
                "valid_surfaces": [
                    "paper",
                    "wall",
                    "book",
                    "letter",
                    "note",
                    "document",
                    "stone",
                    "wood",
                ],
            },
            "open": {
                "verbs": ["open", "unlock", "unseal"],
                "required_tools": ["key", "keycard", "lockpick", "crowbar"],
                "valid_targets": ["door", "chest", "box", "container", "book", "file"],
            },
            "close": {
                "verbs": ["close", "lock", "seal", "shut"],
                "required_tools": ["key", "keycard"],
                "valid_targets": ["door", "chest", "box", "container", "book", "file"],
            },
            "turn": {
                "verbs": ["turn", "rotate", "twist", "flip"],
                "required_tools": [],
                "valid_targets": ["page", "book", "knob", "dial", "wheel", "handle"],
            },
            "adjust": {
                "verbs": ["adjust", "set", "configure", "tune"],
                "required_tools": ["screwdriver", "wrench", "tool"],
                "valid_targets": ["machine", "device", "instrument", "control", "dial"],
            },
            "repair": {
                "verbs": ["repair", "fix", "mend", "restore"],
                "required_tools": ["tool", "kit", "hammer", "screwdriver", "wrench"],
                "valid_targets": ["machine", "device", "equipment", "instrument"],
            },
            "break": {
                "verbs": ["break", "smash", "destroy", "damage"],
                "required_tools": ["hammer", "rock", "tool"],
                "valid_targets": ["glass", "window", "vase", "pottery", "machine"],
            },
            "combine": {
                "verbs": ["combine", "mix", "merge", "attach", "connect"],
                "required_tools": [],
                "valid_targets": [
                    "any"
                ],  # Special case - depends on object compatibility
            },
            "clean": {
                "verbs": ["clean", "wipe", "polish", "scrub"],
                "required_tools": ["cloth", "rag", "brush", "cleaner"],
                "valid_targets": ["surface", "object", "equipment", "glass", "mirror"],
            },
        }

    async def handle(self, command: ParsedCommand) -> ActionResult:
        """
        Handle object modification commands.

        Args:
            command: The parsed modification command

        Returns:
            ActionResult describing the outcome of the modification
        """
        try:
            # Get required game state
            player_state, current_location, world_state = (
                await self.get_required_state()
            )

            if not current_location or not player_state:
                return ActionResult(
                    success=False,
                    feedback_message="Error: Cannot determine current location or player state.",
                )

            # Parse the modification command
            modification_info = self._parse_modification_command(command)
            if not modification_info:
                return ActionResult(
                    success=False,
                    feedback_message="I don't understand that type of modification.",
                )

            # Validate the modification is possible
            validation_result = await self._validate_modification(
                modification_info, player_state, current_location
            )
            if not validation_result.success:
                return validation_result

            # Execute the modification
            modification_result = await self._execute_modification(
                modification_info, player_state, current_location
            )

            return modification_result

        except Exception as e:
            logger.error(f"Error handling object modification: {e}")
            return ActionResult(
                success=False,
                feedback_message="Something went wrong with the modification. Please try again.",
            )

    def _parse_modification_command(
        self, command: ParsedCommand
    ) -> dict[str, Any] | None:
        """Parse the modification command to extract type and details."""
        action = command.action.lower()
        subject = command.subject.lower() if command.subject else ""
        target = command.target.lower() if command.target else ""

        # Determine modification type based on action verb
        modification_type = None
        for mod_type, mod_info in self.modification_types.items():
            if action in mod_info["verbs"]:
                modification_type = mod_type
                break

        if not modification_type:
            return None

        # Handle special patterns for writing commands
        if modification_type == "write":
            # Extract text to write and target surface
            text_to_write, target_surface = self._parse_writing_command(command)
            return {
                "type": modification_type,
                "text_content": text_to_write,
                "target_object": target_surface,
                "required_tools": self.modification_types[modification_type][
                    "required_tools"
                ],
                "action_verb": action,
            }

        # Handle other modification types
        return {
            "type": modification_type,
            "target_object": target or subject,
            "required_tools": self.modification_types[modification_type][
                "required_tools"
            ],
            "action_verb": action,
            "details": {"subject": subject, "target": target},
        }

    def _parse_writing_command(self, command: ParsedCommand) -> tuple[str, str]:
        """Parse writing command to extract text and target surface."""
        # Handle patterns like:
        # "write your name on the letter" -> text="your name", target="letter"
        # "write 'Hello World' on wall" -> text="Hello World", target="wall"

        full_command = (
            f"{command.action} {command.subject or ''} {command.target or ''}".strip()
        )

        # Look for "on" keyword to separate text from target
        if " on " in full_command:
            parts = full_command.split(" on ", 1)
            if len(parts) == 2:
                text_part = parts[0].replace("write", "").strip()
                target_part = parts[1].strip()

                # Clean up "the" from target
                target_part = target_part.replace("the ", "").strip()

                return text_part, target_part

        # Fallback: assume subject is text, target is surface
        text_content = command.subject or "something"
        target_surface = command.target or "surface"

        return text_content, target_surface

    async def _validate_modification(
        self,
        modification_info: dict[str, Any],
        player_state: "PlayerState",
        current_location: "Location",
    ) -> ActionResult:
        """Validate that the modification can be performed."""
        modification_type = modification_info["type"]
        target_object = modification_info["target_object"]
        required_tools = modification_info["required_tools"]

        # Check if player has required tools
        if required_tools:
            available_tools = await self._get_available_tools(
                player_state, required_tools
            )
            if not available_tools:
                tool_list = ", ".join(required_tools[:3])
                return ActionResult(
                    success=False,
                    feedback_message=f"You need a tool to {modification_info['action_verb']}. "
                    f"Try finding: {tool_list}",
                )

        # Check if target object exists in location
        target_exists = await self._find_target_object(target_object, current_location)
        if not target_exists:
            return ActionResult(
                success=False,
                feedback_message=f"I don't see '{target_object}' here to {modification_info['action_verb']}.",
            )

        # Check if modification type is valid for this target
        if modification_type != "combine":  # Combine is special case
            mod_type_info = self.modification_types.get(modification_type, {})
            valid_targets = mod_type_info.get("valid_targets", [])
            if valid_targets and not self._is_valid_target(
                target_object, valid_targets
            ):
                return ActionResult(
                    success=False,
                    feedback_message=f"You can't {modification_info['action_verb']} the {target_object}.",
                )

        return ActionResult(success=True)

    async def _get_available_tools(
        self, player_state: "PlayerState", required_tools: list[str]
    ) -> list[str]:
        """Get tools from inventory that match requirements."""
        available_tools = []
        inventory = getattr(player_state, "inventory", [])

        for item in inventory:
            item_name = getattr(item, "name", "").lower()
            for tool in required_tools:
                if tool in item_name:
                    available_tools.append(item_name)
                    break

        return available_tools

    async def _find_target_object(self, target_name: str, location: "Location") -> bool:
        """Check if target object exists in location."""
        # Check location description for mentioned objects
        location_desc = getattr(location, "description", "").lower()
        if target_name in location_desc:
            return True

        # Check location objects
        objects = getattr(location, "objects", [])
        for obj in objects:
            obj_name = getattr(obj, "name", "").lower()
            if target_name in obj_name or obj_name in target_name:
                return True

        return False

    def _is_valid_target(self, target_name: str, valid_targets: list[str]) -> bool:
        """Check if target is valid for this modification type."""
        if "any" in valid_targets:
            return True

        for valid_target in valid_targets:
            if valid_target in target_name or target_name in valid_target:
                return True

        return False

    async def _execute_modification(
        self,
        modification_info: dict[str, Any],
        player_state: "PlayerState",
        current_location: "Location",
    ) -> ActionResult:
        """Execute the object modification."""
        modification_type = modification_info["type"]

        if modification_type == "write":
            return await self._execute_writing_modification(
                modification_info, player_state, current_location
            )
        elif modification_type == "open":
            return await self._execute_opening_modification(
                modification_info, player_state, current_location
            )
        elif modification_type == "turn":
            return await self._execute_turning_modification(
                modification_info, player_state, current_location
            )
        else:
            return await self._execute_generic_modification(
                modification_info, player_state, current_location
            )

    async def _execute_writing_modification(
        self,
        modification_info: dict[str, Any],
        player_state: "PlayerState",
        current_location: "Location",
    ) -> ActionResult:
        """Execute writing/inscription modification."""
        text_content = modification_info["text_content"]
        target_object = modification_info["target_object"]

        # Store the modification in object state
        modification_record = {
            "type": "inscription",
            "text": text_content,
            "author": getattr(player_state, "player_name", "unknown"),
            "timestamp": "recent",
            "location": getattr(current_location, "location_id", "unknown"),
        }

        # In a real implementation, this would be persisted to database
        success_message = f"You write '{text_content}' on the {target_object}."

        # Add atmospheric details
        if "pen" in str(modification_info.get("required_tools", [])):
            success_message += " The ink flows smoothly across the surface."
        elif "chalk" in str(modification_info.get("required_tools", [])):
            success_message += " The chalk leaves clear, white marks."

        return ActionResult(
            success=True,
            feedback_message=success_message,
            object_changes=[
                {
                    "modification_type": "write",
                    "target_object": target_object,
                    "modification_record": modification_record,
                }
            ],
        )

    async def _execute_opening_modification(
        self,
        modification_info: dict[str, Any],
        player_state: "PlayerState",
        current_location: "Location",
    ) -> ActionResult:
        """Execute opening/unlocking modification."""
        target_object = modification_info["target_object"]

        # Generate contents or reveal what's inside
        if "book" in target_object:
            success_message = f"You open the {target_object}. The pages reveal detailed text and diagrams."
        elif "chest" in target_object or "box" in target_object:
            success_message = (
                f"You open the {target_object}. Inside you see various items."
            )
        else:
            success_message = f"You open the {target_object}."

        return ActionResult(
            success=True,
            feedback_message=success_message,
            object_changes=[
                {
                    "modification_type": "open",
                    "target_object": target_object,
                    "state_change": "opened",
                }
            ],
        )

    async def _execute_turning_modification(
        self,
        modification_info: dict[str, Any],
        player_state: "PlayerState",
        current_location: "Location",
    ) -> ActionResult:
        """Execute turning/rotating modification."""
        target_object = modification_info["target_object"]

        if "page" in target_object:
            success_message = (
                "You turn the page. New content is revealed on the next page."
            )
        elif "dial" in target_object or "knob" in target_object:
            success_message = (
                f"You turn the {target_object}. It clicks into a new position."
            )
        else:
            success_message = f"You turn the {target_object}."

        return ActionResult(
            success=True,
            feedback_message=success_message,
            object_changes=[
                {
                    "modification_type": "turn",
                    "target_object": target_object,
                    "state_change": "turned",
                }
            ],
        )

    async def _execute_generic_modification(
        self,
        modification_info: dict[str, Any],
        player_state: "PlayerState",
        current_location: "Location",
    ) -> ActionResult:
        """Execute generic modification for other types."""
        modification_type = modification_info["type"]
        target_object = modification_info["target_object"]
        action_verb = modification_info["action_verb"]

        success_message = f"You {action_verb} the {target_object}."

        # Add type-specific flavor
        if modification_type == "repair":
            success_message += " It looks much better now."
        elif modification_type == "clean":
            success_message += " It's now spotless and shining."
        elif modification_type == "break":
            success_message += " It breaks with a satisfying crash."

        return ActionResult(
            success=True,
            feedback_message=success_message,
            object_changes=[
                {
                    "modification_type": modification_type,
                    "target_object": target_object,
                    "state_change": f"{modification_type}ed",
                }
            ],
        )

    def can_handle_command(self, command: ParsedCommand) -> bool:
        """Check if this handler can process the given command."""
        action = command.action.lower()

        # Check if action matches any modification verbs
        for mod_type, mod_info in self.modification_types.items():
            if action in mod_info["verbs"]:
                return True

        return False
