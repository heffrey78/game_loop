"""
Input processing for Game Loop.
Handles parsing and validation of player input commands.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from rich.console import Console

if TYPE_CHECKING:
    from ..llm.nlp_processor import NLPProcessor
    from ..state.manager import GameStateManager


class CommandType(Enum):
    """Enumeration of possible command types."""

    MOVEMENT = auto()
    LOOK = auto()
    INVENTORY = auto()
    TAKE = auto()
    DROP = auto()
    USE = auto()
    EXAMINE = auto()
    TALK = auto()
    HELP = auto()
    QUIT = auto()
    UNKNOWN = auto()


@dataclass
class ParsedCommand:
    """Represents a parsed command from user input."""

    command_type: CommandType
    action: str
    subject: str | None = None
    target: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.parameters is None:
            self.parameters = {}


class InputProcessor:
    """
    Processes and validates user input, converting it into structured commands.
    Enhanced with GameStateManager integration for context-aware processing.
    """

    def __init__(
        self,
        console: Console | None = None,
        game_state_manager: "GameStateManager | None" = None,
        nlp_processor: "NLPProcessor | None" = None,
    ):
        """
        Initialize the input processor.

        Args:
            console: Console for output, will create a new one if not provided
            game_state_manager: GameStateManager for retrieving game context
            nlp_processor: NLPProcessor for enhanced natural language processing
        """
        self.console = console if console else Console()
        self.game_state_manager = game_state_manager
        self.nlp_processor = nlp_processor
        self._setup_command_patterns()

    def _setup_command_patterns(self) -> None:
        """Set up the command patterns for basic command recognition."""
        # Movement commands
        self.movement_commands = {
            "north": ["north", "n", "go north"],
            "south": ["south", "s", "go south"],
            "east": ["east", "e", "go east"],
            "west": ["west", "w", "go west"],
            "up": ["up", "u", "go up"],
            "down": ["down", "d", "go down"],
            # Note: "exit" is handled specially - it can be movement or quit depending on context
        }

        # Flattened list of all movement command variants
        self.all_movement_commands = []
        for cmd_list in self.movement_commands.values():
            self.all_movement_commands.extend(cmd_list)

        # Look commands
        self.look_commands = ["look", "l", "look around", "examine surroundings"]

        # Inventory commands
        self.inventory_commands = ["inventory", "i", "items", "check inventory"]

        # Help commands
        self.help_commands = ["help", "h", "?", "commands"]

        # Quit commands
        self.quit_commands = ["quit", "exit", "q", "bye"]

        # Action command prefixes
        self.take_prefixes = ["take", "get", "pick up", "grab"]
        self.drop_prefixes = ["drop", "discard", "throw"]
        self.use_prefixes = ["use", "apply", "activate"]
        self.examine_prefixes = ["examine", "inspect", "look at", "check"]
        self.talk_prefixes = ["talk", "speak", "chat", "converse"]

        # Complex action prefixes (for multi-object interactions)
        self.put_prefixes = ["put", "place", "insert", "stick", "store"]

    def process_input(
        self, user_input: str, game_context: dict[str, Any] | None = None
    ) -> ParsedCommand:
        """
        Process user input and return a structured command.

        This is a synchronous wrapper around the async version for
        compatibility with existing tests.

        Args:
            user_input: The raw user input string
            game_context: Optional context information from the game

        Returns:
            A ParsedCommand object representing the processed input
        """
        # Normalize input
        normalized_input = self._normalize_input(user_input)
        if not normalized_input:
            return ParsedCommand(
                command_type=CommandType.UNKNOWN, action="unknown", subject=None
            )

        # Try to match to known command patterns
        return self._match_command_pattern(normalized_input, game_context)

    async def process(
        self, user_input: str, context: dict[str, Any] | None = None
    ) -> ParsedCommand:
        """
        Enhanced process method with context-aware processing.

        This is the main entry point for enhanced input processing that
        integrates with GameStateManager and NLPProcessor.

        Args:
            user_input: The raw user input string
            context: Optional external context (overrides auto-retrieved context)

        Returns:
            A ParsedCommand object representing the processed input
        """
        # Get current game context if not provided
        if context is None:
            context = await self._get_current_context()

        # Use enhanced context-aware processing if available
        if self.nlp_processor and context:
            return await self._process_with_context(user_input, context)

        # Fall back to standard processing for backward compatibility
        return self.process_input(user_input, context)

    async def process_input_async(
        self, user_input: str, game_context: dict[str, Any] | None = None
    ) -> ParsedCommand:
        """
        Process user input and return a structured command asynchronously.

        This method provides backward compatibility with existing tests
        and simply delegates to the synchronous process_input method.

        Args:
            user_input: The raw user input string
            game_context: Optional context information from the game

        Returns:
            A ParsedCommand object representing the processed input
        """
        return self.process_input(user_input, game_context)

    async def _get_current_context(self) -> dict[str, Any]:
        """
        Retrieve current game context from GameStateManager.

        Returns:
            Dictionary containing current game state context
        """
        context: dict[str, Any] = {}

        if not self.game_state_manager:
            return context

        try:
            # Get current location details
            location_details = (
                await self.game_state_manager.get_current_location_details()
            )
            if location_details:
                context["current_location"] = {
                    "id": str(location_details.location_id),
                    "name": location_details.name,
                    "description": location_details.description,
                }

                # Add connections if available
                if (
                    hasattr(location_details, "connections")
                    and location_details.connections
                ):
                    context["connections"] = dict(location_details.connections)

                # Add objects if available
                if hasattr(location_details, "objects") and location_details.objects:
                    context["visible_objects"] = [
                        {"name": obj.name, "description": obj.description}
                        for obj in location_details.objects.values()
                    ]

                # Add NPCs if available
                if hasattr(location_details, "npcs") and location_details.npcs:
                    context["npcs"] = [
                        {"name": npc.name, "description": npc.description}
                        for npc in location_details.npcs.values()
                    ]

            # Get player state
            player_state, _ = self.game_state_manager.get_current_state()
            if player_state:
                context["player"] = {
                    "name": player_state.name,
                    "current_location_id": (
                        str(player_state.current_location_id)
                        if player_state.current_location_id
                        else None
                    ),
                }

                # Add inventory if available
                if hasattr(player_state, "inventory") and player_state.inventory:
                    # Handle both dict and list inventory formats
                    if hasattr(player_state.inventory, "values"):
                        inventory_items = player_state.inventory.values()
                    else:
                        inventory_items = player_state.inventory

                    context["inventory"] = [
                        {
                            "name": item.name,
                            "description": getattr(item, "description", ""),
                        }
                        for item in inventory_items
                    ]

        except Exception:
            # Log error but don't fail processing
            # Could use logger here if available
            pass

        return context

    async def _process_with_context(
        self, user_input: str, context: dict[str, Any]
    ) -> ParsedCommand:
        """
        Process input using enhanced context and NLP processing.

        Args:
            user_input: The raw user input string
            context: Game context for enhanced processing

        Returns:
            A ParsedCommand object representing the processed input
        """
        try:
            # Use NLP processor for context-aware processing
            if self.nlp_processor:
                parsed_command = await self.nlp_processor.process_input(
                    user_input, context
                )

                # Validate and enhance the result if needed
                if (
                    parsed_command
                    and parsed_command.command_type != CommandType.UNKNOWN
                ):
                    return parsed_command

        except Exception:
            # Log error but don't fail processing
            # Could use logger here if available
            pass

        # Fall back to pattern matching if NLP processing fails
        return self.process_input(user_input, context)

    def _normalize_input(self, user_input: str) -> str:
        """
        Normalize user input by converting to lowercase and removing extra whitespace.

        Args:
            user_input: The raw user input string

        Returns:
            Normalized input string
        """
        return " ".join(user_input.lower().strip().split())

    def _match_command_pattern(
        self, input_text: str, game_context: dict[str, Any] | None = None
    ) -> ParsedCommand:
        """
        Match the input text against known command patterns.

        Args:
            input_text: The normalized input text
            game_context: Optional game context for contextual decisions

        Returns:
            A ParsedCommand object representing the processed input
        """
        # Check for movement commands
        if input_text in self.all_movement_commands:
            direction = self._get_direction_from_input(input_text)
            return ParsedCommand(
                command_type=CommandType.MOVEMENT, action="go", subject=direction
            )

        # Check for look commands
        if input_text in self.look_commands:
            return ParsedCommand(command_type=CommandType.LOOK, action="look")

        # Check for inventory commands
        if input_text in self.inventory_commands:
            return ParsedCommand(command_type=CommandType.INVENTORY, action="inventory")

        # Check for help commands
        if input_text in self.help_commands:
            return ParsedCommand(command_type=CommandType.HELP, action="help")

        # Check for quit commands - but skip "exit" for now as it might be a movement
        if input_text in self.quit_commands and input_text != "exit":
            return ParsedCommand(command_type=CommandType.QUIT, action="quit")

        # Check for more complex commands
        complex_command = self._parse_complex_command(input_text)
        if complex_command:
            return complex_command

        # Special handling for "exit" - check if it's a movement direction
        if input_text == "exit":
            # Check if "exit" is a valid movement direction in current context
            if self._is_exit_a_valid_direction(game_context):
                return ParsedCommand(
                    command_type=CommandType.MOVEMENT, action="go", subject="exit"
                )
            else:
                # Otherwise treat it as a quit command
                return ParsedCommand(command_type=CommandType.QUIT, action="quit")

        # Default to unknown command
        return ParsedCommand(
            command_type=CommandType.UNKNOWN, action="unknown", subject=input_text
        )

    def _get_direction_from_input(self, input_text: str) -> str:
        """
        Extract the actual direction from a movement command.

        Args:
            input_text: The normalized input text

        Returns:
            The direction as a string (north, south, etc.)
        """
        for direction, command_variants in self.movement_commands.items():
            if input_text in command_variants:
                return direction

        # Default fallback - should not reach here if input validation is working
        return "unknown"

    def _parse_complex_command(self, input_text: str) -> ParsedCommand | None:
        """
        Parse more complex commands like 'take key' or 'examine book'.

        Args:
            input_text: The normalized input text

        Returns:
            A ParsedCommand object if a complex command is recognized, None otherwise
        """
        # Special case for "pick up" command pattern
        if input_text.startswith("pick up "):
            # Extract the object (everything after "pick up ")
            object_name = input_text[len("pick up ") :].strip()
            return ParsedCommand(
                command_type=CommandType.TAKE, action="take", subject=object_name
            )

        # Check for take commands
        take_command = self._check_action_with_object(input_text, self.take_prefixes)
        if take_command:
            return ParsedCommand(
                command_type=CommandType.TAKE, action="take", subject=take_command
            )

        # Check for drop commands
        drop_command = self._check_action_with_object(input_text, self.drop_prefixes)
        if drop_command:
            return ParsedCommand(
                command_type=CommandType.DROP, action="drop", subject=drop_command
            )

        # Check for put commands (e.g., "put sword in pouch")
        put_command, put_target = self._check_put_command(input_text)
        if put_command and put_target:
            return ParsedCommand(
                command_type=CommandType.USE,
                action="put",
                subject=put_command,
                target=put_target,
            )

        # Check for use commands
        use_command, target = self._check_use_command(input_text)
        if use_command:
            return ParsedCommand(
                command_type=CommandType.USE,
                action="use",
                subject=use_command,
                target=target,
            )

        # Check for examine commands
        examine_command = self._check_action_with_object(
            input_text, self.examine_prefixes
        )
        if examine_command:
            return ParsedCommand(
                command_type=CommandType.EXAMINE,
                action="examine",
                subject=examine_command,
            )

        # Check for talk commands
        talk_command = self._check_action_with_object(input_text, self.talk_prefixes)
        if talk_command:
            return ParsedCommand(
                command_type=CommandType.TALK, action="talk", subject=talk_command
            )

        return None

    def _check_action_with_object(
        self, input_text: str, action_prefixes: list[str]
    ) -> str | None:
        """
        Check if the input matches an action with an object (e.g., "take key").

        Args:
            input_text: The normalized input text
            action_prefixes: List of prefixes that match this action type

        Returns:
            The object of the action if found, None otherwise
        """
        for prefix in action_prefixes:
            if input_text.startswith(prefix + " "):
                # Extract the object (everything after the prefix and a space)
                return input_text[len(prefix) + 1 :].strip()
        return None

    def _check_use_command(self, input_text: str) -> tuple[str | None, str | None]:
        """
        Check for use commands which may include a target (e.g., "use key on door").

        Args:
            input_text: The normalized input text

        Returns:
            A tuple of (object_to_use, target_object) or (None, None) if not a use
            command
        """
        for prefix in self.use_prefixes:
            if input_text.startswith(prefix + " "):
                remaining = input_text[len(prefix) + 1 :].strip()

                # Check for "use X on Y" pattern
                parts = remaining.split(" on ")
                if len(parts) == 2:
                    return parts[0].strip(), parts[1].strip()

                # Check for "use X with Y" pattern
                parts = remaining.split(" with ")
                if len(parts) == 2:
                    return parts[0].strip(), parts[1].strip()

                # Simple "use X" pattern
                return remaining, None

        return None, None

    def _check_put_command(self, input_text: str) -> tuple[str | None, str | None]:
        """
        Check for put commands which may include a target (e.g., "put sword in pouch").

        Args:
            input_text: The normalized input text

        Returns:
            A tuple of (object_to_put, target_object) or (None, None) if not a put
            command
        """
        for prefix in self.put_prefixes:
            if input_text.startswith(prefix + " "):
                remaining = input_text[len(prefix) + 1 :].strip()

                # Check for "put X in Y" pattern
                parts = remaining.split(" in ")
                if len(parts) == 2:
                    return parts[0].strip(), parts[1].strip()

                # Check for "put X into Y" pattern
                parts = remaining.split(" into ")
                if len(parts) == 2:
                    return parts[0].strip(), parts[1].strip()

                # Check for "put X on Y" pattern
                parts = remaining.split(" on ")
                if len(parts) == 2:
                    return parts[0].strip(), parts[1].strip()

                # Check for "put X onto Y" pattern
                parts = remaining.split(" onto ")
                if len(parts) == 2:
                    return parts[0].strip(), parts[1].strip()

                # Simple "put X" pattern (incomplete command)
                return remaining, None

        return None, None

    def _is_exit_a_valid_direction(
        self, game_context: dict[str, Any] | None = None
    ) -> bool:
        """
        Check if 'exit' is a valid movement direction in the current context.

        Args:
            game_context: Optional game context to check

        Returns:
            True if 'exit' is a valid direction, False otherwise
        """
        # If context is provided, use it
        if game_context and "connections" in game_context:
            return "exit" in game_context["connections"]

        # If we don't have a game state manager, we can't check context
        if not self.game_state_manager:
            return False

        try:
            # Get current game context synchronously
            import asyncio

            loop = asyncio.new_event_loop()
            try:
                context = loop.run_until_complete(self._get_current_context())
            finally:
                loop.close()

            # Check if we have connections in the context
            if "connections" in context and context["connections"]:
                # Check if "exit" is one of the available directions
                return "exit" in context["connections"]

        except Exception:
            # If we can't get context, assume exit is not a direction
            pass

        return False

    def generate_command_suggestions(self, partial_input: str) -> list[str]:
        """
        Generate command suggestions based on partial input.

        Args:
            partial_input: Partial command input from the user

        Returns:
            List of suggested commands that match the partial input
        """
        normalized = self._normalize_input(partial_input)
        if not normalized:
            return []

        # Compile all known commands
        all_commands = []
        all_commands.extend(self.all_movement_commands)
        all_commands.extend(self.look_commands)
        all_commands.extend(self.inventory_commands)
        all_commands.extend(self.help_commands)
        all_commands.extend(self.quit_commands)

        # Add action prefixes with placeholder objects
        for prefix in self.take_prefixes:
            all_commands.append(f"{prefix} [object]")
        for prefix in self.drop_prefixes:
            all_commands.append(f"{prefix} [object]")
        for prefix in self.use_prefixes:
            all_commands.append(f"{prefix} [object]")
        for prefix in self.examine_prefixes:
            all_commands.append(f"{prefix} [object]")
        for prefix in self.talk_prefixes:
            all_commands.append(f"{prefix} [character]")
        # Add complex command patterns
        for prefix in self.put_prefixes:
            all_commands.append(f"{prefix} [object] in [container]")
            all_commands.append(f"{prefix} [object] on [surface]")

        # Filter commands that start with the partial input
        return [cmd for cmd in all_commands if cmd.startswith(normalized)]

    def format_error_message(self, parsed_command: ParsedCommand) -> str:
        """
        Format an appropriate error message for unrecognized commands.

        Args:
            parsed_command: The parsed command that wasn't recognized

        Returns:
            A user-friendly error message
        """
        if parsed_command.command_type == CommandType.UNKNOWN:
            return (
                f"I don't understand '{parsed_command.subject}'.\n"
                "            Type 'help' for a list of commands."
            )
        return "I'm not sure what you mean. Type 'help' for a list of commands."
