"""
Enhanced Input Processor for Game Loop.
Combines pattern matching and NLP approaches for more natural language understanding.
"""

import logging
from typing import TYPE_CHECKING, Any

from rich.console import Console

from game_loop.config.manager import ConfigManager
from game_loop.core.command_mapper import CommandMapper
from game_loop.core.input_processor import CommandType, InputProcessor, ParsedCommand
from game_loop.llm.conversation_context import ConversationContext
from game_loop.llm.nlp_processor import NLPProcessor

if TYPE_CHECKING:
    from game_loop.state.manager import GameStateManager

logger = logging.getLogger(__name__)


class EnhancedInputProcessor(InputProcessor):
    """
    Input processor with natural language understanding capabilities.
    Combines pattern matching and NLP approaches.
    """

    def __init__(
        self,
        config_manager: ConfigManager | None = None,
        console: Console | None = None,
        use_nlp: bool = True,
        game_state_manager: "GameStateManager | None" = None,
    ):
        """
        Initialize with both pattern matching and NLP capabilities.

        Args:
            config_manager: Configuration manager for LLM settings
            console: Console for output
            use_nlp: Whether to use NLP processing (fall back to pattern)
            game_state_manager: GameStateManager for context
        """
        self.config_manager = config_manager or ConfigManager()
        self.nlp_processor = NLPProcessor(config_manager=self.config_manager)

        # Initialize parent class with enhanced capabilities
        super().__init__(
            console=console,
            game_state_manager=game_state_manager,
            nlp_processor=self.nlp_processor,
        )

        self.use_nlp = use_nlp
        self.command_mapper = CommandMapper()
        self.conversation_context = ConversationContext()

    def process_input(
        self, user_input: str, game_context: dict[str, Any] | None = None
    ) -> ParsedCommand:
        """
        Process user input with pattern matching (synchronous version).
        This overrides the parent method to ensure type compatibility.

        Args:
            user_input: Raw user input text
            game_context: Current game state for context

        Returns:
            Parsed command representing user intent
        """
        import asyncio

        # Use the synchronous method to get a result from the async method
        try:
            # For environments that support it, use the get_event_loop approach
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new loop for this call if the main loop is running
                new_loop = asyncio.new_event_loop()
                try:
                    return new_loop.run_until_complete(
                        self.process_input_async(user_input, game_context)
                    )
                finally:
                    new_loop.close()
            else:
                return loop.run_until_complete(
                    self.process_input_async(user_input, game_context)
                )
        except RuntimeError:
            # If no event loop is available, create one
            return asyncio.run(self.process_input_async(user_input, game_context))

    async def process_input_async(
        self, user_input: str, game_context: dict[str, Any] | None = None
    ) -> ParsedCommand:
        """
        Process user input with NLP, falling back to pattern matching (async version).

        Args:
            user_input: Raw user input text
            game_context: Current game state for context

        Returns:
            Parsed command representing user intent
        """
        # Initialize game_context if None
        if game_context is None:
            game_context = {}

        # Normalize input
        normalized_input = self._normalize_input(user_input)
        if not normalized_input:
            return ParsedCommand(
                command_type=CommandType.UNKNOWN, action="unknown", subject=None
            )

        # Before anything else, check for specific complex patterns like "put X in Y"
        complex_command = self._check_complex_command_patterns(normalized_input)
        if complex_command:
            return complex_command

        # Try pattern matching for simple commands
        pattern_command, is_simple_command = self._try_pattern_matching(
            normalized_input, game_context
        )

        # If pattern matching yields a definitive result for a simple command, use it
        if is_simple_command and pattern_command.command_type != CommandType.UNKNOWN:
            return pattern_command

        # If NLP is disabled, return pattern matching result
        if not self.use_nlp:
            return pattern_command

        try:
            # Process with NLP if available
            if self.nlp_processor:
                nlp_command = await self.nlp_processor.process_input(
                    normalized_input, game_context
                )

            # Map NLP intent to command
            if nlp_command.command_type != CommandType.UNKNOWN:
                mapped_command = self.command_mapper.map_intent_to_command(
                    {
                        "command_type": nlp_command.command_type.name,
                        "action": nlp_command.action,
                        "subject": nlp_command.subject,
                        "target": nlp_command.target,
                        "confidence": nlp_command.parameters.get("confidence", 0.0),
                    }
                )

                # Only use NLP result if confidence is reasonable
                if mapped_command.parameters.get("confidence", 0.0) > 0.6:
                    return mapped_command

                # If confidence is low but higher than threshold, try disambiguation
                if mapped_command.parameters.get("confidence", 0.0) > 0.3:
                    # Use both pattern matching and NLP results for disambiguation
                    if pattern_command.command_type != CommandType.UNKNOWN:
                        disambiguated = await self._disambiguate_commands(
                            normalized_input,
                            [pattern_command, mapped_command],
                            game_context or {},
                        )
                        return disambiguated

            # Fall back to pattern matching if NLP fails or has low confidence
            return pattern_command

        except Exception as e:
            logger.error(f"Error in NLP processing: {e}")
            # Fall back to pattern matching
            return pattern_command

    def _try_pattern_matching(
        self, normalized_input: str, game_context: dict[str, Any] | None = None
    ) -> tuple[ParsedCommand, bool]:
        """
        Try to match input using pattern matching approach.

        Args:
            normalized_input: Normalized input text
            game_context: Optional game context for pattern matching

        Returns:
            Tuple of (parsed command, whether it's a simple command)
        """
        # Detect if this is a simple command (e.g., single word commands like "look")
        is_simple_command = len(normalized_input.split()) <= 2

        # Use the parent class's pattern matching with context
        command = super()._match_command_pattern(normalized_input, game_context)

        return command, is_simple_command

    async def _disambiguate_commands(
        self,
        user_input: str,
        possible_commands: list[ParsedCommand],
        game_context: dict[str, Any],
    ) -> ParsedCommand:
        """
        Resolve ambiguity between multiple possible command interpretations.

        Args:
            user_input: Original user input
            possible_commands: List of possible command interpretations
            game_context: Game context for disambiguation

        Returns:
            The most likely command interpretation
        """
        # Convert ParsedCommands to dictionary format for disambiguation
        interpretations = []
        for cmd in possible_commands:
            interpretations.append(
                {
                    "command_type": cmd.command_type.name,
                    "action": cmd.action,
                    "subject": cmd.subject,
                    "target": cmd.target,
                    "confidence": cmd.parameters.get("confidence", 0.5),
                }
            )

        # Use NLP processor to disambiguate if available
        if not self.nlp_processor:
            # Fall back to first command if no NLP processor
            return (
                possible_commands[0]
                if possible_commands
                else ParsedCommand(
                    command_type=CommandType.UNKNOWN, action="unknown", subject=None
                )
            )

        context_str = self.nlp_processor._format_context(game_context)
        result = await self.nlp_processor.disambiguate_input(
            user_input, interpretations, context_str
        )

        # Convert back to ParsedCommand
        selected_index = result.get("selected_index", 0)
        if 0 <= selected_index < len(possible_commands):
            selected = possible_commands[selected_index]
            selected.parameters["confidence"] = result.get("confidence", 0.5)
            if isinstance(selected, ParsedCommand):
                return selected
            else:
                raise TypeError("Selected command is not of type ParsedCommand")

        # Default to first command if disambiguation fails
        return possible_commands[0]

    async def update_conversation_context(
        self,
        user_input: str,
        system_response: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Update conversation context with a new exchange.

        Args:
            user_input: User's input text
            system_response: System's response text
            metadata: Optional metadata for this exchange
        """
        self.conversation_context.add_exchange(
            user_input=user_input, system_response=system_response, metadata=metadata
        )

    def _check_complex_command_patterns(
        self, normalized_input: str
    ) -> ParsedCommand | None:
        """
        Check input against specific complex command patterns.

        Args:
            normalized_input: Normalized user input

        Returns:
            ParsedCommand if a complex pattern is matched, None otherwise
        """
        import re

        # Check for "pick up X" pattern - add this before other patterns
        pick_up_match = re.search(
            r"pick\s+up\s+(?:the\s+)?([a-zA-Z0-9_\s]+)",
            normalized_input,
        )
        if pick_up_match:
            subject = pick_up_match.group(1).strip()
            return ParsedCommand(
                command_type=CommandType.TAKE,
                action="take",
                subject=subject,
                target=None,
                parameters={"confidence": 0.95},
            )

        # Check for "put X in/into Y" pattern
        put_in_match = re.search(
            r"put\s+(?:the\s+)?([a-zA-Z0-9_\s]+)\s+(?:in|into)\s+(?:the\s+)?([a-zA-Z0-9_\s]+)",
            normalized_input,
        )
        if put_in_match:
            subject = put_in_match.group(1).strip()
            target = put_in_match.group(2).strip()
            return ParsedCommand(
                command_type=CommandType.USE,
                action="put",
                subject=subject,
                target=target,
                parameters={"confidence": 0.95},
            )

        # Check for "place X on/onto Y" pattern
        place_on_match = re.search(
            r"place\s+(?:the\s+)?([a-zA-Z0-9_\s]+)\s+(?:on|onto)\s+(?:the\s+)?([a-zA-Z0-9_\s]+)",
            normalized_input,
        )
        if place_on_match:
            subject = place_on_match.group(1).strip()
            target = place_on_match.group(2).strip()
            return ParsedCommand(
                command_type=CommandType.USE,
                action="place",
                subject=subject,
                target=target,
                parameters={"confidence": 0.95},
            )

        # Check for "insert X into Y" pattern
        insert_into_match = re.search(
            r"insert\s+(?:the\s+)?([a-zA-Z0-9_\s]+)\s+(?:in|into)\s+(?:the\s+)?([a-zA-Z0-9_\s]+)",
            normalized_input,
        )
        if insert_into_match:
            subject = insert_into_match.group(1).strip()
            target = insert_into_match.group(2).strip()
            return ParsedCommand(
                command_type=CommandType.USE,
                action="insert",
                subject=subject,
                target=target,
                parameters={"confidence": 0.95},
            )

        # Check for "store X in Y" pattern
        store_in_match = re.search(
            r"store\s+(?:the\s+)?([a-zA-Z0-9_\s]+)\s+(?:in|inside)\s+(?:the\s+)?([a-zA-Z0-9_\s]+)",
            normalized_input,
        )
        if store_in_match:
            subject = store_in_match.group(1).strip()
            target = store_in_match.group(2).strip()
            return ParsedCommand(
                command_type=CommandType.USE,
                action="store",
                subject=subject,
                target=target,
                parameters={"confidence": 0.95},
            )

        # No complex pattern matched
        return None
