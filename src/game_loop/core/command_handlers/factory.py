"""
Factory for creating command handlers based on command type.
"""

from collections.abc import Callable

from rich.console import Console

from game_loop.core.command_handlers.base_handler import CommandHandler
from game_loop.core.command_handlers.use_handler.use_handler import UseHandler
from game_loop.core.input_processor import CommandType
from game_loop.state.manager import GameStateManager


class CommandHandlerFactory:
    """
    Factory for creating command handlers.

    This class implements the Factory pattern to create appropriate command handlers
    for different command types, allowing for a clean separation of handling logic.
    """

    def __init__(self, console: Console, state_manager: GameStateManager):
        """
        Initialize the command handler factory.

        Args:
            console: Rich console for output
            state_manager: Game state manager for accessing and updating game state
        """
        self.console = console
        self.state_manager = state_manager
        self._handlers: dict[CommandType, Callable[[], CommandHandler]] = {}
        self._register_handlers()

    def _register_handlers(self) -> None:
        """Register all available command handlers."""
        # TODO: Add more handlers as they are implemented
        self._handlers[CommandType.USE] = self._create_use_handler

        # Additional handlers would be registered here:
        # self._handlers[CommandType.EXAMINE] = self._create_examine_handler
        # self._handlers[CommandType.TAKE] = self._create_take_handler
        # etc.

    def get_handler(self, command_type: CommandType) -> CommandHandler:
        """
        Get the appropriate handler for a command type.

        Args:
            command_type: The type of command to handle

        Returns:
            A command handler instance for the specified command type
        """
        handler_factory = self._handlers.get(command_type)

        if handler_factory:
            return handler_factory()

        # If no specific handler is registered, return a default handler
        # In a full implementation, create a DefaultHandler class
        return self._create_use_handler()  # Temporary fallback

    def _create_use_handler(self) -> CommandHandler:
        """Create and return a new use handler instance."""
        return UseHandler(self.console, self.state_manager)
