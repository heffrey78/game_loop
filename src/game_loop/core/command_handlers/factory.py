"""
Factory for creating command handlers based on command type.
"""

from collections.abc import Callable
from typing import Any

from rich.console import Console

from game_loop.config.manager import ConfigManager
from game_loop.core.command_handlers.base_handler import CommandHandler
from game_loop.core.command_handlers.enhanced_conversation_handler import (
    EnhancedConversationCommandHandler,
)
from game_loop.core.command_handlers.enhanced_movement_handler import (
    EnhancedMovementCommandHandler,
)
from game_loop.core.command_handlers.inventory_handler import InventoryCommandHandler
from game_loop.core.command_handlers.object_modification_handler import (
    ObjectModificationHandler,
)
from game_loop.core.command_handlers.observation_handler import (
    ObservationCommandHandler,
)
from game_loop.core.command_handlers.system_command_processor import (
    SystemCommandProcessor,
)
from game_loop.core.command_handlers.use_handler.use_handler import UseHandler
from game_loop.core.input_processor import CommandType
from game_loop.database.session_factory import DatabaseSessionFactory
from game_loop.llm.ollama.client import OllamaClient
from game_loop.search.semantic_search import SemanticSearchService
from game_loop.state.manager import GameStateManager
from game_loop.state.models import ActionResult

from ..actions.action_classifier import ActionTypeClassifier


class CommandHandlerFactory:
    """
    Factory for creating command handlers.

    This class implements the Factory pattern to create appropriate command handlers
    for different command types, allowing for a clean separation of handling logic.
    """

    def __init__(
        self,
        console: Console,
        state_manager: GameStateManager,
        session_factory: DatabaseSessionFactory | None = None,
        config_manager: ConfigManager | None = None,
        llm_client: OllamaClient | None = None,
        semantic_search: SemanticSearchService | None = None,
        action_classifier: ActionTypeClassifier | None = None,
    ):
        """
        Initialize the command handler factory.

        Args:
            console: Rich console for output
            state_manager: Game state manager for accessing and updating game state
            session_factory: Database session factory for system commands
            config_manager: Configuration manager
            llm_client: LLM client for system commands
            semantic_search: Semantic search service
            action_classifier: Action classifier for system command detection
        """
        self.console = console
        self.state_manager = state_manager
        self.session_factory = session_factory
        self.config_manager = config_manager
        self.llm_client = llm_client
        self.semantic_search = semantic_search
        self.action_classifier = action_classifier

        # Initialize system command processor if core dependencies are available
        self.system_processor = None
        if all([session_factory, config_manager, llm_client]):
            self.system_processor = SystemCommandProcessor(
                session_factory,
                state_manager,
                config_manager,
                llm_client,
                semantic_search,  # Can be None
            )

        self._handlers: dict[CommandType, Callable[[], CommandHandler]] = {}
        self._register_handlers()

    def _register_handlers(self) -> None:
        """Register all available command handlers."""
        # Core command handlers
        self._handlers[CommandType.MOVEMENT] = self._create_movement_handler
        self._handlers[CommandType.LOOK] = self._create_observation_handler
        self._handlers[CommandType.EXAMINE] = self._create_observation_handler
        self._handlers[CommandType.INVENTORY] = self._create_inventory_handler
        self._handlers[CommandType.TAKE] = self._create_inventory_handler
        self._handlers[CommandType.DROP] = self._create_inventory_handler
        self._handlers[CommandType.TALK] = self._create_conversation_handler
        self._handlers[CommandType.USE] = self._create_use_handler

        # Object modification handlers
        self._handlers[CommandType.UNKNOWN] = (
            self._create_smart_handler
        )  # Will route to appropriate handler

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

        # If no specific handler is registered, return use handler as fallback
        # This maintains backward compatibility for unhandled command types
        return self._create_use_handler()

    def _create_use_handler(self) -> CommandHandler:
        """Create and return a new use handler instance."""
        return UseHandler(self.console, self.state_manager)

    def _create_movement_handler(self) -> CommandHandler:
        """Create and return a new enhanced movement handler instance."""
        return EnhancedMovementCommandHandler(self.console, self.state_manager)

    def _create_observation_handler(self) -> CommandHandler:
        """Create and return a new observation handler instance."""
        return ObservationCommandHandler(self.console, self.state_manager)

    def _create_inventory_handler(self) -> CommandHandler:
        """Create and return a new inventory handler instance."""
        return InventoryCommandHandler(self.console, self.state_manager)

    def _create_conversation_handler(self) -> CommandHandler:
        """Create and return a new enhanced conversation handler instance."""
        return EnhancedConversationCommandHandler(self.console, self.state_manager)

    def _create_object_modification_handler(self) -> CommandHandler:
        """Create and return a new object modification handler instance."""
        return ObjectModificationHandler(self.console, self.state_manager)

    def _create_smart_handler(self) -> CommandHandler:
        """Create smart handler that can route to appropriate specialized handlers."""
        return ObjectModificationHandler(self.console, self.state_manager)

    async def route_system_command(
        self, text: str, context: dict[str, Any]
    ) -> ActionResult | None:
        """Route system commands to SystemCommandProcessor."""
        try:
            if not self.action_classifier or not self.system_processor:
                return None

            # Check if this is a system command
            system_classification = (
                await self.action_classifier.classify_system_command(text)
            )
            if not system_classification:
                return None

            # Process the system command
            await self.system_processor.initialize()
            return await self.system_processor.process_command(
                system_classification.command_type, system_classification.args, context
            )

        except Exception as e:
            return ActionResult(
                success=False,
                feedback_message=f"Error processing system command: {str(e)}",
            )

    async def handle_command(
        self, text: str, context: dict[str, Any]
    ) -> ActionResult | None:
        """
        Handle any command, checking for system commands first.

        Args:
            text: Input text to process
            context: Current game context

        Returns:
            ActionResult if this was a system command, None otherwise
        """
        # First check if this is a system command
        system_result = await self.route_system_command(text, context)
        if system_result:
            return system_result

        # If not a system command, return None to let normal processing continue
        return None
