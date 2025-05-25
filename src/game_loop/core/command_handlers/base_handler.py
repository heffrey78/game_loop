"""
Base command handler interface for the Game Loop.
"""

from abc import ABC, abstractmethod

from rich.console import Console

from game_loop.core.input_processor import ParsedCommand
from game_loop.state.manager import GameStateManager
from game_loop.state.models import ActionResult, Location, PlayerState, WorldState


class CommandHandler(ABC):
    """
    Base interface for all command handlers.

    This abstract class defines the contract that all command handlers must implement,
    following the strategy pattern.
    """

    def __init__(self, console: Console, state_manager: GameStateManager):
        """
        Initialize the command handler.

        Args:
            console: Rich console for output
            state_manager: Game state manager for accessing and updating game state
        """
        self.console = console
        self.state_manager = state_manager

    @abstractmethod
    async def handle(self, command: ParsedCommand) -> ActionResult:
        """
        Handle the command and return an ActionResult.

        Args:
            command: The parsed command to handle

        Returns:
            ActionResult object containing the results of the command
        """
        pass

    async def get_required_state(
        self,
    ) -> tuple[PlayerState | None, Location | None, WorldState | None]:
        """
        Get the required state objects for command processing.

        Returns:
            Tuple of (player_state, current_location, world_state)
        """
        player_state = self.state_manager.player_tracker.get_state()
        await self.state_manager.player_tracker.get_current_location_id()
        current_location = await self.state_manager.get_current_location_details()
        world_state = self.state_manager.world_tracker.get_state()

        return player_state, current_location, world_state

    def display_message(self, message: str, success: bool = True) -> None:
        """
        Display a message to the player.

        Args:
            message: The message to display
            success: Whether the message represents a success (True) or failure (False)
        """
        if success:
            self.console.print(f"[green]{message}[/green]")
        else:
            self.console.print(f"[yellow]{message}[/yellow]")

    def normalize_name(self, name: str | None) -> str | None:
        """
        Normalize an object name by removing articles and extra whitespace.

        Args:
            name: The name to normalize

        Returns:
            The normalized name, or None if the input was None
        """
        if not name:
            return None

        return name.replace("the ", "").strip().lower()
