#!/usr/bin/env python3
"""
Game Loop - Text adventure with natural language processing capabilities.
Main entry point for the application.
"""

import asyncio
import sys

from rich.console import Console

from game_loop.config.models import GameConfig
from game_loop.core.game_loop import GameLoop


async def _main_async() -> int:
    """Async implementation of the main entry point for the Game Loop application."""
    console = Console()

    try:
        # Display welcome banner
        console.print("[bold green]Welcome to Game Loop![/bold green]", style="bold")
        console.print("A text adventure with natural language processing capabilities.")
        console.print()

        # Load configuration (in a full implementation, this would use CLI args, etc.)
        config = GameConfig()

        # Create and initialize the game loop
        game = GameLoop(config, console)
        game.initialize()

        # Start the game loop
        await game.start()

    except KeyboardInterrupt:
        console.print("\n[bold]Game interrupted. Farewell![/bold]")
        return 1
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        console.print_exception()
        return 1

    return 0


def main() -> int:
    """
    Entry point for the Game Loop application.
    This non-async wrapper ensures the async code is properly executed.
    """
    return asyncio.run(_main_async())


if __name__ == "__main__":
    sys.exit(main())
