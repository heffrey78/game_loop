#!/usr/bin/env python3
"""
Game Loop - Text adventure with natural language processing capabilities.
Main entry point for the application.
"""

import asyncio
import logging
import os
import sys

import asyncpg
from rich.console import Console

from game_loop.config.models import GameConfig
from game_loop.core.game_loop import GameLoop

logger = logging.getLogger(__name__)


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

        # Initialize database connection pool
        db_pool = await create_db_pool()
        if not db_pool:
            console.print(
                "[bold red]Failed to connect to database. "
                "Please check if the database is running.[/bold red]"
            )
            return 1

        # Create and initialize the game loop
        game = GameLoop(config, db_pool, console)
        await game.initialize()

        # Start the game loop
        await game.run()

        # Close database connection pool when done
        await db_pool.close()

    except KeyboardInterrupt:
        console.print("\n[bold]Game interrupted. Farewell![/bold]")
        return 1
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        console.print_exception()
        return 1

    return 0


async def create_db_pool() -> asyncpg.Pool:
    """Create and return a connection pool to the PostgreSQL database."""
    try:
        # Get database connection details from environment variables, with defaults
        host = os.environ.get("POSTGRES_HOST", "localhost")
        port = os.environ.get("POSTGRES_PORT", "5432")
        database = os.environ.get("POSTGRES_DB", "game_loop")
        user = os.environ.get("POSTGRES_USER", "postgres")
        password = os.environ.get("POSTGRES_PASSWORD", "postgres")

        # Create the connection pool
        pool = await asyncpg.create_pool(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
        )
        logger.info(f"Successfully connected to PostgreSQL database at {host}:{port}")
        return pool
    except Exception as e:
        logger.error(f"Failed to create database connection pool: {e}")
        return None


def main() -> int:
    """
    Entry point for the Game Loop application.
    This non-async wrapper ensures the async code is properly executed.
    """
    return asyncio.run(_main_async())


if __name__ == "__main__":
    sys.exit(main())
