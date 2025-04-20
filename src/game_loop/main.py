#!/usr/bin/env python3
"""
Game Loop - Text adventure with natural language processing capabilities.
Main entry point for the application.
"""

from rich.console import Console


console = Console()


def main() -> None:
    """Main entry point for the Game Loop application."""
    console.print("[bold green]Welcome to Game Loop![/bold green]", style="bold")
    console.print("A text adventure with natural language processing capabilities.")
    console.print("\nThis is a placeholder. Implementation coming soon...")


if __name__ == "__main__":
    main()
