#!/usr/bin/env python3
"""
Demo script to show the difference between canned and LLM-powered NPC dialogue.
"""

import asyncio

# Add project root to path
import sys
from pathlib import Path

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()

# Simulated NPC responses
CANNED_RESPONSES = {
    "neutral": "Security Guard nods politely. 'Greetings, traveler.'",
    "friendly": "Security Guard smiles at you warmly. 'Hello there! How can I help you?'",
    "hostile": "Security Guard glares at you suspiciously. 'What do you want?'",
}

LLM_RESPONSE_EXAMPLE = """Security Guard stands tall and speaks firmly. 'Welcome to the reception area. Please be on alert for any unauthorized visitors.'

Security Guard adds: 'There's a restricted section where maintenance happens at night. It's best to avoid it alone.'"""


async def demo_dialogue():
    """Demonstrate the difference between dialogue systems."""

    console.print("\n[bold cyan]NPC Dialogue System Comparison[/bold cyan]\n")

    # Show the user's input
    console.print("[bold]Player Input:[/bold] talk to the security guard\n")

    # Old system
    old_panel = Panel(
        CANNED_RESPONSES["neutral"],
        title="[red]Old System (Canned Response)[/red]",
        border_style="red",
    )

    # New system
    new_panel = Panel(
        LLM_RESPONSE_EXAMPLE,
        title="[green]New System (LLM-Powered)[/green]",
        border_style="green",
    )

    # Display side by side
    console.print(Columns([old_panel, new_panel], equal=True))

    console.print("\n[bold]Key Differences:[/bold]")
    console.print(
        "• [red]Old:[/red] Fixed responses based on dialogue state (neutral/friendly/hostile)"
    )
    console.print("• [red]Old:[/red] Same response every time for the same state")
    console.print("• [red]Old:[/red] No awareness of location or context")
    console.print()
    console.print(
        "• [green]New:[/green] Dynamic responses generated based on NPC role and location"
    )
    console.print(
        "• [green]New:[/green] Personality traits influence behavior (serious, cautious, professional)"
    )
    console.print(
        "• [green]New:[/green] Provides useful local knowledge about the area"
    )
    console.print(
        "• [green]New:[/green] Can discuss multiple topics (security, building layout, etc.)"
    )
    console.print(
        "• [green]New:[/green] Adapts to player's exploration style and experience level"
    )

    console.print("\n[bold]Additional Features with LLM:[/bold]")
    console.print("• Different responses for friendly vs suspicious approaches")
    console.print("• Special interactions triggered by specific questions")
    console.print("• Contextual awareness of the abandoned office setting")
    console.print("• Can provide hints about keycards, restricted areas, etc.")

    console.print(
        "\n[bold yellow]Note:[/bold yellow] LLM generation requires Ollama to be running with a compatible model (qwen2.5:3b)"
    )


if __name__ == "__main__":
    asyncio.run(demo_dialogue())
