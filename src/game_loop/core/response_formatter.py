"""
ResponseFormatter for Game Loop - Rich text formatting utilities.
Handles basic text formatting for game output when templates aren't used.
"""

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class ResponseFormatter:
    """Handles basic Rich text formatting for game output."""

    def __init__(self, console: Console):
        """
        Initialize the ResponseFormatter.

        Args:
            console: Rich console instance for output
        """
        self.console = console

    def format_location_basic(self, location_data: Any) -> None:
        """
        Format location data using basic Rich formatting.

        Args:
            location_data: Dictionary or Location object containing
                location information
        """
        # Handle both Location objects and dictionaries
        if hasattr(location_data, "name"):
            # Location object
            name = location_data.name
            description = location_data.description
            exits = (
                list(location_data.connections.keys())
                if location_data.connections
                else []
            )
            items = list(location_data.objects.keys()) if location_data.objects else []
        else:
            # Dictionary
            name = location_data.get("name", "Unknown Location")
            description = location_data.get("description", "No description.")
            exits = location_data.get("exits", [])
            items = location_data.get("items", [])

        # Create location panel
        content = f"[bold]{description}[/bold]\n"

        if exits:
            content += f"\n[dim]Exits: {', '.join(exits)}[/dim]"

        if items:
            content += f"\n[dim]Items: {', '.join(items)}[/dim]"

        panel = Panel(content, title=f"[blue]{name}[/blue]", border_style="blue")
        self.console.print(panel)

    def format_error_basic(self, error: str, error_type: str = "error") -> None:
        """
        Format error message using basic Rich formatting.

        Args:
            error: Error message text
            error_type: Type of error (error, warning, info)
        """
        if error_type == "warning":
            self.console.print(f"[yellow]⚠ {error}[/yellow]")
        elif error_type == "info":
            self.console.print(f"[blue]ℹ {error}[/blue]")
        else:
            self.console.print(f"[red]✗ {error}[/red]")

    def format_system_message_basic(
        self, message: str, message_type: str = "info"
    ) -> None:
        """
        Format system message using basic Rich formatting.

        Args:
            message: System message text
            message_type: Type of message (info, warning, success)
        """
        if message_type == "success":
            self.console.print(f"[green]✓ {message}[/green]")
        elif message_type == "warning":
            self.console.print(f"[yellow]⚠ {message}[/yellow]")
        else:
            self.console.print(f"[blue]ℹ {message}[/blue]")

    def format_inventory(self, inventory_data: list[dict[str, Any]]) -> Table:
        """
        Format inventory data as a Rich table.

        Args:
            inventory_data: List of inventory items

        Returns:
            Rich Table object
        """
        table = Table(title="Inventory", show_header=True, header_style="bold")
        table.add_column("Item", style="cyan")
        table.add_column("Quantity", justify="center", style="green")
        table.add_column("Description", style="dim")

        for item in inventory_data:
            name = item.get("name", "Unknown")
            quantity = str(item.get("quantity", 1))
            description = item.get("description", "")

            table.add_row(name, quantity, description)

        return table

    def format_dialogue(
        self, speaker: str, text: str, npc_data: dict | None = None
    ) -> Panel:
        """
        Format NPC dialogue as a Rich panel.

        Args:
            speaker: Name of the speaking character
            text: Dialogue text
            npc_data: Additional NPC information

        Returns:
            Rich Panel object
        """
        # Style based on NPC data or defaults
        if npc_data:
            speaker_style = npc_data.get("color", "bold")
            border_style = npc_data.get("border_color", "white")
        else:
            speaker_style = "bold"
            border_style = "white"

        panel = Panel(
            text,
            title=f"[{speaker_style}]{speaker}[/{speaker_style}]",
            border_style=border_style,
            padding=(1, 2),
        )

        return panel

    def format_action_feedback(
        self, action_result: Any, result: Any = None, success: Any = None
    ) -> Text:
        """
        Format action feedback as Rich text.

        Args:
            action_result: ActionResult object or action string
            result: Result description (for backward compatibility)
            success: Whether the action was successful
                (for backward compatibility)

        Returns:
            Rich Text object
        """
        # Handle ActionResult object
        if hasattr(action_result, "feedback_message"):
            feedback_text = action_result.feedback_message or ""
            is_success = action_result.success
        else:
            # Handle as separate parameters (backward compatibility)
            action = action_result
            feedback_text = result or ""
            is_success = success if success is not None else True

        if is_success:
            icon = "✓"
            color = "green"
        else:
            icon = "✗"
            color = "red"

        text = Text()
        text.append(f"{icon} ", style=color)

        if hasattr(action_result, "feedback_message"):
            # For ActionResult, just show the feedback message
            text.append(feedback_text, style=color)
        else:
            # For backward compatibility with separate parameters
            text.append(f"{action}: ", style="bold")
            text.append(feedback_text, style=color)

        return text

    def format_health_status(self, health: int, max_health: int) -> Text:
        """
        Format health status with color coding.

        Args:
            health: Current health points
            max_health: Maximum health points

        Returns:
            Rich Text object
        """
        percentage = health / max_health if max_health > 0 else 0

        if percentage > 0.7:
            color = "green"
        elif percentage > 0.3:
            color = "yellow"
        else:
            color = "red"

        text = Text()
        text.append("Health: ", style="bold")
        text.append(f"{health}/{max_health}", style=color)

        # Add health bar
        bar_length = 20
        filled = int(bar_length * percentage)
        empty = bar_length - filled

        text.append(" [")
        text.append("█" * filled, style=color)
        text.append("░" * empty, style="dim")
        text.append("]")

        return text

    def format_stats_table(self, stats: dict[str, Any]) -> Table:
        """
        Format character stats as a Rich table.

        Args:
            stats: Dictionary of stat names and values

        Returns:
            Rich Table object
        """
        table = Table(title="Stats", show_header=True, header_style="bold")
        table.add_column("Stat", style="cyan")
        table.add_column("Value", justify="center", style="green")

        for stat_name, value in stats.items():
            # Format stat name (convert snake_case to Title Case)
            display_name = stat_name.replace("_", " ").title()
            table.add_row(display_name, str(value))

        return table
