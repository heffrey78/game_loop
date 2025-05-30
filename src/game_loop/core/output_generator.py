"""
OutputGenerator for Game Loop - Central output management system.
Handles all game output generation with rich text formatting.
"""

from collections.abc import Iterator
from typing import Any

from rich.console import Console

from game_loop.core.response_formatter import ResponseFormatter
from game_loop.core.streaming_handler import StreamingHandler
from game_loop.core.template_manager import TemplateManager
from game_loop.state.models import ActionResult


class OutputGenerator:
    """Central output generation system for the game loop."""

    def __init__(self, console: Console, template_dir: str = "templates"):
        """
        Initialize the OutputGenerator.

        Args:
            console: Rich console instance for output
            template_dir: Directory containing Jinja2 templates
        """
        self.console = console
        self.template_manager = TemplateManager(template_dir)
        self.response_formatter = ResponseFormatter(console)
        self.streaming_handler = StreamingHandler(console)

    def generate_response(
        self, action_result: ActionResult, context: dict[str, Any]
    ) -> None:
        """
        Generate and display response from ActionResult.

        Args:
            action_result: Result of a player action
            context: Additional context for response generation
        """
        try:
            # Use template manager to render the response
            rendered_response = self.template_manager.render_action_result(
                action_result, context
            )

            if rendered_response:
                # Display the rendered response
                if action_result.success:
                    self.console.print(f"[green]{rendered_response}[/green]")
                else:
                    self.console.print(f"[yellow]{rendered_response}[/yellow]")
            else:
                # Fallback to response formatter
                formatted_text = self.response_formatter.format_action_feedback(
                    action_result
                )
                self.console.print(formatted_text)

        except Exception as e:
            # Fallback to basic output on template error
            if action_result.feedback_message:
                if action_result.success:
                    msg = action_result.feedback_message
                    self.console.print(f"[green]{msg}[/green]")
                else:
                    msg = action_result.feedback_message
                    self.console.print(f"[yellow]{msg}[/yellow]")
            else:
                error_msg = f"Error generating response: {str(e)}"
                self.console.print(f"[red]{error_msg}[/red]")

    def format_location_description(self, location_data: Any) -> None:
        """
        Format and display location description with rich text.

        Args:
            location_data: Dictionary containing location information
        """
        try:
            # Use template manager for location rendering
            rendered_location = self.template_manager.render_template(
                "locations/description.j2", location_data
            )

            if rendered_location:
                self.console.print(rendered_location)
            else:
                # Fallback to basic location display
                formatter = self.response_formatter
                formatter.format_location_basic(location_data)

        except Exception:
            # Fallback to basic formatting on error
            formatter = self.response_formatter
            formatter.format_location_basic(location_data)

    def format_error_message(self, error: str, error_type: str = "error") -> None:
        """
        Format and display error messages.

        Args:
            error: Error message text
            error_type: Type of error (error, warning, info)
        """
        try:
            # Use template for error formatting
            context = {"error": error, "error_type": error_type}
            rendered_error = self.template_manager.render_template(
                "messages/error.j2", context
            )

            if rendered_error:
                self.console.print(rendered_error)
            else:
                # Fallback to basic error formatting
                formatter = self.response_formatter
                formatter.format_error_basic(error, error_type)

        except Exception:
            # Ultimate fallback
            formatter = self.response_formatter
            formatter.format_error_basic(error, error_type)

    def format_system_message(self, message: str, message_type: str = "info") -> None:
        """
        Format and display system messages.

        Args:
            message: System message text
            message_type: Type of message (info, warning, success)
        """
        try:
            # Use template for system message formatting
            context = {"message": message, "message_type": message_type}
            rendered_message = self.template_manager.render_template(
                "messages/info.j2", context
            )

            if rendered_message:
                self.console.print(rendered_message)
            else:
                # Fallback to basic message formatting
                formatter = self.response_formatter
                formatter.format_system_message_basic(message, message_type)

        except Exception:
            # Ultimate fallback
            formatter = self.response_formatter
            formatter.format_system_message_basic(message, message_type)

    def stream_llm_response(
        self, response_generator: Iterator[str], response_type: str = "narrative"
    ) -> None:
        """
        Handle streaming LLM responses with real-time display.

        Args:
            response_generator: Iterator yielding response chunks
            response_type: Type of response (narrative, dialogue, action)
        """
        try:
            # Use streaming handler for real-time display
            final_response = self.streaming_handler.stream_response(
                response_generator, response_type
            )

            # Optionally log the final response or process it further
            if final_response:
                pass  # Could add logging here

        except Exception as e:
            error_msg = f"Error streaming response: {str(e)}"
            self.console.print(f"[red]{error_msg}[/red]")

    def format_inventory_display(self, inventory_data: list[dict[str, Any]]) -> None:
        """
        Format and display inventory information.

        Args:
            inventory_data: List of inventory items
        """
        try:
            # Use response formatter for inventory
            inventory_table = self.response_formatter.format_inventory(inventory_data)
            self.console.print(inventory_table)

        except Exception as e:
            error_msg = f"Error displaying inventory: {str(e)}"
            self.console.print(f"[red]{error_msg}[/red]")

    def format_dialogue(
        self, speaker: str, text: str, npc_data: dict | None = None
    ) -> None:
        """
        Format and display NPC dialogue.

        Args:
            speaker: Name of the speaking character
            text: Dialogue text
            npc_data: Additional NPC information
        """
        try:
            # Use response formatter for dialogue
            dialogue_panel = self.response_formatter.format_dialogue(
                speaker, text, npc_data
            )
            self.console.print(dialogue_panel)

        except Exception as e:
            error_msg = f"Error displaying dialogue: {str(e)}"
            self.console.print(f"[red]{error_msg}[/red]")

    def format_action_feedback(self, action: str, result: str, success: bool) -> None:
        """
        Format and display action feedback.

        Args:
            action: Action that was performed
            result: Result description
            success: Whether the action was successful
        """
        try:
            # Use response formatter for action feedback
            feedback_text = self.response_formatter.format_action_feedback(
                action, result, success
            )
            self.console.print(feedback_text)

        except Exception as e:
            error_msg = f"Error displaying action feedback: {str(e)}"
            self.console.print(f"[red]{error_msg}[/red]")

    def display_output(self, message: str) -> None:
        """
        Display a simple output message.

        Args:
            message: Message to display
        """
        self.console.print(message)

    def display_location(self, location_data: Any) -> None:
        """
        Display location information.

        Args:
            location_data: Dictionary or Location object containing location information
        """
        self.format_location_description(location_data)

    def display_error(self, error: str, error_type: str = "error") -> None:
        """
        Display an error message.

        Args:
            error: Error message text
            error_type: Type of error (error, warning, info)
        """
        self.format_error_message(error, error_type)

    def display_system_message(self, message: str, message_type: str = "info") -> None:
        """
        Display a system message.

        Args:
            message: System message text
            message_type: Type of message (info, warning, success)
        """
        self.format_system_message(message, message_type)
