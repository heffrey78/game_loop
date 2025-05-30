"""
StreamingHandler for Game Loop - Real-time streaming response display.
Handles live streaming of LLM responses with Rich formatting.
"""

import time
from collections.abc import Iterator
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text


class StreamingHandler:
    """Handles real-time streaming display of LLM responses."""

    def __init__(self, console: Console):
        """
        Initialize the StreamingHandler.

        Args:
            console: Rich console instance for output
        """
        self.console = console

    def stream_response(
        self, response_generator: Iterator[str], response_type: str = "narrative"
    ) -> str:
        """
        Stream LLM response with real-time display.

        Args:
            response_generator: Iterator yielding response chunks
            response_type: Type of response (narrative, dialogue, action)

        Returns:
            Complete response text
        """
        full_response = ""
        display_text = Text()

        # Configure styling based on response type
        style_config = self._get_style_config(response_type)

        try:
            with Live(
                self._create_panel(display_text, style_config),
                refresh_per_second=10,
                console=self.console,
            ) as live:
                for chunk in response_generator:
                    full_response += chunk
                    display_text.append(chunk, style=style_config["text_style"])

                    # Update the live display
                    live.update(self._create_panel(display_text, style_config))

                    # Small delay to make streaming visible
                    time.sleep(0.05)

        except Exception:
            # If streaming fails, just print the accumulated text
            if full_response:
                self.console.print(full_response)

        return full_response

    def _get_style_config(self, response_type: str) -> dict[str, Any]:
        """
        Get styling configuration for different response types.

        Args:
            response_type: Type of response

        Returns:
            Dictionary with styling configuration
        """
        configs = {
            "narrative": {
                "title": "Story",
                "border_style": "blue",
                "text_style": "white",
                "title_style": "bold blue",
            },
            "dialogue": {
                "title": "Speech",
                "border_style": "green",
                "text_style": "white",
                "title_style": "bold green",
            },
            "action": {
                "title": "Action",
                "border_style": "yellow",
                "text_style": "white",
                "title_style": "bold yellow",
            },
            "system": {
                "title": "System",
                "border_style": "red",
                "text_style": "dim white",
                "title_style": "bold red",
            },
        }

        return configs.get(response_type, configs["narrative"])

    def _create_panel(self, text: Text, style_config: dict[str, Any]) -> Panel:
        """
        Create a Rich panel for streaming display.

        Args:
            text: Text content to display
            style_config: Styling configuration

        Returns:
            Rich Panel object
        """
        return Panel(
            text,
            title=f"[{style_config['title_style']}]"
            f"{style_config['title']}[/{style_config['title_style']}]",
            border_style=style_config["border_style"],
            padding=(1, 2),
        )

    def stream_typewriter_effect(
        self, text: str, delay: float = 0.03, response_type: str = "narrative"
    ) -> None:
        """
        Display text with typewriter effect.

        Args:
            text: Text to display
            delay: Delay between characters
            response_type: Type of response for styling
        """
        display_text = Text()
        style_config = self._get_style_config(response_type)

        try:
            with Live(
                self._create_panel(display_text, style_config),
                refresh_per_second=30,
                console=self.console,
            ) as live:
                for char in text:
                    display_text.append(char, style=style_config["text_style"])
                    live.update(self._create_panel(display_text, style_config))
                    time.sleep(delay)

        except Exception:
            # If typewriter effect fails, just print the text
            self.console.print(text)

    def stream_with_progress(
        self,
        response_generator: Iterator[str],
        total_expected: int = 100,
        response_type: str = "narrative",
    ) -> str:
        """
        Stream response with progress indication.

        Args:
            response_generator: Iterator yielding response chunks
            total_expected: Expected total length (for progress estimation)
            response_type: Type of response

        Returns:
            Complete response text
        """
        full_response = ""
        display_text = Text()
        style_config = self._get_style_config(response_type)

        try:
            with Live(
                self._create_progress_panel(
                    display_text, style_config, 0, total_expected
                ),
                refresh_per_second=10,
                console=self.console,
            ) as live:
                for chunk in response_generator:
                    full_response += chunk
                    display_text.append(chunk, style=style_config["text_style"])

                    progress = min(len(full_response), total_expected)
                    live.update(
                        self._create_progress_panel(
                            display_text, style_config, progress, total_expected
                        )
                    )

                    time.sleep(0.05)

        except Exception:
            if full_response:
                self.console.print(full_response)

        return full_response

    def _create_progress_panel(
        self, text: Text, style_config: dict[str, Any], progress: int, total: int
    ) -> Panel:
        """
        Create panel with progress indication.

        Args:
            text: Text content
            style_config: Styling configuration
            progress: Current progress
            total: Total expected

        Returns:
            Rich Panel with progress indication
        """
        percentage = (progress / total * 100) if total > 0 else 0
        title = (
            f"[{style_config['title_style']}]"
            f"{style_config['title']} ({percentage:.1f}%)"
            f"[/{style_config['title_style']}]"
        )

        return Panel(
            text, title=title, border_style=style_config["border_style"], padding=(1, 2)
        )
