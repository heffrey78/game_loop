"""
TemplateManager for Game Loop - Jinja2 template rendering system.
Handles loading and rendering of templates for game output.
"""

from pathlib import Path
from typing import Any

import jinja2

from game_loop.state.models import ActionResult


class TemplateManager:
    """Manages Jinja2 templates for game output generation."""

    def __init__(self, template_dir: str = "templates"):
        """
        Initialize the TemplateManager.

        Args:
            template_dir: Directory containing Jinja2 templates
        """
        self.template_dir = Path(template_dir)
        self._setup_environment()

    def _setup_environment(self) -> None:
        """Set up the Jinja2 environment with proper configuration."""
        try:
            # Ensure template directory exists
            self.template_dir.mkdir(parents=True, exist_ok=True)

            # Create Jinja2 environment
            self.env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(str(self.template_dir)),
                autoescape=True,
                trim_blocks=True,
                lstrip_blocks=True,
            )

            # Add custom filters for game-specific formatting
            self.env.filters["highlight"] = self._highlight_filter
            self.env.filters["color"] = self._color_filter

        except Exception:
            # If template setup fails, create a minimal environment
            self.env = jinja2.Environment(loader=jinja2.DictLoader({}))

    def _highlight_filter(self, text: str, style: str = "bold") -> str:
        """
        Custom filter to highlight text with Rich markup.

        Args:
            text: Text to highlight
            style: Rich style to apply

        Returns:
            Text wrapped in Rich markup
        """
        return f"[{style}]{text}[/{style}]"

    def _color_filter(self, text: str, color: str = "white") -> str:
        """
        Custom filter to color text with Rich markup.

        Args:
            text: Text to color
            color: Rich color to apply

        Returns:
            Text wrapped in Rich color markup
        """
        return f"[{color}]{text}[/{color}]"

    def render_template(
        self, template_name: str, context: dict[str, Any]
    ) -> str | None:
        """
        Render a template with the given context.

        Args:
            template_name: Name of the template file
            context: Variables to pass to the template

        Returns:
            Rendered template string or None if template not found
        """
        try:
            template = self.env.get_template(template_name)
            rendered = template.render(**context)
            return str(rendered)
        except jinja2.TemplateNotFound:
            return None
        except Exception:
            return None

    def render_action_result(
        self, action_result: ActionResult, context: dict[str, Any]
    ) -> str | None:
        """
        Render an ActionResult using appropriate template.

        Args:
            action_result: The action result to render
            context: Additional context variables

        Returns:
            Rendered template string or None if no template found
        """
        # Determine template based on action result type
        template_name = self._get_action_template(action_result)

        if not template_name:
            return None

        # Combine action result data with context
        template_context = {
            **context,
            "action_result": action_result,
            "success": action_result.success,
            "message": action_result.feedback_message,
            "location_change": action_result.location_change,
            "inventory_changes": action_result.inventory_changes,
            "knowledge_updates": action_result.knowledge_updates,
            "stat_changes": action_result.stat_changes,
        }

        return self.render_template(template_name, template_context)

    def _get_action_template(self, action_result: ActionResult) -> str | None:
        """
        Determine the appropriate template for an ActionResult.

        Args:
            action_result: The action result to get template for

        Returns:
            Template filename or None if no template available
        """
        # Try to determine template based on action result characteristics
        if action_result.location_change:
            template_name = "actions/location_change.j2"
            if self._template_exists(template_name):
                return template_name

        if action_result.inventory_changes:
            template_name = "actions/inventory_change.j2"
            if self._template_exists(template_name):
                return template_name

        if action_result.triggers_evolution:
            template_name = "actions/evolution.j2"
            if self._template_exists(template_name):
                return template_name

        # Fall back to success/error templates
        if action_result.success:
            template_name = "actions/success.j2"
        else:
            template_name = "actions/error.j2"

        if self._template_exists(template_name):
            return template_name

        return None

    def _template_exists(self, template_name: str) -> bool:
        """
        Check if a template file exists.

        Args:
            template_name: Name of the template file

        Returns:
            True if template exists, False otherwise
        """
        template_path = self.template_dir / template_name
        return template_path.exists() and template_path.is_file()

    def create_default_templates(self) -> None:
        """Create default template files if they don't exist."""
        default_templates = {
            "actions/success.j2": """
[green]✓ {{ message or "Action completed successfully" }}[/green]
{% if location_change %}
[dim]Location changed[/dim]
{% endif %}
{% if inventory_changes %}
[dim]Inventory updated[/dim]
{% endif %}
{% if stat_changes %}
[dim]Stats modified[/dim]
{% endif %}
            """.strip(),
            "actions/error.j2": """
[red]✗ {{ message or "Action failed" }}[/red]
            """.strip(),
            "locations/description.j2": """
[bold blue]{{ name or "Unknown Location" }}[/bold blue]

{{ description or "No description available." }}

{% if exits %}
[dim]Exits: {{ exits | join(", ") }}[/dim]
{% endif %}

{% if items %}
[dim]Items: {{ items | join(", ") }}[/dim]
{% endif %}
            """.strip(),
            "messages/error.j2": """
{% if error_type == "warning" %}
[yellow]⚠ {{ error }}[/yellow]
{% elif error_type == "info" %}
[blue]ℹ {{ error }}[/blue]
{% else %}
[red]✗ {{ error }}[/red]
{% endif %}
            """.strip(),
            "messages/info.j2": """
{% if message_type == "success" %}
[green]✓ {{ message }}[/green]
{% elif message_type == "warning" %}
[yellow]⚠ {{ message }}[/yellow]
{% else %}
[blue]ℹ {{ message }}[/blue]
{% endif %}
            """.strip(),
        }

        for template_name, content in default_templates.items():
            template_path = self.template_dir / template_name
            template_path.parent.mkdir(parents=True, exist_ok=True)

            if not template_path.exists():
                template_path.write_text(content)
