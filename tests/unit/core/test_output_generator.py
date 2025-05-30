"""Unit tests for OutputGenerator and supporting classes."""

from unittest.mock import patch
from uuid import uuid4

import pytest
from rich.console import Console

from game_loop.core.output_generator import OutputGenerator
from game_loop.core.response_formatter import ResponseFormatter
from game_loop.core.streaming_handler import StreamingHandler
from game_loop.core.template_manager import TemplateManager
from game_loop.state.models import ActionResult, Location


class TestOutputGenerator:
    """Test cases for OutputGenerator class."""

    @pytest.fixture
    def output_generator(self):
        """Create OutputGenerator instance for testing."""
        return OutputGenerator(console=Console())

    @pytest.fixture
    def mock_action_result(self):
        """Create mock ActionResult for testing."""
        return ActionResult(
            success=True,
            feedback_message="Action completed successfully",
            location_change=False,
            inventory_changes=None,
            stat_changes=None,
        )

    @pytest.fixture
    def mock_location(self):
        """Create mock Location for testing."""
        return Location(
            location_id=uuid4(),
            name="Test Room",
            description="A simple test room.",
            objects={},
            npcs={},
            connections={"north": uuid4(), "south": uuid4()},
            state_flags={},
        )

    def test_init_creates_components(self, output_generator):
        """Test that OutputGenerator initializes all components."""
        assert isinstance(output_generator.template_manager, TemplateManager)
        assert isinstance(output_generator.response_formatter, ResponseFormatter)
        assert isinstance(output_generator.streaming_handler, StreamingHandler)
        assert output_generator.console is not None

    def test_generate_response_success(self, output_generator, mock_action_result):
        """Test successful response generation."""
        with patch.object(
            output_generator.template_manager, "render_action_result"
        ) as mock_render:
            mock_render.return_value = "Template rendered output"

            output_generator.generate_response(mock_action_result, {})

            mock_render.assert_called_once_with(mock_action_result, {})

    def test_generate_response_fallback(self, output_generator, mock_action_result):
        """Test response generation falls back to formatter."""
        with (
            patch.object(
                output_generator.template_manager, "render_action_result"
            ) as mock_render,
            patch.object(
                output_generator.response_formatter, "format_action_feedback"
            ) as mock_format,
        ):
            mock_render.return_value = None
            mock_format.return_value = "Formatted output"

            output_generator.generate_response(mock_action_result, {})

            mock_format.assert_called_once_with(mock_action_result)

    def test_display_output(self, output_generator):
        """Test output display."""
        with patch.object(output_generator.console, "print") as mock_print:
            output_generator.display_output("Test message")
            mock_print.assert_called_once()


class TestTemplateManager:
    """Test cases for TemplateManager class."""

    @pytest.fixture
    def template_manager(self, tmp_path):
        """Create TemplateManager instance for testing."""
        return TemplateManager(str(tmp_path / "templates"))

    def test_init_creates_environment(self, template_manager):
        """Test that TemplateManager creates Jinja2 environment."""
        assert template_manager.env is not None
        assert hasattr(template_manager, "_highlight_filter")
        assert hasattr(template_manager, "_color_filter")

    def test_render_template_not_found(self, template_manager):
        """Test rendering non-existent template returns None."""
        result = template_manager.render_template("nonexistent.j2", {})
        assert result is None

    def test_highlight_filter(self, template_manager):
        """Test highlight filter functionality."""
        result = template_manager._highlight_filter("test", "bold")
        assert result == "[bold]test[/bold]"

    def test_color_filter(self, template_manager):
        """Test color filter functionality."""
        result = template_manager._color_filter("test", "red")
        assert result == "[red]test[/red]"


class TestResponseFormatter:
    """Test cases for ResponseFormatter class."""

    @pytest.fixture
    def formatter(self):
        """Create ResponseFormatter instance for testing."""
        return ResponseFormatter(console=Console())

    @pytest.fixture
    def mock_location(self):
        """Create mock Location for testing."""
        return Location(
            location_id=uuid4(),
            name="Test Room",
            description="A test room",
            objects={},
            npcs={},
            connections={"north": uuid4()},
            state_flags={},
        )

    def test_format_location_panel(self, formatter, mock_location):
        """Test location panel formatting."""
        result = formatter.format_location_basic({"location": mock_location})
        # Should complete without errors
        assert result is None  # method returns None

    def test_format_error_message(self, formatter):
        """Test error message formatting."""
        result = formatter.format_error_basic("Test error", "error")
        # Should complete without errors
        assert result is None  # method returns None

    def test_format_inventory_table_empty(self, formatter):
        """Test empty inventory formatting."""
        result = formatter.format_inventory([])
        assert result is not None

    def test_format_inventory_table_with_items(self, formatter):
        """Test inventory formatting with items."""
        items = [
            {"name": "Sword", "description": "Sharp blade", "quantity": 1},
            {"name": "Potion", "description": "Healing", "quantity": 2},
        ]
        result = formatter.format_inventory(items)
        assert result is not None


class TestStreamingHandler:
    """Test cases for StreamingHandler class."""

    @pytest.fixture
    def streaming_handler(self):
        """Create StreamingHandler instance for testing."""
        return StreamingHandler(console=Console())

    def test_get_style_config(self, streaming_handler):
        """Test style configuration retrieval."""
        config = streaming_handler._get_style_config("narrative")
        assert "title" in config
        assert "border_style" in config
        assert "text_style" in config
        assert "title_style" in config

    def test_stream_response_empty_generator(self, streaming_handler):
        """Test streaming with empty generator."""
        result = streaming_handler.stream_response(iter([]))
        assert result == ""

    def test_stream_response_with_chunks(self, streaming_handler):
        """Test streaming with response chunks."""
        chunks = ["Hello", " ", "World"]
        with patch.object(streaming_handler.console, "print"):
            result = streaming_handler.stream_response(iter(chunks))
            assert result == "Hello World"
