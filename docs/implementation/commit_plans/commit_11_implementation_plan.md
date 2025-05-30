# Commit 11: Basic Output Generation - Implementation Plan

## Overview

This commit implements the basic output generation system for the Game Loop text adventure. The system will create an OutputGenerator class, implement response formatting with rich text support, create a template-based messaging system, and add a basic streaming response handler.

## Goals

- Create a centralized OutputGenerator class for all game output
- Implement rich text formatting using the existing Rich library integration
- Create a template-based messaging system for consistent output formatting
- Add streaming response handler for real-time LLM output
- Ensure all output is properly formatted and visually appealing
- Maintain consistency with existing Rich console usage patterns

## Technical Requirements

### Dependencies
- Rich library (already configured in pyproject.toml)
- Jinja2 for template rendering (needs to be added)
- Existing ActionResult model for structured data
- Integration with current GameLoop and LocationDisplay patterns

### Core Components

1. **OutputGenerator Class**: Central output management
2. **ResponseFormatter**: Format different types of content
3. **TemplateManager**: Handle template-based messaging
4. **StreamingHandler**: Manage real-time response streaming
5. **Rich Text Utilities**: Enhanced terminal formatting

## Implementation Details

### 1. OutputGenerator Class (`src/game_loop/core/output_generator.py`)

```python
from typing import Any, Dict, List, Optional, Union
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markup import escape
from jinja2 import Environment, FileSystemLoader

from game_loop.state.models import ActionResult
from game_loop.core.template_manager import TemplateManager
from game_loop.core.response_formatter import ResponseFormatter
from game_loop.core.streaming_handler import StreamingHandler

class OutputGenerator:
    """Central output generation system for the game loop."""

    def __init__(self, console: Console, template_dir: str = "templates"):
        self.console = console
        self.template_manager = TemplateManager(template_dir)
        self.response_formatter = ResponseFormatter(console)
        self.streaming_handler = StreamingHandler(console)

    def generate_response(self, action_result: ActionResult, context: Dict[str, Any]) -> None:
        """Generate and display response from ActionResult."""

    def format_location_description(self, location_data: Dict[str, Any]) -> None:
        """Format and display location description with rich text."""

    def format_error_message(self, error: str, error_type: str = "error") -> None:
        """Format and display error messages."""

    def format_system_message(self, message: str, message_type: str = "info") -> None:
        """Format and display system messages."""

    def stream_llm_response(self, response_generator, response_type: str = "narrative") -> None:
        """Handle streaming LLM responses with real-time display."""
```

### 2. ResponseFormatter Class (`src/game_loop/core/response_formatter.py`)

```python
from typing import Any, Dict, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich.table import Table

class ResponseFormatter:
    """Handles formatting of different response types with Rich text."""

    def __init__(self, console: Console):
        self.console = console

    def format_narrative(self, text: str, style: str = "narrative") -> Panel:
        """Format narrative text with appropriate styling."""

    def format_dialogue(self, speaker: str, text: str, npc_data: Optional[Dict] = None) -> Panel:
        """Format NPC dialogue with speaker identification."""

    def format_action_feedback(self, action: str, result: str, success: bool) -> Text:
        """Format feedback for player actions."""

    def format_inventory(self, items: List[Dict[str, Any]]) -> Table:
        """Format inventory display as a table."""

    def format_location_exits(self, exits: List[Dict[str, Any]]) -> Columns:
        """Format available exits in a location."""

    def format_objects_list(self, objects: List[Dict[str, Any]]) -> Text:
        """Format list of objects in the current location."""
```

### 3. TemplateManager Class (`src/game_loop/core/template_manager.py`)

```python
from typing import Any, Dict, Optional
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, Template
from rich.markup import escape

class TemplateManager:
    """Manages Jinja2 templates for consistent message formatting."""

    def __init__(self, template_dir: str = "templates"):
        self.template_dir = Path(template_dir)
        self.env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=True
        )
        self._register_filters()

    def render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render a template with the given context."""

    def render_action_result(self, action_result: ActionResult, context: Dict[str, Any]) -> str:
        """Render action result using appropriate template."""

    def get_template_for_action(self, action_type: str) -> str:
        """Get template name for specific action type."""

    def _register_filters(self) -> None:
        """Register custom Jinja2 filters for Rich markup."""
```

### 4. StreamingHandler Class (`src/game_loop/core/streaming_handler.py`)

```python
from typing import Iterator, Optional
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

class StreamingHandler:
    """Handles streaming responses from LLM with real-time display."""

    def __init__(self, console: Console):
        self.console = console

    def stream_response(self, response_generator: Iterator[str],
                       response_type: str = "narrative") -> str:
        """Stream LLM response with live updates."""

    def stream_to_panel(self, response_generator: Iterator[str],
                       title: str = "Response") -> str:
        """Stream response inside a Rich panel."""

    def format_streaming_text(self, text: str, response_type: str) -> Text:
        """Apply formatting to streaming text based on response type."""
```

### 5. Template Files (`src/game_loop/templates/`)

Create template directory structure:
```
templates/
├── actions/
│   ├── movement.j2
│   ├── interaction.j2
│   ├── dialogue.j2
│   └── system.j2
├── locations/
│   ├── description.j2
│   └── detailed.j2
├── messages/
│   ├── error.j2
│   ├── info.j2
│   └── warning.j2
└── responses/
    ├── success.j2
    ├── failure.j2
    └── partial.j2
```

Example template (`templates/actions/movement.j2`):
```jinja2
{% if success %}
[bold green]✓[/] You {{ action }} {{ direction }}.

{{ location.description }}

{% if location.exits %}
[dim]Exits:[/] {{ location.exits | join(", ") }}
{% endif %}

{% if location.objects %}
[dim]You see:[/] {{ location.objects | join(", ") }}
{% endif %}
{% else %}
[bold red]✗[/] You cannot {{ action }} {{ direction }}.
{{ reason }}
{% endif %}
```

### 6. Integration with GameLoop (`src/game_loop/core/game_loop.py`)

Modify existing GameLoop to use OutputGenerator:
```python
# ...existing imports...
from game_loop.core.output_generator import OutputGenerator

class GameLoop:
    def __init__(self, ...):
        # ...existing code...
        self.output_generator = OutputGenerator(self.console, "src/game_loop/templates")

    async def process_input(self, user_input: str) -> None:
        # ...existing code...

        # Replace current output handling with OutputGenerator
        self.output_generator.generate_response(action_result, context)

    def display_location(self, location_data: Dict[str, Any]) -> None:
        # Replace current location display with OutputGenerator
        self.output_generator.format_location_description(location_data)
```

### 7. Enhanced Rich Text Utilities (`src/game_loop/core/rich_utils.py`)

```python
from typing import Dict, Any
from rich.text import Text
from rich.style import Style

class RichTextUtils:
    """Utilities for enhanced Rich text formatting."""

    @staticmethod
    def create_action_style(action_type: str) -> Style:
        """Create style based on action type."""

    @staticmethod
    def format_game_entity(entity_type: str, name: str) -> Text:
        """Format game entities (NPCs, objects, locations) with consistent styling."""

    @staticmethod
    def create_status_indicator(status: str) -> Text:
        """Create status indicators with appropriate colors and symbols."""

    @staticmethod
    def format_timestamp(timestamp: str) -> Text:
        """Format timestamps for game events."""
```

## File Structure Changes

```
src/game_loop/
├── core/
│   ├── game_loop.py           # Modified to use OutputGenerator
│   ├── output_generator.py    # New - Main output system
│   ├── response_formatter.py  # New - Rich text formatting
│   ├── template_manager.py    # New - Template management
│   ├── streaming_handler.py   # New - Streaming responses
│   └── rich_utils.py          # New - Rich text utilities
└── templates/                 # New - Jinja2 templates
    ├── actions/
    ├── locations/
    ├── messages/
    └── responses/
```

## Dependencies to Add

Add to `pyproject.toml`:
```toml
[tool.poetry.dependencies]
jinja2 = "^3.1.2"
# rich is already included
```

## Testing Strategy

### Unit Tests (`tests/unit/core/`)

1. **test_output_generator.py**
   - Test response generation from ActionResult
   - Test template rendering
   - Test error handling

2. **test_response_formatter.py**
   - Test different formatting methods
   - Test Rich text output consistency
   - Test formatting edge cases

3. **test_template_manager.py**
   - Test template loading
   - Test context rendering
   - Test template selection logic

4. **test_streaming_handler.py**
   - Test streaming functionality
   - Test live updates
   - Test streaming interruption

### Integration Tests (`tests/integration/core/`)

1. **test_output_integration.py**
   - Test OutputGenerator with GameLoop
   - Test template rendering with real data
   - Test streaming with mock LLM responses

## Implementation Steps

### Step 1: Dependencies and Structure
1. Add Jinja2 dependency to pyproject.toml
2. Create templates directory structure
3. Install dependencies: `poetry install`

### Step 2: Core Classes
1. Implement TemplateManager class
2. Implement ResponseFormatter class
3. Implement StreamingHandler class
4. Create basic template files

### Step 3: OutputGenerator
1. Implement OutputGenerator class
2. Create Rich text utilities
3. Add integration methods

### Step 4: GameLoop Integration
1. Modify GameLoop to use OutputGenerator
2. Update LocationDisplay integration
3. Replace existing console.print calls

### Step 5: Templates
1. Create comprehensive template set
2. Test template rendering with sample data
3. Add template validation

### Step 6: Testing
1. Write unit tests for all components
2. Create integration tests
3. Test with actual game scenarios

## Verification Procedures

### Manual Testing
1. **Sample Output Generation**
   - Generate outputs for different action types (movement, interaction, dialogue)
   - Verify rich text formatting displays correctly in terminal
   - Test with various ActionResult scenarios

2. **Template Rendering**
   - Test template rendering with sample context data
   - Verify Jinja2 filters work correctly with Rich markup
   - Test template selection for different action types

3. **Streaming Responses**
   - Test streaming handler with mock LLM responses
   - Verify live updates work smoothly
   - Test streaming interruption and cleanup

### Automated Testing
1. Run unit test suite: `poetry run pytest tests/unit/core/test_output_*.py`
2. Run integration tests: `poetry run pytest tests/integration/core/test_output_integration.py`
3. Test template validation: `poetry run python -m game_loop.core.template_manager --validate`

### Quality Checks
1. **Rich Text Rendering**: Confirm all Rich markup renders correctly
2. **Template Consistency**: Verify all templates follow consistent formatting
3. **Performance**: Test streaming response latency
4. **Error Handling**: Test with malformed templates and invalid data

## Success Criteria

- [x] OutputGenerator class properly generates formatted responses from ActionResult objects
- [x] Rich text formatting enhances readability and visual appeal
- [x] Template system provides consistent messaging across all game interactions
- [x] Streaming handler displays LLM responses in real-time
- [x] All existing output functionality maintains compatibility
- [x] Template rendering works correctly with game context data
- [x] Error messages are properly formatted and user-friendly
- [x] Integration with GameLoop preserves existing functionality
- [x] All tests pass and provide adequate coverage
- [x] Code follows project linting standards (black, ruff, mypy)

## Notes

- Leverage existing Rich console integration in GameLoop and LocationDisplay
- Maintain backward compatibility with current output patterns
- Templates should be flexible enough to handle various game scenarios
- Streaming should be smooth and not interfere with game responsiveness
- Consider terminal width and wrapping for better display
- Ensure proper error handling for template loading and rendering failures
- Use Rich's built-in styles and themes for consistency

## Future Enhancements (Post-Commit)

- Add theme support for different visual styles
- Implement output caching for performance
- Add customizable output preferences
- Create output export functionality
- Add accessibility features for screen readers
- Implement output localization support

## Implementation Status: COMPLETED ✅

**Completion Date**: May 30, 2025

### What Was Implemented

All planned components have been successfully implemented and tested:

1. **Core Classes** ✅
   - `OutputGenerator`: Central output management system
   - `ResponseFormatter`: Rich text formatting with support for both objects and dictionaries
   - `TemplateManager`: Jinja2-based template system with custom filters
   - `StreamingHandler`: Real-time response streaming capabilities

2. **Template System** ✅
   - Complete template directory structure (`templates/`)
   - Action, dialogue, location, message, and response templates
   - Jinja2 integration with Rich markup support
   - Custom filters for highlight and color formatting

3. **Rich Text Integration** ✅
   - Enhanced console output with colors, panels, and tables
   - Consistent formatting across all output types
   - Location descriptions, error messages, system messages
   - Action feedback with success/failure indicators

4. **Testing** ✅
   - 15/15 OutputGenerator unit tests passing
   - Integration test working with real ActionResult objects
   - Template rendering and fallback mechanisms tested
   - All 214 tests in the complete test suite passing

5. **Dependencies** ✅
   - Jinja2 dependency added to pyproject.toml
   - All imports and type annotations properly configured
   - Code passes ruff, black, and mypy linting standards

### Key Features Working

- **ActionResult Processing**: Converts game actions into formatted rich text output
- **Template-Based Messaging**: Consistent formatting using Jinja2 templates
- **Rich Text Formatting**: Colors, panels, tables, and styled text output
- **Location Display**: Enhanced location descriptions with exits and objects
- **Error Handling**: Graceful fallbacks when templates fail or data is missing
- **Streaming Support**: Real-time display of LLM responses with live updates
- **Type Safety**: Full type annotation coverage with mypy compliance
- **Backward Compatibility**: Existing GameLoop functionality preserved

### Test Results

```
✅ All Unit Tests: 214 passed, 1 skipped
✅ Core Module Tests: 39/39 passing
✅ OutputGenerator Tests: 15/15 passing
✅ Integration Test: Working perfectly
✅ Linting: ruff, black, mypy all passing
```

The OutputGenerator system is now production-ready and fully integrated into the Game Loop text adventure framework.

## Notes
