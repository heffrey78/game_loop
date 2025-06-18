# Conversation-Dialogue Integration Fix Plan

## Problem Statement

**Issue**: The conversation system and dialogue template system are currently disconnected, creating an architectural gap where:

1. **Conversation models** exist and are well-tested for business logic
2. **Dialogue templates** exist but are unused (orphaned code)
3. **Current dialogue rendering** uses hardcoded Rich markup instead of the flexible template system
4. **Missing integration tests** between conversation data and dialogue output

This creates maintenance issues, reduces flexibility, and prevents the template system from being utilized as designed.

## Current Architecture Analysis

### What Works
- ✅ **Conversation Models**: Well-structured data models with comprehensive unit tests
- ✅ **Template Infrastructure**: TemplateManager with Jinja2 support and custom filters
- ✅ **Rich Output**: Current dialogue output works but is hardcoded

### What's Broken
- ❌ **Template Usage**: `dialogue/speech.j2` template is unused
- ❌ **Integration Testing**: No tests verify conversation → template → output flow
- ❌ **Hardcoded Formatting**: ResponseFormatter uses fixed Rich markup
- ❌ **Data Mapping**: No clear mapping between conversation models and template variables

## Implementation Plan

### Phase 1: Template Integration (Priority: High)

#### 1.1 Update OutputGenerator to Use Dialogue Template
**File**: `src/game_loop/core/output_generator.py`

```python
def format_dialogue(
    self, 
    exchange: ConversationExchange, 
    npc_data: dict[str, Any] | None = None
) -> str:
    """Format dialogue using Jinja2 template instead of hardcoded markup."""
    template_context = {
        "text": exchange.message_text,
        "speaker": exchange.speaker_id,
        "npc_data": npc_data,
        "emotion": exchange.emotion,
        "message_type": exchange.message_type.value,
        "metadata": exchange.metadata
    }
    
    # Try template first, fallback to ResponseFormatter if template fails
    rendered = self.template_manager.render_template("dialogue/speech.j2", template_context)
    if rendered:
        return rendered
    else:
        # Fallback to existing hardcoded method
        return self.response_formatter.format_dialogue(exchange.message_text, exchange.speaker_id)
```

#### 1.2 Enhance Dialogue Template
**File**: `templates/dialogue/speech.j2`

```jinja2
{# Enhanced Dialogue Template #}
{% if npc_data and npc_data.name -%}
[bold magenta]{{ npc_data.name }}:[/bold magenta] {% if emotion %}[italic dim]({{ emotion }})[/italic dim] {% endif %}"{{ text | rich_markup }}"
{%- else -%}
[bold magenta]{{ speaker }}:[/bold magenta] {% if emotion %}[italic dim]({{ emotion }})[/italic dim] {% endif %}"{{ text | rich_markup }}"
{%- endif %}
{%- if metadata and metadata.get('internal_thought') %}
[dim]{{ speaker }} thinks: {{ metadata.internal_thought }}[/dim]
{%- endif %}
```

#### 1.3 Add Rich Markup Filter
**File**: `src/game_loop/core/template_manager.py`

```python
def _rich_markup_filter(self, text: str) -> str:
    """
    Custom filter to ensure text is properly escaped for Rich markup.
    
    Args:
        text: Text that may contain Rich markup
        
    Returns:
        Text with proper Rich markup handling
    """
    # Escape any unintended markup while preserving intentional markup
    # This prevents user input from breaking Rich formatting
    return text.replace("[", "\\[").replace("]", "\\]") if isinstance(text, str) else str(text)
```

### Phase 2: Integration Testing (Priority: High)

#### 2.1 Create Template Integration Tests
**File**: `tests/unit/core/conversation/test_dialogue_integration.py`

```python
"""Tests for conversation-dialogue template integration."""

import pytest
from unittest.mock import Mock

from game_loop.core.conversation.conversation_models import ConversationExchange, MessageType
from game_loop.core.output_generator import OutputGenerator
from game_loop.core.template_manager import TemplateManager


class TestDialogueTemplateIntegration:
    """Test integration between conversation models and dialogue templates."""

    @pytest.fixture
    def template_manager(self):
        """Create a template manager for testing."""
        return TemplateManager("templates")

    @pytest.fixture
    def output_generator(self, template_manager):
        """Create an output generator for testing."""
        console = Mock()
        generator = OutputGenerator(console)
        generator.template_manager = template_manager
        return generator

    def test_basic_dialogue_rendering(self, output_generator):
        """Test basic dialogue rendering with template."""
        exchange = ConversationExchange.create_npc_message(
            npc_id="guard_captain",
            message_text="Halt! Who goes there?",
            message_type=MessageType.QUESTION,
            emotion="stern"
        )
        
        npc_data = {"name": "Captain Marcus", "title": "Guard Captain"}
        
        result = output_generator.format_dialogue(exchange, npc_data)
        
        assert "Captain Marcus:" in result
        assert "Halt! Who goes there?" in result
        assert "(stern)" in result
        assert "[bold magenta]" in result

    def test_dialogue_with_player_exchange(self, output_generator):
        """Test dialogue rendering for player messages."""
        exchange = ConversationExchange.create_player_message(
            player_id="player_001",
            message_text="I'm just a traveler seeking shelter.",
            message_type=MessageType.STATEMENT
        )
        
        result = output_generator.format_dialogue(exchange)
        
        assert "player_001:" in result
        assert "I'm just a traveler seeking shelter." in result

    def test_dialogue_template_fallback(self, output_generator):
        """Test fallback when template fails."""
        # Mock template manager to return None (template failure)
        output_generator.template_manager.render_template = Mock(return_value=None)
        
        exchange = ConversationExchange.create_npc_message(
            npc_id="guard",
            message_text="Hello there!"
        )
        
        result = output_generator.format_dialogue(exchange)
        
        # Should fallback to ResponseFormatter
        assert result is not None
        assert "Hello there!" in result

    def test_rich_markup_safety(self, output_generator):
        """Test that user input doesn't break Rich markup."""
        exchange = ConversationExchange.create_player_message(
            player_id="player",
            message_text="I have [bold]magic items[/bold] for sale!",
            message_type=MessageType.STATEMENT
        )
        
        result = output_generator.format_dialogue(exchange)
        
        # Markup should be escaped to prevent formatting injection
        assert "\\[bold\\]" in result or "[bold]" not in result
```

#### 2.2 Add Conversation Output Tests
**File**: `tests/unit/core/conversation/test_conversation_output.py`

```python
"""Tests for conversation system output generation."""

import pytest
from unittest.mock import Mock, patch

from game_loop.core.conversation.conversation_manager import ConversationManager
from game_loop.core.conversation.conversation_models import ConversationResult, ConversationExchange
from game_loop.core.output_generator import OutputGenerator


class TestConversationOutput:
    """Test conversation system output generation."""

    def test_conversation_result_to_output(self):
        """Test converting ConversationResult to formatted output."""
        npc_response = ConversationExchange.create_npc_message(
            npc_id="elder",
            message_text="Welcome, young one. What brings you to our village?",
            emotion="welcoming"
        )
        
        result = ConversationResult.success_result(
            npc_response=npc_response,
            relationship_change=0.1,
            mood_change="friendly"
        )
        
        # Test that result contains proper dialogue exchange
        assert result.npc_response.message_text == "Welcome, young one. What brings you to our village?"
        assert result.npc_response.emotion == "welcoming"

    def test_end_to_end_conversation_flow(self):
        """Test complete conversation flow from input to formatted output."""
        # This would test the full pipeline:
        # User input → ConversationManager → ConversationResult → OutputGenerator → Formatted display
        pass  # Implementation depends on full conversation system integration
```

### Phase 3: Enhanced Template Features (Priority: Medium)

#### 3.1 Create Specialized Dialogue Templates
**Files**: 
- `templates/dialogue/npc_speech.j2` (for NPC-specific formatting)
- `templates/dialogue/player_speech.j2` (for player-specific formatting)  
- `templates/dialogue/system_message.j2` (for system/narrator messages)

#### 3.2 Add Personality-Based Template Selection
**Enhancement**: Modify OutputGenerator to select templates based on NPC personality traits:

```python
def _select_dialogue_template(self, exchange: ConversationExchange, npc_data: dict) -> str:
    """Select appropriate dialogue template based on context."""
    if exchange.message_type == MessageType.SYSTEM:
        return "dialogue/system_message.j2"
    elif npc_data and npc_data.get('personality'):
        # Select template based on personality traits
        if npc_data['personality'].get('formal', 0) > 0.7:
            return "dialogue/formal_speech.j2"
        elif npc_data['personality'].get('casual', 0) > 0.7:
            return "dialogue/casual_speech.j2"
    
    return "dialogue/speech.j2"  # Default template
```

### Phase 4: Deprecation and Cleanup (Priority: Low)

#### 4.1 Deprecate Hardcoded Dialogue Formatting
- Add deprecation warnings to `ResponseFormatter.format_dialogue()`
- Update all callers to use `OutputGenerator.format_dialogue()` instead
- Plan removal of hardcoded formatting in future release

#### 4.2 Template Migration
- Ensure all dialogue formatting goes through template system
- Remove duplicate/redundant formatting code
- Update documentation to reflect template-based approach

## Testing Strategy

### Unit Tests
- ✅ Test template rendering with various conversation models
- ✅ Test template variable mapping from conversation data
- ✅ Test Rich markup safety and escaping
- ✅ Test fallback behavior when templates fail

### Integration Tests
- ✅ Test complete conversation → template → output pipeline
- ✅ Test template selection based on context
- ✅ Test compatibility with existing conversation system

### Regression Tests
- ✅ Ensure existing dialogue output still works
- ✅ Verify no breaking changes to conversation API
- ✅ Test performance impact of template rendering

## Implementation Steps

### Step 1: Quick Fix (1-2 hours)
1. Update `OutputGenerator.format_dialogue()` to use template
2. Add `rich_markup` filter to TemplateManager
3. Create basic integration test

### Step 2: Comprehensive Testing (2-3 hours)
1. Create full test suite for dialogue integration
2. Test edge cases and error scenarios
3. Verify Rich markup safety

### Step 3: Enhanced Features (3-4 hours)
1. Create specialized dialogue templates
2. Add personality-based template selection
3. Enhance template with more context variables

### Step 4: Cleanup (1-2 hours)
1. Add deprecation warnings to old code
2. Update documentation
3. Plan future removal of hardcoded formatting

## Success Criteria

### Functional Requirements
- ✅ `dialogue/speech.j2` template is actively used
- ✅ Conversation models integrate seamlessly with templates
- ✅ Rich markup is properly rendered and safe from injection
- ✅ Fallback mechanism works when templates fail

### Quality Requirements
- ✅ Test coverage > 90% for dialogue integration
- ✅ No performance regression in dialogue rendering
- ✅ All existing functionality preserved
- ✅ Code follows project linting standards

### Architectural Requirements
- ✅ Template system is the primary dialogue formatting mechanism
- ✅ Clear separation between data models and presentation
- ✅ Flexible template selection based on context
- ✅ Easy to extend with new dialogue types

## Risk Mitigation

### Performance Risks
- **Risk**: Template rendering slower than hardcoded formatting
- **Mitigation**: Implement template caching, benchmark performance
- **Fallback**: Keep hardcoded formatting as fallback option

### Compatibility Risks
- **Risk**: Breaking existing dialogue output
- **Mitigation**: Maintain existing API, gradual migration
- **Testing**: Comprehensive regression testing

### Template Risks
- **Risk**: Template errors breaking dialogue system
- **Mitigation**: Robust error handling, fallback to hardcoded formatting
- **Monitoring**: Log template errors for debugging

## Files to Modify

### Core Changes
```
src/game_loop/core/output_generator.py          # Update format_dialogue method
src/game_loop/core/template_manager.py          # Add rich_markup filter
templates/dialogue/speech.j2                    # Enhance template
```

### New Test Files
```
tests/unit/core/conversation/test_dialogue_integration.py
tests/unit/core/conversation/test_conversation_output.py
tests/integration/dialogue/test_template_rendering.py
```

### Documentation Updates
```
docs/architecture/conversation_system.md       # Update architecture docs
docs/templates/dialogue_templates.md           # Document template usage
README.md                                       # Update feature list
```

This fix plan addresses the architectural gap between conversation models and dialogue templates, ensuring the template system is properly utilized while maintaining backward compatibility and system reliability.