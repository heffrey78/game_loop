# Commit 8: NLP Processing Pipeline Implementation Plan

## Overview

This commit will implement the Natural Language Processing (NLP) Pipeline to support natural language interactions with the CLI. The system will enable understanding more varied input patterns (e.g., "Pick up the stick" and "Grab the stick" should be interpreted the same way).

## Prerequisites

- Ollama client is already implemented (checked)
- Config management system is in place (checked)
- Basic input processing exists (InputProcessor)

## Components to Implement

### 1. Prompt Templates

**Location**: `src/game_loop/llm/prompts/`

- `intent_recognition.txt`: For classifying player intent
- `disambiguation.txt`: For resolving ambiguous commands
- `action_extraction.txt`: For extracting action details from natural language

### 2. NLP Processor Class

**Location**: `src/game_loop/llm/nlp_processor.py`

```python
class NLPProcessor:
    """
    Processes natural language input using LLM to extract intent and entities.
    Works alongside InputProcessor to handle complex language structures.
    """

    def __init__(self, config_manager=None, ollama_client=None):
        """
        Initialize the NLP processor with config and client.
        """

    async def process_input(self, user_input, game_context=None):
        """
        Process natural language input and extract structured command.
        """

    async def extract_intent(self, normalized_input, game_context=None):
        """
        Use LLM to recognize user intent from natural language.
        """

    async def disambiguate_input(self, normalized_input, possible_interpretations, game_context=None):
        """
        Resolve ambiguous commands when multiple interpretations are possible.
        """

    async def generate_semantic_query(self, intent_data):
        """
        Generate semantic search query based on extracted intent.
        """

    def _normalize_input(self, user_input):
        """
        Prepare input for NLP processing.
        """
```

### 3. Enhanced Input Processor

**Location**: `src/game_loop/core/enhanced_input_processor.py`

```python
class EnhancedInputProcessor(InputProcessor):
    """
    Input processor with natural language understanding capabilities.
    Combines pattern matching and NLP approaches.
    """

    def __init__(self, config_manager=None, console=None):
        """
        Initialize with both pattern matching and NLP capabilities.
        """

    async def process_input(self, user_input, game_state=None):
        """
        Process user input with NLP, falling back to pattern matching.
        """

    async def extract_game_context(self, game_state):
        """
        Extract relevant context from game state for NLP processing.
        """
```

### 4. Command Mapper

**Location**: `src/game_loop/core/command_mapper.py`

```python
class CommandMapper:
    """
    Maps NLP intents to game commands and handles synonym resolution.
    """

    def __init__(self):
        """
        Initialize with action mappings and synonyms.
        """

    def map_intent_to_command(self, intent_data):
        """
        Convert NLP intent data to a ParsedCommand.
        """

    def get_canonical_action(self, action_text):
        """
        Map various action phrasings to canonical commands.
        """
```

### 5. Conversation Context Manager

**Location**: `src/game_loop/llm/conversation_context.py`

```python
class ConversationContext:
    """
    Tracks conversational context for more natural dialogue interactions.
    """

    def __init__(self, max_history=5):
        """
        Initialize conversation context with history limit.
        """

    def add_exchange(self, user_input, system_response):
        """
        Add a conversation exchange to history.
        """

    def get_recent_context(self, max_tokens=None):
        """
        Get recent conversation history formatted for context.
        """

    def clear_context(self):
        """
        Reset conversation context.
        """
```

## Implementation Steps

1. **Create Prompt Templates**
   - Implement intent recognition prompt based on docs/llm_prompts.md
   - Implement disambiguation prompt
   - Add action extraction prompt

2. **Create NLPProcessor Class**
   - Implement basic structure and initialization
   - Create intent recognition with LLM
   - Add entity extraction methods
   - Implement disambiguation logic
   - Create semantic search query generation

3. **Command Mapping System**
   - Create action synonym mappings
   - Implement object resolution logic
   - Add context-aware mapping

4. **Enhance InputProcessor**
   - Create EnhancedInputProcessor class
   - Implement hybrid processing (pattern+NLP)
   - Add context extraction for NLP
   - Ensure fallback to pattern matching

5. **Conversation Context Management**
   - Create ConversationContext class
   - Implement context tracking
   - Add history management
   - Create context formatting for prompts

6. **Integration with Game Loop**
   - Update GameLoop to use EnhancedInputProcessor
   - Pass game context to input processor
   - Handle NLP processing results

## Testing Plan

1. **Unit Tests**
   - `test_nlp_processor.py`: Test intent recognition, disambiguation, etc.
   - `test_enhanced_input_processor.py`: Test NLP integration with pattern matching
   - `test_command_mapper.py`: Test synonym resolution and command mapping
   - `test_conversation_context.py`: Test context management

2. **Integration Tests**
   - `test_nlp_integration.py`: Test full NLP pipeline with Ollama
   - `test_input_chain.py`: Test the entire input processing chain

3. **Test Vectors**
   - Create varied phrasings for the same commands
   - Test with ambiguous inputs
   - Test with contextual inputs that require game state
   - Create examples of conversational inputs

## Success Criteria

1. **Functional Requirements**
   - Commands like "Pick up the stick" and "Grab the stick" produce the same result
   - System recognizes player intent from natural language
   - Ambiguous commands trigger disambiguation
   - Basic conversation handling works for NPCs

2. **Technical Requirements**
   - Integration with Ollama works reliably
   - Fallback to pattern matching when appropriate
   - Input processing remains responsive (<500ms when possible)
   - Graceful error handling when LLM fails

## Metrics and Validation

1. **Intent Recognition Accuracy**
   - >90% accuracy on test command set
   - Track average processing time

2. **Command Mapping Quality**
   - >95% correct mapping of recognized intents
   - Verify synonym handling works correctly

3. **User Experience**
   - Verify error messages are helpful
   - Ensure disambiguation questions are clear

## Documentation Updates

1. **README.md**
   - Add section on NLP capabilities
   - Document command variation support
   - Note conversation features

2. **Code Documentation**
   - Add docstrings to all new classes and methods
   - Include examples in key method docstrings
   - Document the NLP pipeline flow

## Implementation Schedule

1. Day 1: Create prompt templates and initial NLPProcessor structure
2. Day 2: Implement intent recognition and command mapping
3. Day 3: Develop EnhancedInputProcessor and integration
4. Day 4: Add conversation context and disambiguation
5. Day 5: Write tests and documentation, final integration

## Future Enhancements (Post-Commit)

1. Caching of similar requests for performance
2. More sophisticated disambiguation strategies
3. Enhanced conversation memory and context
4. Integration with world model for better context
5. Learning from player patterns over time
