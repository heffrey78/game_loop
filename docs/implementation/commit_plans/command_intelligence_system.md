# Command Intelligence and Suggestion System

## Overview

Implement intelligent command processing that helps players discover available interactions, provides helpful suggestions for failed commands, and guides exploration through contextual hints.

## Current Issues from User Testing

- `write your name on the resignation letter` → "I don't understand that command"
- `examine the books` → "You don't see any 'the books' here"
- `explore the room` → "You don't see any 'room' here"
- `climb a bookcase` → "You cannot go bookcase from here"

Players receive unhelpful error messages without guidance on alternative actions or discovery of what's actually possible.

## Proposed Solution: Intelligent Command Processing

### 1. Command Intent Analysis

**Purpose**: Understand what players are trying to do, even when commands fail.

```python
class CommandIntentAnalyzer:
    """Analyze failed commands to understand player intent."""
    
    INTENT_PATTERNS = {
        'object_interaction': {
            'verbs': ['write', 'inscribe', 'carve', 'mark', 'sign'],
            'patterns': [r'(\w+)\s+(.+)\s+on\s+(.+)', r'(\w+)\s+(.+)\s+with\s+(.+)'],
            'suggestion_type': 'object_modification'
        },
        'collection_examination': {
            'verbs': ['examine', 'look at', 'inspect', 'study'],
            'patterns': [r'examine\s+the\s+(\w+)', r'look\s+at\s+the\s+(\w+)'],
            'suggestion_type': 'collection_interaction'
        },
        'environmental_action': {
            'verbs': ['climb', 'push', 'pull', 'move', 'lift'],
            'patterns': [r'(\w+)\s+(.+)', r'(\w+)\s+the\s+(.+)'],
            'suggestion_type': 'environmental_interaction'
        },
        'exploration': {
            'verbs': ['explore', 'search', 'investigate', 'scan'],
            'patterns': [r'explore\s+(.+)', r'search\s+(.+)'],
            'suggestion_type': 'detailed_exploration'
        }
    }
    
    def analyze_failed_command(self, command_text, context):
        """Analyze a failed command to determine player intent."""
        command_lower = command_text.lower().strip()
        
        for intent_type, intent_data in self.INTENT_PATTERNS.items():
            # Check if command uses verbs associated with this intent
            if any(verb in command_lower for verb in intent_data['verbs']):
                # Try to match patterns to extract objects/targets
                for pattern in intent_data['patterns']:
                    match = re.search(pattern, command_lower)
                    if match:
                        return {
                            'intent_type': intent_type,
                            'suggestion_type': intent_data['suggestion_type'],
                            'verb': match.group(1) if match.groups() else intent_data['verbs'][0],
                            'targets': match.groups()[1:] if len(match.groups()) > 1 else [],
                            'confidence': 0.8
                        }
        
        return {'intent_type': 'unknown', 'confidence': 0.0}
```

### 2. Contextual Suggestion Engine

**Purpose**: Provide helpful suggestions based on current location and available objects.

```python
class ContextualSuggestionEngine:
    """Generate helpful suggestions based on context and failed commands."""
    
    def __init__(self, semantic_search_service):
        self.semantic_search = semantic_search_service
    
    async def generate_suggestions(self, failed_command, intent_analysis, context):
        """Generate helpful suggestions for failed commands."""
        suggestion_type = intent_analysis.get('suggestion_type', 'general')
        
        if suggestion_type == 'object_modification':
            return await self._suggest_object_modifications(intent_analysis, context)
        elif suggestion_type == 'collection_interaction':
            return await self._suggest_collection_interactions(intent_analysis, context)
        elif suggestion_type == 'environmental_interaction':
            return await self._suggest_environmental_actions(intent_analysis, context)
        elif suggestion_type == 'detailed_exploration':
            return await self._suggest_exploration_alternatives(intent_analysis, context)
        else:
            return await self._suggest_general_alternatives(failed_command, context)
    
    async def _suggest_object_modifications(self, intent_analysis, context):
        """Suggest object modification alternatives."""
        targets = intent_analysis.get('targets', [])
        verb = intent_analysis.get('verb', 'use')
        
        suggestions = []
        
        # Check if player has required tools
        inventory = context.get('inventory', {})
        writing_tools = ['pen', 'pencil', 'marker', 'quill']
        has_writing_tool = any(tool in str(inventory).lower() for tool in writing_tools)
        
        if verb in ['write', 'inscribe'] and not has_writing_tool:
            suggestions.append("You might need something to write with. Look for a pen or pencil.")
        
        # Check for similar objects in location
        location_objects = context.get('location_objects', [])
        if targets:
            target = targets[0]
            similar_objects = await self._find_similar_objects(target, location_objects)
            if similar_objects:
                suggestions.append(f"Try '{verb} on {similar_objects[0]}' instead.")
        
        # Suggest examining objects first
        suggestions.append(f"Try examining objects more closely to see what you can do with them.")
        
        return suggestions
    
    async def _suggest_collection_interactions(self, intent_analysis, context):
        """Suggest alternatives for examining collections."""
        targets = intent_analysis.get('targets', [])
        
        suggestions = []
        location_objects = context.get('location_objects', [])
        
        if targets:
            target = targets[0]
            
            # Look for individual items of this type
            individual_items = await self._find_individual_items(target, location_objects)
            if individual_items:
                suggestions.extend([
                    f"Try examining individual items: {', '.join(individual_items[:3])}",
                    f"You might want to 'examine {individual_items[0]}' for more detail."
                ])
            
            # Suggest more general examination
            suggestions.append(f"Try 'look around' to see what specific {target} are available.")
        
        return suggestions
    
    async def _suggest_environmental_actions(self, intent_analysis, context):
        """Suggest environmental interaction alternatives."""
        targets = intent_analysis.get('targets', [])
        verb = intent_analysis.get('verb', 'interact')
        
        suggestions = []
        
        if targets:
            target = targets[0]
            
            # Check if target is mentioned in location description
            location_description = context.get('location_description', '')
            if target.lower() in location_description.lower():
                suggestions.append(f"The {target} is mentioned in the room description. Try examining it first.")
                suggestions.append(f"You might need to 'examine {target}' before you can {verb} it.")
            else:
                suggestions.append(f"I don't see a {target} here. Try 'look around' to see what's available.")
        
        # Suggest examining environment
        suggestions.append("Look around carefully - you might have missed something.")
        suggestions.append("Try examining objects mentioned in the location description.")
        
        return suggestions
    
    async def _suggest_exploration_alternatives(self, intent_analysis, context):
        """Suggest exploration alternatives."""
        suggestions = [
            "Try 'look around' to examine your surroundings in detail.",
            "Use 'examine [object]' to look at specific items you see.",
            "Check your 'inventory' to see what tools you have available."
        ]
        
        # Add location-specific suggestions
        location_objects = context.get('location_objects', [])
        if location_objects:
            suggestions.append(f"You might examine: {', '.join(location_objects[:3])}")
        
        available_exits = context.get('exits', [])
        if available_exits:
            suggestions.append(f"You could explore: {', '.join(available_exits)}")
        
        return suggestions
```

### 3. Smart Error Response Generator

**Purpose**: Generate helpful error messages with actionable suggestions.

```python
class SmartErrorResponseGenerator:
    """Generate intelligent error responses with helpful suggestions."""
    
    def __init__(self, intent_analyzer, suggestion_engine):
        self.intent_analyzer = intent_analyzer
        self.suggestion_engine = suggestion_engine
    
    async def generate_smart_error_response(self, failed_command, context):
        """Generate helpful error response instead of generic failure message."""
        
        # Analyze what the player was trying to do
        intent_analysis = self.intent_analyzer.analyze_failed_command(failed_command, context)
        
        # Generate contextual suggestions
        suggestions = await self.suggestion_engine.generate_suggestions(
            failed_command, intent_analysis, context
        )
        
        # Build helpful response
        response_parts = []
        
        # Acknowledge the attempt
        if intent_analysis['confidence'] > 0.5:
            response_parts.append(self._generate_intent_acknowledgment(intent_analysis))
        else:
            response_parts.append("I don't understand that command.")
        
        # Provide suggestions
        if suggestions:
            response_parts.append("\nHere are some things you might try:")
            for i, suggestion in enumerate(suggestions[:3], 1):
                response_parts.append(f"  {i}. {suggestion}")
        
        # Add discovery hint
        response_parts.append(f"\nType 'help' for more commands or 'look' to examine your surroundings.")
        
        return "\n".join(response_parts)
    
    def _generate_intent_acknowledgment(self, intent_analysis):
        """Generate acknowledgment of what player was trying to do."""
        intent_type = intent_analysis['intent_type']
        
        acknowledgments = {
            'object_interaction': "I can see you're trying to interact with an object.",
            'collection_examination': "It looks like you want to examine a collection of items.",
            'environmental_action': "You're trying to interact with something in the environment.",
            'exploration': "You want to explore and investigate the area."
        }
        
        return acknowledgments.get(intent_type, "I'm not sure what you're trying to do.")
```

### 4. Progressive Discovery System

**Purpose**: Gradually reveal interaction possibilities as players learn.

```python
class ProgressiveDiscoveryManager:
    """Manage progressive revelation of interaction capabilities."""
    
    def __init__(self):
        self.discovered_interactions = {}  # player_id -> set of discovered interaction types
        self.hint_triggers = {}  # context patterns that trigger hints
    
    async def check_for_discovery_opportunities(self, command_result, context):
        """Check if this is a good time to hint at new interactions."""
        player_id = context.get('player_id', 'player')
        
        # Track successful interactions
        if command_result.success:
            await self._record_successful_interaction(player_id, command_result)
        
        # Check for hint opportunities
        hints = await self._generate_discovery_hints(player_id, context)
        
        if hints:
            return self._format_discovery_hints(hints)
        
        return None
    
    async def _generate_discovery_hints(self, player_id, context):
        """Generate hints about undiscovered interactions."""
        discovered = self.discovered_interactions.get(player_id, set())
        hints = []
        
        # Hint about object modification if player has picked up writing tools
        if 'object_modification' not in discovered:
            inventory = context.get('inventory', {})
            if any(tool in str(inventory).lower() for tool in ['pen', 'pencil', 'marker']):
                hints.append("discovery_writing")
        
        # Hint about collection examination if in library-like location
        if 'collection_examination' not in discovered:
            location_name = context.get('location_name', '').lower()
            if any(keyword in location_name for keyword in ['library', 'archive', 'study']):
                hints.append("discovery_collections")
        
        # Hint about environmental interaction if location has interactive elements
        if 'environmental_interaction' not in discovered:
            location_description = context.get('location_description', '').lower()
            interactive_keywords = ['bookcase', 'shelf', 'door', 'window', 'lever', 'switch']
            if any(keyword in location_description for keyword in interactive_keywords):
                hints.append("discovery_environment")
        
        return hints
    
    def _format_discovery_hints(self, hints):
        """Format discovery hints as helpful suggestions."""
        hint_messages = {
            'discovery_writing': "💡 Hint: You can write on objects using 'write [text] on [object]'",
            'discovery_collections': "💡 Hint: Try examining specific books or items rather than collections",
            'discovery_environment': "💡 Hint: You can interact with environmental objects like climbing or pushing them"
        }
        
        return [hint_messages.get(hint, hint) for hint in hints]
```

### 5. Integration with Existing Command Processing

**Purpose**: Integrate smart error handling with existing command flow.

```python
class EnhancedCommandProcessor:
    """Enhanced command processor with intelligent error handling."""
    
    def __init__(self, existing_processor, intent_analyzer, suggestion_engine, discovery_manager):
        self.existing_processor = existing_processor
        self.intent_analyzer = intent_analyzer
        self.suggestion_engine = suggestion_engine
        self.discovery_manager = discovery_manager
        self.error_response_generator = SmartErrorResponseGenerator(
            intent_analyzer, suggestion_engine
        )
    
    async def process_command(self, command_text, context):
        """Process command with intelligent error handling and suggestions."""
        
        # Try existing command processing first
        result = await self.existing_processor.process_command(command_text, context)
        
        # If command failed, provide intelligent error response
        if not result.success and "don't understand" in result.feedback_message.lower():
            smart_response = await self.error_response_generator.generate_smart_error_response(
                command_text, context
            )
            result.feedback_message = smart_response
        
        # Check for discovery opportunities
        discovery_hints = await self.discovery_manager.check_for_discovery_opportunities(
            result, context
        )
        
        if discovery_hints:
            result.feedback_message += "\n\n" + "\n".join(discovery_hints)
        
        return result
```

## Implementation Strategy

### Phase 1: Intent Analysis (Week 1)
1. Implement `CommandIntentAnalyzer` with basic pattern matching
2. Create intent categories for common failed commands
3. Test with user session examples
4. Refine patterns based on testing

### Phase 2: Suggestion Engine (Week 2)
1. Implement `ContextualSuggestionEngine` with location-aware suggestions
2. Integrate with semantic search for object similarity
3. Create suggestion templates for different intent types
4. Test suggestion quality and relevance

### Phase 3: Smart Error Responses (Week 3)
1. Implement `SmartErrorResponseGenerator`
2. Replace generic error messages with helpful responses
3. Integrate with existing command processing flow
4. Add contextual hints based on available actions

### Phase 4: Progressive Discovery (Week 4)
1. Implement `ProgressiveDiscoveryManager`
2. Create discovery hint system
3. Track player learning progression
4. Add achievement-style discovery notifications

## Success Criteria

### Immediate Goals
- `write your name on the resignation letter` provides helpful guidance about writing tools
- `examine the books` suggests examining individual book objects
- `climb a bookcase` explains environmental interaction possibilities
- Error messages include actionable suggestions

### Advanced Goals
- Players discover new interaction types through contextual hints
- Suggestion quality improves based on player behavior patterns
- Failed commands become learning opportunities rather than frustrations
- Command discovery feels natural and progressive

## Integration Points

### Existing Systems
- Integrate with current command processing pipeline
- Use existing semantic search for object similarity
- Leverage location generation system for contextual hints
- Work with observation system for object discovery

### Database Integration
```sql
-- Track player discovery progression
CREATE TABLE player_discovery_progress (
    id UUID PRIMARY KEY,
    player_id UUID NOT NULL,
    interaction_type VARCHAR(100),
    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    discovery_context JSONB
);

-- Store command intent analysis for improvement
CREATE TABLE command_intent_analysis (
    id UUID PRIMARY KEY,
    command_text TEXT,
    intent_type VARCHAR(100),
    confidence_score FLOAT,
    context_data JSONB,
    suggestion_provided TEXT,
    player_response VARCHAR(255)
);
```

This system transforms failed commands from frustrating dead-ends into learning opportunities that guide players toward successful interactions and help them discover the full range of possibilities in the game world.