# Object Interaction Enhancement Plan

## Overview

Enhance object interaction capabilities to support complex object manipulation, detailed examination with sub-object generation, and meaningful object state changes.

## Current Gaps from User Testing

- `write your name on the resignation letter` - Object modification not supported
- `examine the books` - Collection examination not available  
- `take the scientific instruments` - Complex object selection within descriptions
- `climb a bookcase` - Environmental object interaction

## Proposed Solutions

### 1. Object Modification Handler

**Purpose**: Handle all forms of object state changes and modifications.

```python
class ObjectModificationHandler(BaseCommandHandler):
    """Handle object modifications including writing, opening, adjusting, combining."""
    
    MODIFICATION_TYPES = {
        'write': ['write', 'inscribe', 'mark', 'sign', 'scribble'],
        'open': ['open', 'unlock', 'unseal'],
        'close': ['close', 'lock', 'seal', 'shut'],
        'turn': ['turn', 'rotate', 'twist', 'flip'],
        'adjust': ['adjust', 'set', 'configure', 'tune'],
        'repair': ['repair', 'fix', 'mend', 'restore'],
        'break': ['break', 'smash', 'destroy', 'damage'],
        'combine': ['combine', 'mix', 'merge', 'attach'],
        'clean': ['clean', 'wipe', 'polish', 'scrub']
    }
    
    async def handle(self, command: ParsedCommand) -> ActionResult:
        # Identify modification type and target objects
        # Validate modification is possible (tools, permissions, object state)
        # Execute modification and update object state
        # Generate consequences and state changes
        # Persist changes to world state
```

**Target Interactions**:
- `write "Digi was here" on wall` - Text inscription with persistence
- `open chest` - Container state changes
- `turn page` - Sequential content navigation
- `combine wire with battery` - Object fusion
- `break window` - Destructive environmental changes

### 2. Detailed Examination with Sub-Object Generation

**Purpose**: Integrate with existing world generation to create interactive sub-objects during examination.

```python
class InteractiveExaminationHandler(BaseCommandHandler):
    """Enhanced examination that generates sub-objects and interactive elements."""
    
    def __init__(self, object_generator, world_state_manager):
        self.object_generator = object_generator
        self.world_state_manager = world_state_manager
    
    async def handle(self, command: ParsedCommand) -> ActionResult:
        # Parse examination target (collection vs individual)
        # Generate sub-objects if examining collections
        # Integrate sub-objects into current location
        # Provide progressive examination depth
        # Track examination history for discovery
```

**Sub-Object Generation Examples**:

```python
# Books generate pages
book_object = {
    'name': 'ancient tome',
    'type': 'book',
    'sub_objects': {
        'pages': self._generate_book_pages(book_theme, page_count),
        'bookmark': self._generate_bookmark(),
        'binding': self._generate_binding_details()
    }
}

# Shelves generate individual items
shelf_object = {
    'name': 'library shelf',
    'type': 'furniture',
    'sub_objects': {
        'books': self._generate_shelf_books(shelf_theme),
        'ornaments': self._generate_decorative_items(),
        'hidden_compartment': self._maybe_generate_secret()
    }
}
```

**Target Interactions**:
- `examine books` → Generates individual book objects in location
- `turn to next page` → Navigate through generated book content
- `search shelf` → Reveals hidden sub-objects
- `look behind picture` → Generates concealed objects

### 3. Environmental Object Interaction

**Purpose**: Enable physical interaction with environmental elements described in locations.

```python
class EnvironmentalInteractionHandler(BaseCommandHandler):
    """Handle physical interactions with environmental objects."""
    
    ENVIRONMENTAL_ACTIONS = {
        'climb': ['climb', 'scale', 'ascend'],
        'push': ['push', 'shove', 'press'],
        'pull': ['pull', 'drag', 'yank'],
        'move': ['move', 'shift', 'relocate'],
        'lean': ['lean', 'rest', 'prop'],
        'hide': ['hide', 'conceal', 'duck']
    }
    
    async def handle(self, command: ParsedCommand) -> ActionResult:
        # Parse environmental action and target
        # Check if target exists in location description
        # Validate action feasibility (player stats, object properties)
        # Generate new objects or modify location based on action
        # Update player position or location state
```

**Target Interactions**:
- `climb bookcase` → Reach high objects, new vantage points
- `push desk` → Reveal hidden objects underneath
- `lean against wall` → Discover loose stones, hidden switches
- `hide behind curtain` → Temporary concealment, stealth mechanics

## Integration with Existing Systems

### World Generation Integration

The detailed examination system should work with existing generators:

```python
class ExaminationWorldIntegration:
    """Integrate examination with dynamic world generation."""
    
    async def generate_sub_objects_on_examination(self, parent_object, location):
        # Use ObjectGenerator to create contextually appropriate sub-objects
        # Integrate with location theme and consistency systems
        # Use semantic search to ensure sub-objects fit narrative
        # Cache generated sub-objects for consistency
```

### Object State Persistence

```python
class ObjectStateManager:
    """Manage persistent object modifications and state changes."""
    
    def track_modification(self, object_id, modification_type, details):
        # Store modification in database
        # Update object state in world state
        # Track modification history
        # Handle modification cascading effects
    
    def restore_object_state(self, object_id, location_id):
        # Load all modifications for object
        # Apply modifications to base object state
        # Return current object state
```

## Implementation Approach

### Phase 1: Object Modification Foundation
1. Implement `ObjectModificationHandler` with basic modification types
2. Add object state persistence to database
3. Create modification validation system
4. Test with writing/inscription use cases

### Phase 2: Examination Enhancement  
1. Integrate `InteractiveExaminationHandler` with existing `ObservationCommandHandler`
2. Connect to `ObjectGenerator` for sub-object creation
3. Implement progressive examination depth
4. Add examination history tracking

### Phase 3: Environmental Integration
1. Implement `EnvironmentalInteractionHandler`
2. Parse location descriptions for interactive elements
3. Add environmental modification consequences
4. Create location state change system

## Success Criteria

### Immediate Goals
- `write [text] on [object]` works with appropriate tools and surfaces
- `examine books` generates individual book objects that can be interacted with
- `turn page` allows navigation through book content
- `climb bookcase` enables environmental interaction

### Advanced Goals
- Complex object combinations create new objects
- Environmental modifications persist and affect world state
- Sub-object generation feels natural and contextually appropriate
- Object modification history creates emergent storytelling

## Technical Considerations

### Performance
- Cache generated sub-objects to avoid regeneration
- Lazy-load sub-object details until specifically examined
- Limit sub-object generation depth to prevent infinite recursion

### Consistency
- Ensure sub-objects match parent object theme and location
- Maintain object state consistency across save/load
- Validate modifications don't break game logic

### Extensibility
- Design modification system to support future object types
- Create plugin architecture for new modification types
- Allow custom modification rules through rules engine

This approach enhances object interaction while integrating with existing systems rather than replacing them, providing the foundation for rich, persistent object manipulation that responds to player creativity.