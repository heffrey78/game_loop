# Commit 30: Static Rules Engine Implementation Plan

Based on the Game Loop Implementation Plan, this document outlines the specific implementation details for Commit 30: Static Rules Engine.

## Overview

Commit 30 is part of Phase 6: Rules and Evolution Systems. This commit focuses on implementing a foundational static rules engine that will serve as the basis for game rule management and validation.

## Implementation Goals

### Primary Objectives
- Implement RulesEngine for game rules
- Create rule definition loading
- Add rule application system
- Implement rule priority handling
- Create rule conflict resolution

### Verification Criteria
- Load sample rule definitions
- Test application with different priorities
- Verify conflict resolution produces consistent results

## Technical Requirements

### Core Components to Implement

#### 1. RulesEngine Class (`src/game_loop/core/rules/engine.py`)
```python
class RulesEngine:
    """Core rules engine for managing game rules and their application."""
    
    def __init__(self, config: GameConfig)
    async def load_rules(self, rule_definitions: List[RuleDefinition])
    async def apply_rules(self, context: RuleContext) -> RuleResult
    def get_applicable_rules(self, action_type: str, context: RuleContext) -> List[Rule]
    def resolve_conflicts(self, conflicting_rules: List[Rule]) -> Rule
```

#### 2. Rule Models (`src/game_loop/core/rules/models.py`)
```python
@dataclass
class RuleDefinition:
    id: str
    name: str
    description: str
    conditions: List[RuleCondition]
    actions: List[RuleAction]
    priority: int
    enabled: bool = True

@dataclass
class RuleCondition:
    type: str  # "location", "inventory", "player_state", "time", etc.
    operator: str  # "equals", "contains", "greater_than", etc.
    value: Any
    target: str

@dataclass
class RuleAction:
    type: str  # "allow", "deny", "modify", "message", etc.
    parameters: Dict[str, Any]

@dataclass
class RuleContext:
    player_state: PlayerState
    current_location: Location
    action_type: str
    action_target: Optional[str]
    game_state: GameState
    
@dataclass
class RuleResult:
    allowed: bool
    modifications: Dict[str, Any]
    messages: List[str]
    applied_rules: List[str]
```

#### 3. Rule Loader (`src/game_loop/core/rules/loader.py`)
```python
class RuleLoader:
    """Loads rule definitions from various sources."""
    
    async def load_from_yaml(self, file_path: str) -> List[RuleDefinition]
    async def load_from_database(self) -> List[RuleDefinition]
    def validate_rule_definition(self, rule_def: dict) -> RuleDefinition
```

#### 4. Rule Validator (`src/game_loop/core/rules/validator.py`)
```python
class RuleValidator:
    """Validates rule conditions and actions."""
    
    def validate_condition(self, condition: RuleCondition, context: RuleContext) -> bool
    def validate_rule_syntax(self, rule_def: RuleDefinition) -> List[str]
    def check_rule_conflicts(self, rules: List[RuleDefinition]) -> List[RuleConflict]
```

#### 5. Conflict Resolution (`src/game_loop/core/rules/conflict_resolver.py`)
```python
class ConflictResolver:
    """Resolves conflicts between multiple applicable rules."""
    
    def resolve_by_priority(self, rules: List[Rule]) -> Rule
    def resolve_by_specificity(self, rules: List[Rule]) -> Rule
    def resolve_by_recency(self, rules: List[Rule]) -> Rule
```

### Database Schema Updates

#### Rules Table (`src/game_loop/database/migrations/030_rules_system.sql`)
```sql
CREATE TABLE IF NOT EXISTS rule_definitions (
    rule_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    rule_data JSONB NOT NULL,
    priority INTEGER NOT NULL DEFAULT 0,
    enabled BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_rule_definitions_priority ON rule_definitions(priority DESC);
CREATE INDEX idx_rule_definitions_enabled ON rule_definitions(enabled);
```

### Configuration Files

#### Sample Rules (`rules/sample_rules.yaml`)
```yaml
rules:
  - id: "movement_basic"
    name: "Basic Movement Rules"
    description: "Fundamental movement validation"
    priority: 100
    conditions:
      - type: "location"
        operator: "has_connection"
        target: "destination"
    actions:
      - type: "allow"
        parameters: {}

  - id: "inventory_weight"
    name: "Inventory Weight Limit"
    description: "Prevent carrying too much"
    priority: 200
    conditions:
      - type: "inventory"
        operator: "total_weight_less_than"
        value: 50
    actions:
      - type: "deny"
        parameters:
          message: "You cannot carry any more items."

  - id: "locked_door"
    name: "Locked Door Access"
    description: "Require key for locked doors"
    priority: 300
    conditions:
      - type: "object_state"
        target: "door"
        operator: "equals"
        value: "locked"
      - type: "inventory"
        operator: "contains"
        value: "key"
    actions:
      - type: "modify"
        parameters:
          unlock_door: true
          message: "You unlock the door with your key."
```

### Integration Points

#### 1. GameLoop Integration
- Integrate RulesEngine into main game loop
- Add rule checking before action execution
- Handle rule results and modifications

#### 2. Action Processor Integration
- Update all action processors to consult rules engine
- Implement rule-based action validation
- Apply rule modifications to action results

#### 3. Database Integration
- Create repository for rule persistence
- Implement rule caching for performance
- Add rule audit logging

## Implementation Steps

### Step 1: Core Models and Interfaces
1. Create rule models with proper type hints
2. Define interfaces for rule components
3. Add validation for model data
4. Write unit tests for models

### Step 2: Rule Engine Implementation
1. Implement basic RulesEngine class
2. Add rule loading and caching
3. Implement rule application logic
4. Add priority-based rule ordering

### Step 3: Rule Validation System
1. Create condition validators
2. Implement rule syntax checking
3. Add conflict detection logic
4. Write comprehensive validation tests

### Step 4: Conflict Resolution
1. Implement priority-based resolution
2. Add specificity-based resolution
3. Create configurable resolution strategies
4. Test conflict resolution scenarios

### Step 5: Database Integration
1. Create database schema
2. Implement rule repository
3. Add rule persistence and retrieval
4. Write database integration tests

### Step 6: Game Loop Integration
1. Integrate rules engine with action processors
2. Add rule checking to game loop
3. Implement rule result handling
4. Test end-to-end rule application

## Testing Strategy

### Unit Tests
- Test rule model validation
- Test individual rule conditions
- Test conflict resolution algorithms
- Test rule loading from various sources

### Integration Tests
- Test rules engine with real game scenarios
- Test database persistence and retrieval
- Test rule application in action processors
- Test rule conflict resolution in practice

### Performance Tests
- Benchmark rule application speed
- Test rule caching effectiveness
- Measure database query performance
- Profile memory usage with large rule sets

## Files to Create/Modify

### New Files
```
src/game_loop/core/rules/
├── __init__.py
├── engine.py
├── models.py
├── loader.py
├── validator.py
└── conflict_resolver.py

src/game_loop/database/models/rules.py
src/game_loop/database/repositories/rule_repository.py
src/game_loop/database/migrations/030_rules_system.sql

rules/
├── sample_rules.yaml
└── basic_game_rules.yaml

tests/unit/core/rules/
├── test_engine.py
├── test_models.py
├── test_loader.py
├── test_validator.py
└── test_conflict_resolver.py

tests/integration/rules/
├── test_rules_integration.py
└── test_database_rules.py
```

### Files to Modify
```
src/game_loop/core/game_loop.py
src/game_loop/core/command_handlers/*.py
src/game_loop/config/models.py
pyproject.toml (if new dependencies needed)
```

## Success Criteria

### Functional Requirements
- ✅ Rules can be loaded from YAML and database
- ✅ Rule conditions are evaluated correctly
- ✅ Rule priorities are respected in application
- ✅ Conflicts are resolved consistently
- ✅ Rule results modify game behavior appropriately

### Quality Requirements
- ✅ All code passes linting (black, ruff, mypy)
- ✅ Test coverage > 90% for rules components
- ✅ Performance benchmarks meet targets
- ✅ Documentation is complete and accurate

### Integration Requirements
- ✅ Rules integrate seamlessly with existing action processors
- ✅ Database operations work reliably
- ✅ No regressions in existing functionality
- ✅ Error handling is robust and user-friendly

## Dependencies

### Internal Dependencies
- Game state management system (Commit 9)
- Action processing system (Commits 18-23)
- Database infrastructure (Commits 3, 10)

### External Dependencies
- PyYAML (for rule definition loading)
- Possibly additional validation libraries

## Risk Mitigation

### Performance Risks
- Implement rule caching to avoid repeated evaluations
- Use database indexing for rule queries
- Profile rule application in game loop

### Complexity Risks
- Start with simple rule types and expand gradually
- Maintain clear separation between rule engine and game logic
- Use comprehensive testing to catch edge cases

### Integration Risks
- Test with existing action processors early
- Maintain backward compatibility during integration
- Use feature flags for gradual rollout

This implementation plan provides a comprehensive roadmap for implementing the Static Rules Engine as outlined in Commit 30 of the Game Loop Implementation Plan.