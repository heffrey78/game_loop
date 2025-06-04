# Commit 18: Action Type Determination

## Overview

This commit establishes the action type determination system that serves as the foundation for processing different types of player actions in the game. Building upon the semantic search and NLP processing systems from previous commits, this implementation creates a sophisticated action classification pipeline that combines rule-based patterns with LLM-powered fallbacks to accurately categorize player inputs and route them to appropriate action processors.

## Goals

1. Create a comprehensive action type classification system
2. Implement rule-based action categorization with clear patterns
3. Add LLM-based action type fallbacks for ambiguous inputs
4. Develop an action routing system for dispatching to appropriate processors
5. Build a pluggable architecture for future action type extensions
6. Create robust testing infrastructure for action classification
7. Integrate with existing NLP and semantic search systems

## Implementation Tasks

### 1. Action Type Classifier (`src/game_loop/core/actions/action_classifier.py`)

**Purpose**: Core classification system that determines the type of action from player input.

**Key Components**:
- Action type enumeration and definitions
- Rule-based pattern matching
- LLM-based classification fallback
- Confidence scoring for classifications
- Classification result formatting

**Methods to Implement**:
```python
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

class ActionType(Enum):
    MOVEMENT = "movement"           # go, walk, run, move
    OBJECT_INTERACTION = "object"   # take, drop, use, examine
    QUEST = "quest"                 # complete, accept, check
    CONVERSATION = "conversation"   # talk, ask, tell
    QUERY = "query"                 # what, where, who, how
    SYSTEM = "system"               # save, load, help, settings
    PHYSICAL = "physical"           # push, pull, climb, jump
    OBSERVATION = "observation"     # look, listen, smell, feel
    UNKNOWN = "unknown"             # cannot determine type

@dataclass
class ActionClassification:
    action_type: ActionType
    confidence: float
    primary_verb: Optional[str]
    target_entity: Optional[str]
    modifiers: List[str]
    raw_input: str
    classification_method: str  # "rule", "llm", "hybrid"
    metadata: Dict[str, Any]

class ActionTypeClassifier:
    def __init__(self, nlp_processor, llm_client, pattern_config=None):
        self.nlp_processor = nlp_processor
        self.llm_client = llm_client
        self.pattern_config = pattern_config or self._load_default_patterns()
        self._pattern_cache = {}
        self._compile_patterns()

    async def classify_action(self, input_text: str, context: Dict[str, Any] = None) -> ActionClassification:
        """Primary method to classify action from player input"""

    async def classify_with_rules(self, input_text: str, parsed_input: Dict[str, Any]) -> Optional[ActionClassification]:
        """Attempt classification using rule-based patterns"""

    async def classify_with_llm(self, input_text: str, parsed_input: Dict[str, Any],
                               context: Dict[str, Any] = None) -> ActionClassification:
        """Use LLM for classification when rules are insufficient"""

    async def hybrid_classification(self, input_text: str, context: Dict[str, Any] = None) -> ActionClassification:
        """Combine rule-based and LLM classification for best results"""

    def extract_action_components(self, input_text: str, parsed_input: Dict[str, Any]) -> Dict[str, Any]:
        """Extract verb, target, and modifiers from input"""

    def calculate_confidence(self, classification_method: str, matches: List[str],
                           llm_confidence: Optional[float] = None) -> float:
        """Calculate confidence score for classification"""

    def _load_default_patterns(self) -> Dict[str, List[str]]:
        """Load default action patterns for rule-based matching"""

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficient matching"""

    def _match_patterns(self, input_text: str, parsed_input: Dict[str, Any]) -> List[Tuple[ActionType, float]]:
        """Match input against compiled patterns"""

    def _normalize_input(self, input_text: str) -> str:
        """Normalize input for consistent pattern matching"""
```

### 2. Action Pattern Configuration (`src/game_loop/core/actions/patterns.py`)

**Purpose**: Define and manage action patterns for rule-based classification.

**Key Components**:
- Pattern definitions for each action type
- Synonym mapping for verbs
- Context-aware pattern modifications
- Pattern priority management
- Custom pattern registration

**Classes to Implement**:
```python
class ActionPatternManager:
    def __init__(self):
        self.verb_patterns = {}
        self.context_patterns = {}
        self.synonym_map = {}
        self._initialize_patterns()

    def get_patterns_for_type(self, action_type: ActionType) -> List[Dict[str, Any]]:
        """Get all patterns associated with an action type"""

    def register_pattern(self, action_type: ActionType, pattern: Dict[str, Any],
                        priority: int = 5) -> None:
        """Register a custom pattern for action classification"""

    def get_verb_synonyms(self, verb: str) -> List[str]:
        """Get all synonyms for a given verb"""

    def apply_context_modifiers(self, patterns: List[Dict[str, Any]],
                              context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Modify patterns based on current game context"""

    def _initialize_patterns(self) -> None:
        """Initialize default action patterns"""
        self.verb_patterns = {
            ActionType.MOVEMENT: {
                "primary_verbs": ["go", "walk", "run", "move", "travel", "head"],
                "prepositions": ["to", "towards", "into", "through", "across"],
                "patterns": [
                    r"^(go|walk|run|move|travel|head)\s+(to|towards|into|through)?\s*(.+)",
                    r"^(north|south|east|west|up|down|in|out)$"
                ]
            },
            ActionType.OBJECT_INTERACTION: {
                "primary_verbs": ["take", "get", "pick", "drop", "use", "examine", "look"],
                "patterns": [
                    r"^(take|get|pick up|grab)\s+(.+)",
                    r"^(drop|put down|discard)\s+(.+)",
                    r"^(use|apply|activate)\s+(.+?)(\s+on\s+(.+))?$",
                    r"^(examine|inspect|look at)\s+(.+)"
                ]
            },
            # Additional patterns for other action types...
        }

    def validate_pattern(self, pattern: Dict[str, Any]) -> bool:
        """Validate a pattern structure"""

    def get_pattern_priority(self, pattern: Dict[str, Any]) -> int:
        """Get priority for pattern matching order"""
```

### 3. Action Router (`src/game_loop/core/actions/action_router.py`)

**Purpose**: Route classified actions to appropriate processors.

**Key Components**:
- Processor registration system
- Action dispatch logic
- Pre/post processing hooks
- Error handling and fallbacks
- Performance monitoring

**Methods to Implement**:
```python
class ActionRouter:
    def __init__(self, game_state_manager):
        self.game_state_manager = game_state_manager
        self.processors = {}
        self.pre_processors = []
        self.post_processors = []
        self._routing_metrics = {}

    def register_processor(self, action_type: ActionType, processor: Any) -> None:
        """Register a processor for a specific action type"""

    def register_pre_processor(self, processor: Callable) -> None:
        """Register a pre-processing hook for all actions"""

    def register_post_processor(self, processor: Callable) -> None:
        """Register a post-processing hook for all actions"""

    async def route_action(self, classification: ActionClassification,
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Route classified action to appropriate processor"""

    async def _execute_pre_processors(self, classification: ActionClassification,
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all pre-processing hooks"""

    async def _execute_processor(self, classification: ActionClassification,
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the main action processor"""

    async def _execute_post_processors(self, result: Dict[str, Any],
                                      classification: ActionClassification) -> Dict[str, Any]:
        """Execute all post-processing hooks"""

    def _handle_processor_error(self, error: Exception,
                              classification: ActionClassification) -> Dict[str, Any]:
        """Handle errors during action processing"""

    def _update_routing_metrics(self, action_type: ActionType,
                              execution_time: float, success: bool) -> None:
        """Update performance metrics for routing"""

    def get_routing_metrics(self) -> Dict[str, Any]:
        """Get routing performance metrics"""

    def _validate_processor(self, processor: Any) -> bool:
        """Validate that processor implements required interface"""
```

### 4. LLM Action Classification (`src/game_loop/core/actions/llm_classifier.py`)

**Purpose**: Specialized LLM-based classification for complex or ambiguous inputs.

**Key Components**:
- Prompt templates for action classification
- Context injection for better classification
- Confidence calibration
- Classification explanation generation
- Fallback handling

**Methods to Implement**:
```python
class LLMActionClassifier:
    def __init__(self, llm_client, prompt_manager):
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        self._classification_cache = {}
        self._load_prompts()

    async def classify_action(self, input_text: str, context: Dict[str, Any] = None,
                             previous_attempts: List[ActionClassification] = None) -> ActionClassification:
        """Classify action using LLM with context awareness"""

    async def get_classification_explanation(self, input_text: str,
                                           classification: ActionClassification) -> str:
        """Generate explanation for why action was classified this way"""

    async def disambiguate_action(self, input_text: str,
                                 possible_types: List[ActionType]) -> ActionType:
        """Use LLM to disambiguate between multiple possible action types"""

    def build_classification_prompt(self, input_text: str, context: Dict[str, Any]) -> str:
        """Build prompt for action classification"""

    def parse_llm_response(self, response: str) -> Tuple[ActionType, float, Dict[str, Any]]:
        """Parse LLM response into classification result"""

    def calibrate_confidence(self, raw_confidence: float, response_metadata: Dict[str, Any]) -> float:
        """Calibrate LLM confidence scores based on response characteristics"""

    def _load_prompts(self) -> None:
        """Load prompt templates for classification"""

    def _extract_context_hints(self, context: Dict[str, Any]) -> str:
        """Extract relevant context for better classification"""

    async def _validate_classification(self, classification: ActionClassification,
                                     input_text: str) -> bool:
        """Validate that classification makes sense for input"""
```

### 5. Action Type Integration (`src/game_loop/core/actions/integration.py`)

**Purpose**: Integrate action classification with the main game loop.

**Key Components**:
- Game loop integration points
- State management integration
- NLP pipeline connection
- Search system integration
- Event emission for actions

**Methods to Implement**:
```python
class ActionTypeIntegration:
    def __init__(self, action_classifier, action_router, game_state_manager, search_service):
        self.classifier = action_classifier
        self.router = action_router
        self.game_state = game_state_manager
        self.search = search_service
        self._action_history = []

    async def process_player_input(self, input_text: str, player_id: str) -> Dict[str, Any]:
        """Main entry point for processing player input through action system"""

    async def enhance_classification_with_context(self, initial_classification: ActionClassification,
                                                player_state: Dict[str, Any]) -> ActionClassification:
        """Enhance classification using game context and player state"""

    async def validate_action_feasibility(self, classification: ActionClassification,
                                        context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check if classified action is feasible in current context"""

    async def search_for_action_targets(self, classification: ActionClassification,
                                      context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Use semantic search to find valid targets for action"""

    def update_action_history(self, classification: ActionClassification,
                            result: Dict[str, Any]) -> None:
        """Update player's action history for pattern analysis"""

    async def emit_action_event(self, classification: ActionClassification,
                              result: Dict[str, Any]) -> None:
        """Emit events for other systems to react to actions"""

    def get_recent_actions(self, player_id: str, count: int = 10) -> List[ActionClassification]:
        """Get recent actions for a player"""

    async def _build_action_context(self, player_id: str) -> Dict[str, Any]:
        """Build comprehensive context for action processing"""

    def _should_use_cached_classification(self, input_text: str,
                                        context: Dict[str, Any]) -> bool:
        """Determine if cached classification can be used"""
```

### 6. Action Testing Framework (`src/game_loop/core/actions/testing.py`)

**Purpose**: Comprehensive testing utilities for action classification.

**Key Components**:
- Test case generation
- Classification accuracy metrics
- Performance benchmarking
- Edge case handling
- Regression testing support

**Classes to Implement**:
```python
class ActionClassificationTester:
    def __init__(self, classifier):
        self.classifier = classifier
        self.test_cases = []
        self.results = []

    def add_test_case(self, input_text: str, expected_type: ActionType,
                     context: Dict[str, Any] = None) -> None:
        """Add a test case for classification"""

    async def run_test_suite(self) -> Dict[str, Any]:
        """Run all test cases and collect results"""

    def calculate_accuracy_metrics(self) -> Dict[str, float]:
        """Calculate accuracy, precision, recall for each action type"""

    def generate_confusion_matrix(self) -> Dict[str, Dict[str, int]]:
        """Generate confusion matrix for classification results"""

    async def benchmark_performance(self, num_iterations: int = 1000) -> Dict[str, Any]:
        """Benchmark classification performance"""

    def export_test_results(self, filepath: str) -> None:
        """Export test results for analysis"""

    def load_test_cases_from_file(self, filepath: str) -> None:
        """Load test cases from file"""

    def _generate_edge_cases(self) -> List[Dict[str, Any]]:
        """Generate edge case test inputs"""

    def _analyze_misclassifications(self) -> List[Dict[str, Any]]:
        """Analyze patterns in misclassified inputs"""
```

## File Structure

```
src/game_loop/
├── core/
│   ├── actions/
│   │   ├── __init__.py
│   │   ├── action_classifier.py     # Main classification system
│   │   ├── patterns.py               # Pattern definitions and management
│   │   ├── action_router.py          # Action routing system
│   │   ├── llm_classifier.py         # LLM-based classification
│   │   ├── integration.py            # Game loop integration
│   │   ├── testing.py                # Testing framework
│   │   └── types.py                  # Action type definitions
```

## Testing Strategy

### Unit Tests

1. **Action Classifier Tests** (`tests/unit/core/actions/test_action_classifier.py`):
   - Test rule-based classification accuracy
   - Test LLM fallback triggering
   - Test confidence calculation
   - Test component extraction
   - Test pattern matching

2. **Pattern Manager Tests** (`tests/unit/core/actions/test_patterns.py`):
   - Test pattern registration
   - Test synonym lookup
   - Test context modifiers
   - Test pattern validation
   - Test priority ordering

3. **Action Router Tests** (`tests/unit/core/actions/test_action_router.py`):
   - Test processor registration
   - Test routing logic
   - Test pre/post processing
   - Test error handling
   - Test metrics collection

4. **LLM Classifier Tests** (`tests/unit/core/actions/test_llm_classifier.py`):
   - Test prompt generation
   - Test response parsing
   - Test confidence calibration
   - Test disambiguation
   - Test explanation generation

### Integration Tests

1. **End-to-End Classification** (`tests/integration/actions/test_classification_e2e.py`):
   - Test complete classification pipeline
   - Test with real game context
   - Test edge cases and ambiguous inputs
   - Test performance under load
   - Test classification consistency

2. **Game Loop Integration** (`tests/integration/actions/test_game_loop_integration.py`):
   - Test integration with input processor
   - Test state management updates
   - Test search system integration
   - Test event emission
   - Test action history tracking

3. **Multi-Component Tests** (`tests/integration/actions/test_action_system.py`):
   - Test classifier with router
   - Test with NLP processor
   - Test with semantic search
   - Test context enhancement
   - Test feasibility validation

### Performance Tests

1. **Classification Benchmarks** (`tests/performance/test_action_classification.py`):
   - Test classification speed
   - Test throughput under load
   - Test memory usage
   - Test cache effectiveness
   - Test pattern matching efficiency

2. **Routing Performance** (`tests/performance/test_action_routing.py`):
   - Test routing overhead
   - Test concurrent action processing
   - Test processor execution time
   - Test metric collection impact
   - Test scalability

## Verification Criteria

### Functional Verification
- [ ] All major action types are correctly classified with > 95% accuracy
- [ ] Rule-based classification works for common actions
- [ ] LLM fallback activates appropriately for ambiguous inputs
- [ ] Action routing dispatches to correct processors
- [ ] Classification confidence scores are calibrated properly
- [ ] Edge cases are handled gracefully
- [ ] Context enhances classification accuracy

### Performance Verification
- [ ] Simple action classification completes in < 10ms
- [ ] Complex/LLM classification completes in < 200ms
- [ ] Action routing adds < 5ms overhead
- [ ] System handles 100+ concurrent classifications
- [ ] Memory usage remains stable under sustained load
- [ ] Pattern matching is optimized for speed
- [ ] Caching improves repeated classification performance

### Integration Verification
- [ ] Integrates seamlessly with existing NLP processor
- [ ] Works with semantic search for target validation
- [ ] Updates game state appropriately
- [ ] Emits events for other systems
- [ ] Maintains action history correctly
- [ ] Handles all input types from game loop
- [ ] Provides clear error messages for invalid actions

## Dependencies

### Existing Components
- NLP Processor (from Commit 8)
- Semantic Search Service (from Commit 16)
- Game State Manager (from Commit 9)
- LLM Client (from Commit 4)
- Input Processor (from Commit 7)

### Configuration Updates
- Add action classification patterns configuration
- Add LLM prompt templates for classification
- Add routing configuration for processors
- Add performance thresholds
- Add classification confidence thresholds

## Integration Points

1. **With Input Processor**: Receive normalized input for classification
2. **With NLP Processor**: Use parsed input structure
3. **With Semantic Search**: Validate action targets
4. **With Game State**: Check action feasibility
5. **With Event System**: Emit action events
6. **With Future Processors**: Route to action-specific processors

## Migration Considerations

- Design for extensibility to add new action types
- Create clear interfaces for action processors
- Plan for backward compatibility as patterns evolve
- Consider versioning for classification rules
- Allow for A/B testing of classification strategies

## Code Quality Requirements

- [ ] All code passes black, ruff, and mypy linting
- [ ] Comprehensive docstrings for all public methods
- [ ] Type hints for all function parameters and return values
- [ ] Error handling for all external dependencies
- [ ] Logging for classification decisions and routing
- [ ] Performance monitoring for critical paths
- [ ] Unit test coverage > 90%

## Documentation Updates

- [ ] Create action type classification guide
- [ ] Document pattern syntax and registration
- [ ] Add examples of each action type
- [ ] Create processor implementation guide
- [ ] Document LLM prompt engineering for classification
- [ ] Add troubleshooting guide for common issues
- [ ] Update architecture diagram with action system

## Future Considerations

This action type determination system will serve as the foundation for:
- **Commit 19**: Physical Action Processing (movement, environment interaction)
- **Commit 20**: Object Interaction System (inventory management)
- **Commit 21**: Quest Interaction System (quest progress)
- **Commit 22**: Query and Conversation System (dialogue)
- **Commit 23**: System Command Processing (meta commands)
- **Future**: Complex multi-step action chains
- **Future**: Context-aware action suggestions
- **Future**: Player behavior pattern analysis

The design should be flexible enough to support these future enhancements while maintaining high performance and accuracy for basic action classification.

## Code Review Addendum - Critical Issues Found

**Date**: January 6, 2025
**Reviewer**: Claude Code Review Agent
**Status**: Implementation partially complete, requires fixes before merge

### Critical Issues Requiring Immediate Attention

#### 1. Duplicate ActionType Definition (CRITICAL)
**Location**: `src/game_loop/core/patterns.py:17-29`
**Issue**: The `ActionType` enum is duplicated - it's already defined in `types.py`
**Fix Required**: Remove the duplicate definition and import from `types.py`:
```python
# Remove lines 17-29 in patterns.py and replace with:
from .types import ActionType
```

#### 2. Pattern System Architecture Conflict (HIGH)
**Issue**: Two different pattern management systems exist:
- `ActionPattern` class in `src/game_loop/core/patterns.py` (comprehensive with regex/verbs/synonyms)
- `ActionPatternManager` in `src/game_loop/core/actions/patterns.py` (simpler regex-based)

**Decision Required**: Choose one approach and consolidate. Recommended approach:
- Keep the simpler `ActionPatternManager` from `actions/patterns.py`
- Remove or refactor the complex `ActionPattern` system in `core/patterns.py`
- Ensure all functionality is preserved in the chosen system

#### 3. Import Path Inconsistency (MEDIUM)
**Fixed**: The test file correctly updates import paths from `src.game_loop` to `game_loop`
**Verify**: Ensure all other files use consistent import paths

### Implementation Quality Assessment

#### Positive Aspects ✅
- Well-structured modular architecture
- Comprehensive test coverage (34 tests passing)
- Hybrid classification approach with confidence scoring
- Smart caching implementation with TTL
- Proper type hints throughout
- Good error handling patterns

#### Areas for Improvement 🔧

#### 4. Configuration Management (MEDIUM)
**Issue**: Magic numbers hardcoded throughout
**Fix**: Make confidence thresholds configurable:
```python
# In action_classifier.py, replace hardcoded values:
self.high_confidence_threshold = config.get('high_confidence_threshold', 0.8)
self.rule_confidence_threshold = config.get('rule_confidence_threshold', 0.7)
self.llm_fallback_threshold = config.get('llm_fallback_threshold', 0.6)
```

#### 5. Performance Optimizations (LOW)
**Issue**: Regex patterns compiled on every pattern creation
**Fix**: Implement pattern caching:
```python
# Add to ActionPatternManager:
@lru_cache(maxsize=1000)
def _compile_pattern(self, pattern_str: str) -> Pattern:
    return re.compile(pattern_str, re.IGNORECASE)
```

#### 6. Error Handling Enhancement (LOW)
**Issue**: Generic exception handling in some methods
**Fix**: Add specific exception types:
```python
class ActionClassificationError(Exception):
    """Base exception for action classification errors"""
    pass

class PatternMatchError(ActionClassificationError):
    """Error in pattern matching"""
    pass

class LLMClassificationError(ActionClassificationError):
    """Error in LLM classification"""
    pass
```

### Required Actions Before Merge

#### Immediate (Before Commit)
1. **Fix duplicate ActionType** - Remove from `patterns.py`, import from `types.py`
2. **Resolve pattern system conflict** - Choose and consolidate to one approach
3. **Verify import consistency** - All imports should use `game_loop.*` not `src.game_loop.*`

#### Short Term (Next Sprint)
4. **Make thresholds configurable** - Move magic numbers to configuration
5. **Add pattern caching** - Improve regex compilation performance
6. **Enhance error types** - Add specific exception classes

#### Long Term (Future Commits)
7. **Performance monitoring** - Add detailed metrics collection
8. **A/B testing framework** - Allow testing different classification strategies
9. **Pattern versioning** - Support evolution of classification rules

### Test Coverage Status
- **Unit Tests**: 34 tests passing ✅
- **Integration Tests**: Not yet implemented ⏳
- **Performance Tests**: Not yet implemented ⏳
- **End-to-End Tests**: Not yet implemented ⏳

### Verification Checklist

#### Must Fix Before Merge
- [ ] Remove duplicate ActionType definition
- [ ] Consolidate pattern management systems
- [ ] Verify all import paths are consistent
- [ ] Run full test suite with fixes
- [ ] Verify mypy passes with fixes

#### Recommended Before Merge
- [ ] Add configuration for threshold values
- [ ] Implement pattern compilation caching
- [ ] Add specific exception types
- [ ] Update documentation for final architecture

### Agent Pickup Instructions

**For next agent working on this task:**

1. **Start with critical fixes**: Address the duplicate ActionType and pattern system conflicts first
2. **Consolidation strategy**: Keep `actions/patterns.py` system, remove complex `core/patterns.py` system
3. **Testing approach**: Ensure all existing tests still pass after consolidation
4. **Configuration path**: Add config file in `src/game_loop/config/` for action classification settings
5. **Performance focus**: Implement caching for compiled patterns and LLM responses

**Files to modify:**
- `src/game_loop/core/patterns.py` - Remove duplicate ActionType, simplify or remove entirely
- `src/game_loop/core/actions/action_classifier.py` - Update imports, add configuration
- `src/game_loop/config/` - Add action classification configuration
- `tests/` - Update any tests that depend on removed functionality

**Dependencies to verify:**
- Ensure NLP processor integration still works
- Verify LLM client connection is stable
- Check that semantic search integration points are preserved

The implementation is solid but needs architectural cleanup before it can be safely merged into the main codebase.
