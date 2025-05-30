# Use Handler Refactoring Plan

## Current Implementation Analysis

The current `UseHandler` implementation has several distinct responsibilities:
1. It processes USE commands according to the Strategy pattern (inheriting from `CommandHandler`)
2. It manages multiple usage scenarios using conditional logic:
   - Self-use (using an item by itself)
   - Using an item on another object
   - Putting an item into a container
3. It handles all the validation and processing logic for these different scenarios

This approach works but has several issues:
- The handler is becoming complex with multiple responsibilities
- Adding new usage scenarios requires modifying the main handler class
- The conditional logic creates tight coupling between usage detection and implementation

## Proposed Refactoring Strategy

### 1. Create a Usage Strategy Pattern

Implement a nested strategy pattern for different usage scenarios:

```
CommandHandler (Strategy)
└── UseHandler
    └── UsageHandler (Strategy)
        ├── ContainerUsageHandler
        ├── TargetUsageHandler
        └── SelfUsageHandler
```

### 2. Create a Usage Handler Factory

Implement a factory pattern to create the appropriate usage handler:

```
UsageHandlerFactory
├── createContainerUsageHandler()
├── createTargetUsageHandler()
└── createSelfUsageHandler()
```

### 3. Implementation Plan

1. **Create Base UsageHandler Interface**:
   - Define a common interface for all usage scenarios
   - Include methods for validation and execution

2. **Implement Concrete UsageHandlers**:
   - `ContainerUsageHandler`: For "put X in Y" scenarios
   - `TargetUsageHandler`: For "use X on Y" scenarios
   - `SelfUsageHandler`: For direct item usage

3. **Create UsageHandlerFactory**:
   - Factory to instantiate the appropriate handler based on usage context
   - Move the detection logic (`_is_container_usage`, etc.) into this factory

4. **Refactor UseHandler**:
   - Simplify by delegating to appropriate UsageHandler
   - Focus on common validation and handler selection

5. **Add Extension Points**:
   - Enable registration of new usage handlers
   - Support plugin-based extension

## File Structure

```
/src/game_loop/core/command_handlers/
    ├── base_handler.py              (existing)
    ├── factory.py                   (existing)
    ├── use_handler/
    │   ├── __init__.py              (new)
    │   ├── base.py                  (new: UsageHandler interface)
    │   ├── container_handler.py     (new: container usage)
    │   ├── target_handler.py        (new: target usage)
    │   ├── self_handler.py          (new: self usage)
    │   └── factory.py               (new: UsageHandlerFactory)
    ├── use_handler.py               (modified)
    └── ... (other handlers)
```

## Implementation Details

### UsageHandler Interface (`base.py`)

```python
from abc import ABC, abstractmethod
from game_loop.state.models import ActionResult, InventoryItem, Location, PlayerState

class UsageHandler(ABC):
    """Base class for all item usage handlers."""

    @abstractmethod
    async def validate(self, item_to_use: InventoryItem,
                       player_state: PlayerState,
                       current_location: Location) -> bool:
        """Validate if this usage is possible."""
        pass

    @abstractmethod
    async def handle(self, item_to_use: InventoryItem,
                    target_name: str | None,
                    player_state: PlayerState,
                    current_location: Location) -> ActionResult:
        """Handle the item usage and return the result."""
        pass
```

### UsageHandlerFactory (`factory.py`)

```python
from .base import UsageHandler
from .container_handler import ContainerUsageHandler
from .target_handler import TargetUsageHandler
from .self_handler import SelfUsageHandler

class UsageHandlerFactory:
    """Factory for creating appropriate usage handlers."""

    def get_handler(self, command_target: str | None) -> UsageHandler:
        """Get the appropriate handler for the usage scenario."""
        if command_target and self._is_container_usage(command_target):
            return ContainerUsageHandler()
        elif command_target:
            return TargetUsageHandler()
        else:
            return SelfUsageHandler()

    def _is_container_usage(self, target_name: str) -> bool:
        """Check if this is a 'put X in Y' usage pattern."""
        return " in " in target_name or " into " in target_name
```

### Refactored UseHandler

```python
from game_loop.core.command_handlers.base_handler import CommandHandler
from game_loop.core.input_processor import ParsedCommand
from game_loop.state.models import ActionResult
from .use_handler.factory import UsageHandlerFactory

class UseHandler(CommandHandler):
    """Handler for USE commands in the Game Loop."""

    async def handle(self, command: ParsedCommand) -> ActionResult:
        # Get required game state
        player_state, current_location, _ = await self.get_required_state()

        # Basic validation (unchanged)
        if not current_location:
            return ActionResult(
                success=False,
                feedback_message="Error: Cannot determine current location.",
            )

        if not command.subject:
            return ActionResult(
                success=False, feedback_message="What do you want to use?"
            )

        # Normalize names (unchanged)
        normalized_item_name = self.normalize_name(command.subject)
        normalized_target_name = self.normalize_name(command.target)

        # Find the item in the player's inventory (unchanged)
        item_to_use = self._find_item_in_inventory(player_state, normalized_item_name)
        if not item_to_use:
            return ActionResult(
                success=False, feedback_message=f"You don't have a {command.subject}."
            )

        # Get the appropriate handler using factory
        usage_factory = UsageHandlerFactory()
        usage_handler = usage_factory.get_handler(command.target)

        # Delegate the handling to the appropriate handler
        return await usage_handler.handle(
            item_to_use, command.target, player_state, current_location
        )

    def _find_item_in_inventory(self, player_state, normalized_item_name):
        """Find an item in the player's inventory by its normalized name."""
        # Unchanged implementation
```

## Benefits

1. **Maintainability**: Each usage scenario is encapsulated in its own class
2. **Extensibility**: New usage types can be added without modifying existing code
3. **Testability**: Each handler can be tested in isolation
4. **Separation of Concerns**: Clear separation between detection and handling logic
5. **Reduced Complexity**: Main handler class becomes simpler and focused

## Implementation Steps

1. Create the directory structure for new files
2. Implement the UsageHandler interface
3. Implement concrete handler classes for each scenario
4. Create the UsageHandlerFactory
5. Refactor the UseHandler class to use the factory
6. Update tests to ensure all scenarios continue to work
7. Add extension points for future handlers

## Future Enhancements

1. **Configuration-based Handler Registration**: Allow registering new handlers through configuration
2. **Dynamic Discovery**: Auto-discover handlers using reflection or module scanning
3. **Runtime Extension**: Allow plugins to register new handlers at runtime
4. **Context-Aware Handler Selection**: Use more sophisticated logic for handler selection based on game context
5. **Composition**: Allow handlers to be composed for complex scenarios
