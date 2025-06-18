# Commit 30.5 Runtime Fixes

## Summary
This document details the runtime issues encountered after implementing Commit 30.5 (Static Rules Engine Integration) and the fixes applied.

## Issues and Fixes

### 1. Player Exploration Tracking Error ✅
**Error**: `list indices must be integers or slices, not str`

**Root Cause**: The code attempted to use `player_state.inventory` as a dictionary when it's actually a list of InventoryItem objects.

**Fix**: 
- Added `behavior_stats: dict[str, Any] | None` field to PlayerState model
- Updated `_track_player_exploration` and `_get_player_preferences` methods to use the new field

**Files Modified**:
- `src/game_loop/state/models.py` - Added behavior_stats field
- `src/game_loop/core/game_loop.py` - Fixed inventory access patterns

### 2. Database Foreign Key Constraint Errors ✅
**Error**: `insert or update on table "location_connections" violates foreign key constraint`

**Root Cause**: Code was attempting to insert placeholder connections with UUID `00000000-0000-0000-0000-000000000000` which doesn't exist in the locations table.

**Fix**: Modified `_create_placeholder_connection` to only store placeholders in memory, not in the database.

**Files Modified**:
- `src/game_loop/core/game_loop.py` - Removed database insertion for placeholder connections

### 3. Missing connection_type Column ✅
**Error**: `column "connection_type" of relation "location_connections" does not exist`

**Root Cause**: Database migrations were not applied completely due to partial failures.

**Fix**: Reset database with `make db-reset` to apply all migrations cleanly.

### 4. ParsedCommand Attribute Error ✅
**Error**: `'ParsedCommand' object has no attribute 'direct_object'`

**Root Cause**: Rule evaluation code was trying to access `direct_object` and `indirect_object` attributes that don't exist on ParsedCommand.

**Fix**: Updated `_convert_to_rule_context` to use the correct attributes: `subject`, `target`, and `parameters`.

**Files Modified**:
- `src/game_loop/core/game_loop.py` - Fixed attribute references in rule context conversion

### 5. LLM Timeout Issues ✅
**Error**: `httpx.ReadTimeout` during dialogue generation

**Root Cause**: 
1. JSON format requirement was making the model struggle
2. Default timeout of 60 seconds was insufficient for complex dialogue generation

**Fix**:
1. Modified dialogue generation to accept plain text responses (not just JSON)
2. Increased Ollama client timeout to 120 seconds
3. Added fallback handling for non-JSON responses

**Files Modified**:
- `src/game_loop/core/game_loop.py` - Updated `_generate_npc_dialogue_response` and OllamaClient initialization

## Testing
After applying all fixes:
- ✅ All 1,278 tests passing
- ✅ Game runs without runtime errors
- ✅ NPC dialogue works (both JSON and plain text responses)
- ✅ Rule evaluation works without attribute errors
- ✅ Dynamic location expansion works without database constraints

### 6. Rule Evaluation Type Errors ✅
**Error**: `Input should be a valid string [type=string_type, input_value=8, input_type=int]`

**Root Cause**: 
1. `command.command_type.value` returns integer but RuleEvaluationContext expects string
2. `current_location.location_id` returns UUID object but expects string

**Fix**: 
1. Changed `command.command_type.value` to `command.command_type.name` 
2. Changed `current_location.location_id` to `str(current_location.location_id)`

**Files Modified**:
- `src/game_loop/core/game_loop.py` - Fixed type conversions in both pre and post command rule evaluation

## Key Learnings
1. Always verify data structure types before accessing (list vs dict)
2. Don't insert placeholder/temporary data into database with foreign key constraints
3. Allow flexible response formats from LLMs to handle timeout/parsing issues
4. Ensure dataclass attributes match actual implementation when refactoring
5. Increase timeouts for LLM operations that require complex generation
6. Always convert enum values and UUID objects to strings when passing to Pydantic models that expect string types