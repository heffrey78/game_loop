# Database Schema Fix Plan

## Overview

This document outlines the necessary fixes to address critical database schema inconsistencies identified during the commit 22 code review. These issues must be resolved before proceeding with commit 23 to ensure system stability and prevent future migration complications.

## Critical Issues Identified

### 1. ID Type Inconsistencies
**Problem**: Mixed use of VARCHAR and UUID types for primary keys
- Conversation system uses VARCHAR(255) for IDs that should be UUIDs
- Existing game state tables use proper UUID types
- This creates foreign key relationship problems and type validation issues

**Files Affected**:
- `src/game_loop/database/migrations/023_conversation_system.sql`
- Related models and repositories

### 2. Foreign Key Relationship Problems
**Problem**: Inconsistent ID types prevent proper foreign key constraints
- Player IDs should reference the players table (UUID)
- NPC IDs should reference NPCs table (UUID)
- Current VARCHAR implementation breaks referential integrity

### 3. Index and Performance Issues
**Problem**: Missing indexes on frequently queried columns
- No indexes on conversation lookup columns
- Missing composite indexes for player-NPC conversations
- No indexes on timestamp columns for temporal queries

## Proposed Solutions

### Phase 1: Schema Correction
1. **Update Migration 023**
   - Change all ID columns from VARCHAR(255) to UUID
   - Add proper foreign key constraints
   - Add appropriate indexes

2. **Update Conversation Models**
   - Ensure all model classes use proper UUID types
   - Update validation to enforce UUID format
   - Fix any hardcoded string ID generation

3. **Update Repository Layer**
   - Modify queries to work with UUID types
   - Add proper type conversion where needed
   - Update any string-based ID handling

### Phase 2: Data Migration Strategy
1. **Create Migration 024**
   - Drop and recreate conversation tables with correct schema
   - Since conversation system is new, no data migration needed
   - Add migration rollback capabilities

2. **Update Related Systems**
   - Ensure query processor uses correct ID types
   - Update conversation manager to generate proper UUIDs
   - Verify all cross-system ID references

### Phase 3: Testing and Validation
1. **Database Integration Tests**
   - Test conversation creation with proper UUIDs
   - Verify foreign key constraints work correctly
   - Test conversation queries and performance

2. **System Integration Tests**
   - End-to-end conversation flow testing
   - Cross-system ID validation
   - Performance benchmarking

## Implementation Plan

### Step 1: Create New Migration (024_fix_conversation_schema.sql)
```sql
-- Drop existing conversation tables
DROP TABLE IF EXISTS conversation_exchanges;
DROP TABLE IF EXISTS conversation_contexts;
DROP TABLE IF EXISTS npc_personalities;
DROP TABLE IF EXISTS conversation_knowledge;

-- Recreate with proper UUID types and constraints
CREATE TABLE npc_personalities (
    npc_id UUID PRIMARY KEY,
    traits JSONB NOT NULL DEFAULT '{}',
    knowledge_areas TEXT[] NOT NULL DEFAULT '{}',
    speech_patterns JSONB NOT NULL DEFAULT '{}',
    relationships JSONB NOT NULL DEFAULT '{}',
    background_story TEXT NOT NULL DEFAULT '',
    default_mood VARCHAR(50) NOT NULL DEFAULT 'neutral',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE conversation_contexts (
    conversation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    player_id UUID NOT NULL,
    npc_id UUID NOT NULL REFERENCES npc_personalities(npc_id),
    topic VARCHAR(255),
    mood VARCHAR(50) NOT NULL DEFAULT 'neutral',
    relationship_level DECIMAL(3,2) NOT NULL DEFAULT 0.0,
    context_data JSONB NOT NULL DEFAULT '{}',
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    started_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT chk_relationship_level CHECK (relationship_level >= -1.0 AND relationship_level <= 1.0)
);

CREATE TABLE conversation_exchanges (
    exchange_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversation_contexts(conversation_id) ON DELETE CASCADE,
    speaker_id UUID NOT NULL,
    message_text TEXT NOT NULL,
    message_type VARCHAR(20) NOT NULL,
    emotion VARCHAR(50),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB NOT NULL DEFAULT '{}'
);

CREATE TABLE conversation_knowledge (
    knowledge_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversation_contexts(conversation_id) ON DELETE CASCADE,
    information_type VARCHAR(50) NOT NULL,
    extracted_info JSONB NOT NULL,
    confidence_score DECIMAL(3,2),
    source_exchange_id UUID REFERENCES conversation_exchanges(exchange_id),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT chk_confidence_score CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0)
);

-- Create indexes for performance
CREATE INDEX idx_conversation_contexts_player_npc ON conversation_contexts(player_id, npc_id);
CREATE INDEX idx_conversation_contexts_status ON conversation_contexts(status);
CREATE INDEX idx_conversation_contexts_last_updated ON conversation_contexts(last_updated);
CREATE INDEX idx_conversation_exchanges_conversation ON conversation_exchanges(conversation_id);
CREATE INDEX idx_conversation_exchanges_timestamp ON conversation_exchanges(timestamp);
CREATE INDEX idx_conversation_knowledge_conversation ON conversation_knowledge(conversation_id);
CREATE INDEX idx_conversation_knowledge_type ON conversation_knowledge(information_type);
```

### Step 2: Update Model Classes
**Files to Update**:
- `src/game_loop/database/models/conversation.py` (create if needed)
- `src/game_loop/core/conversation/conversation_models.py`

**Key Changes**:
- Use `UUID` type for all ID fields
- Add proper SQLAlchemy relationships
- Update validation to ensure UUID format
- Add proper foreign key definitions

### Step 3: Update Conversation Manager
**File**: `src/game_loop/core/conversation/conversation_manager.py`

**Changes**:
- Replace in-memory storage with database persistence
- Use proper UUID generation for all IDs
- Add database session management
- Implement proper error handling for DB operations

### Step 4: Update Repository Layer
**Files to Create/Update**:
- `src/game_loop/database/repositories/conversation.py`
- Update related repositories for cross-references

**Implementation**:
- Add conversation CRUD operations
- Implement efficient querying methods
- Add proper transaction handling
- Include caching for frequently accessed data

### Step 5: Integration Testing
**Test Files to Update**:
- `tests/integration/core/test_conversation_integration.py`
- `tests/integration/database/test_conversation_models.py`

**Test Coverage**:
- UUID type validation
- Foreign key constraint testing
- Conversation workflow end-to-end
- Performance benchmarking

## Risk Assessment

### Low Risk
- Schema recreation (no existing data to migrate)
- UUID type conversion (standard PostgreSQL feature)
- Index addition (improves performance)

### Medium Risk
- Model class updates (requires careful testing)
- Repository layer changes (affects data access patterns)
- Cross-system integration (conversation <-> query systems)

### High Risk
- None identified (conversation system is newly implemented)

## Timeline Estimate

- **Day 1**: Create migration and update models (2-3 hours)
- **Day 2**: Update conversation manager and repositories (3-4 hours)
- **Day 3**: Integration testing and validation (2-3 hours)

**Total Effort**: 7-10 hours

## Success Criteria

1. ✅ All conversation tables use proper UUID types
2. ✅ Foreign key constraints are properly enforced
3. ✅ Conversation system persists to database instead of memory
4. ✅ All existing tests pass with updated schema
5. ✅ Performance benchmarks meet acceptable thresholds
6. ✅ Cross-system integration (query + conversation) works correctly

## Follow-up Actions

After completing this schema fix:
1. Proceed with Commit 23: System Command Processing
2. Ensure all future migrations follow UUID consistency
3. Document database schema standards for the project
4. Add schema validation to CI/CD pipeline

## Notes

- This fix addresses technical debt before it compounds
- Proper schema foundation enables reliable system expansion
- UUID consistency improves system scalability and maintainability
- Database-backed conversation persistence enables session continuity