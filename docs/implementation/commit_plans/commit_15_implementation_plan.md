# Commit 15: Embedding Database Integration

## Overview

This commit implements the integration of entity embeddings with the database system, building upon the entity embedding generator established in commit 14. The database integration will provide persistent storage for embeddings, efficient querying capabilities, versioning, and synchronization between in-memory and database representations, enabling scalable semantic search across large numbers of game entities.

## Goals

1. Create database schemas for storing entity embeddings
2. Implement database storage and retrieval for embeddings
3. Develop efficient batch operations for embedding synchronization
4. Add versioning and change tracking for embeddings
5. Implement embedding persistence strategies
6. Create database initialization and setup utilities

## Implementation Tasks

### 1. Database Schema Design (`src/game_loop/database/schemas/embedding.py`)

**Purpose**: Define the database schema for storing entity embeddings.

**Key Components**:
- Table definitions for entity embeddings
- Indexes for efficient similarity search
- Metadata storage for embedding context
- Version tracking fields

**Schema to Implement**:
```python
from sqlalchemy import Column, String, Float, Integer, ForeignKey, JSON, DateTime, func, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import relationship
from typing import List, Dict, Any

Base = declarative_base()

class EntityEmbedding(Base):
    __tablename__ = "entity_embeddings"

    id = Column(String, primary_key=True)  # entity_id
    entity_type = Column(String, nullable=False)
    embedding = Column(ARRAY(Float), nullable=False)
    dimension = Column(Integer, nullable=False)
    model_version = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    metadata = Column(JSON, nullable=True)

    # Create indexes for faster lookup
    __table_args__ = (
        Index('ix_entity_type', 'entity_type'),
        Index('ix_model_version', 'model_version'),
    )
```

### 2. Embedding Database Manager (`src/game_loop/database/managers/embedding_manager.py`)

**Purpose**: Manage database operations for entity embeddings.

**Key Components**:
- CRUD operations for entity embeddings
- Batch operations for efficient processing
- Versioning and change tracking
- Integration with EntityEmbeddingRegistry

**Methods to Implement**:
```python
class EmbeddingDatabaseManager:
    def __init__(self, session_factory, registry=None):
        self.session_factory = session_factory
        self.registry = registry

    async def store_embedding(self, entity_id: str, entity_type: str, embedding: List[float], metadata: Dict[str, Any] = None) -> None:
        """Store an entity embedding in the database"""

    async def store_embeddings_batch(self, entities: List[Dict[str, Any]]) -> None:
        """Store multiple entity embeddings in a single transaction"""

    async def get_embedding(self, entity_id: str) -> List[float]:
        """Retrieve an entity embedding from the database"""

    async def get_embeddings_batch(self, entity_ids: List[str]) -> Dict[str, List[float]]:
        """Retrieve multiple entity embeddings in a single query"""

    async def update_embedding(self, entity_id: str, embedding: List[float], metadata: Dict[str, Any] = None) -> None:
        """Update an existing entity embedding"""

    async def delete_embedding(self, entity_id: str) -> None:
        """Delete an entity embedding from the database"""

    async def get_embeddings_by_entity_type(self, entity_type: str) -> Dict[str, List[float]]:
        """Get all embeddings for a specific entity type"""

    async def get_embedding_metadata(self, entity_id: str) -> Dict[str, Any]:
        """Get metadata for an entity embedding"""

    async def sync_with_registry(self) -> None:
        """Synchronize database embeddings with the in-memory registry"""
```

### 3. Embedding Persistence Manager (`src/game_loop/embeddings/persistence.py`)

**Purpose**: Handle synchronization between in-memory and database embeddings.

**Key Components**:
- Synchronization strategies (eager, lazy, periodic)
- Change detection and version reconciliation
- Batch processing for efficient database operations
- Error handling and retry logic

**Methods to Implement**:
```python
class EmbeddingPersistenceManager:
    def __init__(self, registry, db_manager, sync_strategy="eager"):
        self.registry = registry
        self.db_manager = db_manager
        self.sync_strategy = sync_strategy
        self._pending_changes = {}
        self._last_sync = None

    async def register_entity_embedding(self, entity_id: str, entity_type: str, embedding: List[float], metadata: Dict[str, Any] = None) -> None:
        """Register an entity embedding and persist according to strategy"""

    async def get_embedding(self, entity_id: str) -> List[float]:
        """Get embedding with fallback between registry and database"""

    async def sync_now(self, force: bool = False) -> None:
        """Force synchronization between registry and database"""

    async def schedule_periodic_sync(self, interval_seconds: int = 300) -> None:
        """Schedule periodic synchronization"""

    def _detect_changes(self) -> Dict[str, Any]:
        """Detect changes between registry and last known database state"""

    async def _process_pending_changes(self) -> None:
        """Process pending changes to database"""

    async def load_from_database(self, entity_type: str = None) -> None:
        """Load embeddings from database into registry"""

    async def flush_to_database(self) -> None:
        """Flush all registry embeddings to database"""
```

### 4. Database Query Extensions for Similarity Search (`src/game_loop/database/queries/similarity.py`)

**Purpose**: Provide PostgreSQL-specific extensions for vector similarity search.

**Key Components**:
- PostgreSQL vector extensions integration
- Distance functions for different metrics
- Optimized query builders for similarity search
- Index utilization for efficient search

**Functions to Implement**:
```python
from sqlalchemy import text
from sqlalchemy.sql import select
from typing import List, Tuple

def create_vector_extension(connection) -> None:
    """Create PostgreSQL vector extension if not exists"""
    connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

def create_embedding_index(connection, table_name: str, column_name: str) -> None:
    """Create vector index for faster similarity search"""
    connection.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_{column_name}_vector ON {table_name} USING vector({column_name})"))

def cosine_similarity_sql(embedding_column: str) -> str:
    """Generate SQL for cosine similarity calculation"""
    return f"1 - ({embedding_column} <=> :query_embedding)"

def euclidean_distance_sql(embedding_column: str) -> str:
    """Generate SQL for euclidean distance calculation"""
    return f"({embedding_column} <-> :query_embedding)"

def build_similarity_query(entity_table, query_embedding: List[float], entity_type: str = None, top_k: int = 10, metric: str = "cosine") -> Tuple:
    """Build a query for finding similar entities in the database"""
    # Implementation here

async def search_similar_entities(session, query_embedding: List[float], entity_type: str = None, top_k: int = 10, metric: str = "cosine") -> List[Tuple[str, float]]:
    """Search for similar entities in the database"""
    # Implementation here
```

### 5. Database Initialization Utilities (`src/game_loop/database/initialization/embedding_setup.py`)

**Purpose**: Provide utilities for initial database setup and embedding generation.

**Key Components**:
- Database table creation and schema initialization
- Batch embedding generation for test data
- Verification utilities for database setup
- Simple admin commands for database reset

**Functions to Implement**:
```python
async def initialize_embedding_database(connection) -> None:
    """Set up initial database tables and extensions for embeddings"""

async def generate_embeddings_for_test_entities(db_session, embedding_generator, batch_size: int = 100) -> dict:
    """Generate embeddings for test entities to populate the database"""

async def verify_database_setup(db_session) -> bool:
    """Verify that embedding tables and extensions are properly set up"""

async def reset_embedding_database(db_session) -> None:
    """Clear all embedding data for development purposes"""

async def calculate_embedding_coverage(db_session) -> dict:
    """Calculate percentage of entities that have embeddings"""
```

### 6. Integration with Game Systems

**EntitySystem Integration**:
- Add embedding generation on entity creation/update
- Ensure embedding consistency with entity changes
- Add embedding lookup methods to entity APIs

**GameStateManager Integration**:
- Initialize embedding database manager during startup
- Configure persistence strategy based on game settings
- Add entity embedding context to game state

**API Integration**:
- Add embedding-based search endpoints
- Enable filtering and ranking by embedding similarity
- Add metadata endpoints for embedding analytics

## File Structure

```
src/game_loop/
├── database/
│   ├── schemas/
│   │   ├── embedding.py          # Database schema for embeddings
│   ├── managers/
│   │   ├── embedding_manager.py  # Database operations for embeddings
│   ├── queries/
│   │   ├── similarity.py         # Specialized queries for vector search
│   ├── migrations/
│   │   ├── embedding_migration.py # Migration utilities
├── embeddings/
│   ├── persistence.py            # Embedding persistence strategies
│   ├── entity_embeddings.py      # (from commit 14)
│   ├── entity_registry.py        # (enhanced with persistence)
│   ├── similarity.py             # (enhanced with database integration)
```

## Testing Strategy

### Unit Tests

1. **Database Schema Tests** (`tests/unit/database/test_embedding_schema.py`):
   - Test schema validation
   - Test relationship integrity
   - Test index creation
   - Test data type constraints

2. **EmbeddingDatabaseManager Tests** (`tests/unit/database/test_embedding_manager.py`):
   - Test CRUD operations
   - Test batch operations efficiency
   - Test version tracking
   - Test error handling

3. **Persistence Manager Tests** (`tests/unit/embeddings/test_persistence.py`):
   - Test synchronization strategies
   - Test change detection
   - Test batch processing
   - Test error recovery

4. **Database Query Tests** (`tests/unit/database/test_similarity_queries.py`):
   - Test similarity search queries
   - Test query performance
   - Test different metrics
   - Test filtering options

### Integration Tests

1. **Database Integration Tests** (`tests/integration/database/test_embedding_db_integration.py`):
   - Test end-to-end database operations
   - Test transaction handling
   - Test concurrent access
   - Test with realistic data volumes

2. **Registry-Database Sync Tests** (`tests/integration/embeddings/test_registry_db_sync.py`):
   - Test bidirectional synchronization
   - Test conflict resolution
   - Test performance at scale
   - Test with network delays

3. **Migration Tests** (`tests/integration/database/test_embedding_migration.py`):
   - Test migration from different data formats
   - Test batch migration performance
   - Test validation utilities
   - Test error handling during migration

### Performance Tests

1. **Benchmark Tests** (`tests/performance/test_embedding_db_performance.py`):
   - Test read/write operations per second
   - Test similarity search latency
   - Test memory usage during synchronization
   - Test scaling with increasing entity counts

## Verification Criteria

### Functional Verification
- [ ] Entity embeddings store and retrieve correctly from database
- [ ] Batch operations perform correctly and efficiently
- [ ] Entity metadata is preserved with embeddings
- [ ] Version tracking correctly identifies changes
- [ ] Synchronization works bidirectionally between registry and database

### Performance Verification
- [ ] Batch storage of 100 embeddings completes in < 5s
- [ ] Similarity search on 10,000 embeddings completes in < 500ms
- [ ] Registry-database synchronization of 1,000 embeddings completes in < 10s
- [ ] Memory usage remains stable during large synchronization operations
- [ ] Database indexes improve search performance by at least 5x

### Integration Verification
- [ ] Works with existing EntityEmbeddingRegistry
- [ ] Integrates with EntityEmbeddingGenerator
- [ ] Compatible with game entity lifecycle events
- [ ] Maintains integrity with database migrations
- [ ] Provides consistent API for embedding access

## Dependencies

### New Dependencies
- Required: `psycopg2-binary` for PostgreSQL vector extensions
- Optional: `pgvector` PostgreSQL extension for efficient vector operations
- Optional: `alembic` for database migrations

### Configuration Updates
- Add database connection settings for embeddings
- Add synchronization strategy configuration
- Add batch size parameters
- Add vector index configuration

## Integration Points

1. **With Database System**: Store and query embeddings in PostgreSQL
2. **With EntityEmbeddingRegistry**: Synchronize in-memory and database embeddings
3. **With EntityEmbeddingGenerator**: Generate embeddings for database storage
4. **With GameStateManager**: Integrate embedding lookup during game operations

## Migration Considerations

- PostgreSQL instance must support vector extensions
- Initial migration may require significant processing time for large entity sets
- Consider implementing progressive migration for production environments
- Create backup of entity data before migration

## Code Quality Requirements

- [ ] All code passes black, ruff, and mypy linting
- [ ] Comprehensive docstrings for all public methods
- [ ] Type hints for all function parameters and return values
- [ ] Error handling for all database operations
- [ ] Comprehensive logging for tracking synchronization issues
- [ ] Performance annotations for resource-intensive operations

## Documentation Updates

- [ ] Update README.md with embedding database integration information
- [ ] Create guide for embedding persistence strategies
- [ ] Document PostgreSQL setup for vector operations
- [ ] Add migration guide for existing installations
- [ ] Update entity embedding system documentation with database integration details

## Future Considerations

This embedding database integration will serve as the foundation for:
- **Commit 16**: Semantic Search Implementation (comprehensive search API)
- **Commit 17**: Search Integration with Game Loop (connecting search to game mechanics)
- **Future**: Advanced entity relationship database structure
- **Future**: Real-time embedding updates during gameplay
- **Future**: Distributed database sharding for very large entity collections

The design should be flexible enough to support these future enhancements while maintaining backward compatibility.
