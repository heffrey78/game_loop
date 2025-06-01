# Commit 14: Entity Embedding Generator

## Overview

This commit implements specialized embedding generation for game entities building upon the core embedding service infrastructure established in commit 13. The entity embedding generator will provide optimized embeddings for different game entity types such as characters, locations, items, and events, enabling more accurate semantic similarity matching and contextually-aware search capabilities.

## Goals

1. Create an EntityEmbeddingGenerator class specialized for game entities
2. Implement entity-specific preprocessing strategies
3. Develop entity context enrichment for better embedding quality
4. Add entity metadata integration with embeddings
5. Implement batch processing optimized for game entities
6. Create utilities for entity embedding visualization and analysis

## Implementation Tasks

### 1. Core EntityEmbeddingGenerator Class (`src/game_loop/embeddings/entity_embeddings.py`)

**Purpose**: Generate optimized embeddings for different game entity types.

**Key Components**:
- `EntityEmbeddingGenerator` class extending the base `EmbeddingService`
- Entity type-specific embedding strategies
- Entity metadata integration
- Customized caching based on entity IDs

**Methods to Implement**:
```python
async def generate_entity_embedding(self, entity: dict, entity_type: str) -> list[float]
async def generate_entity_embeddings_batch(self, entities: list[dict], entity_types: list[str] = None) -> list[list[float]]
def preprocess_entity(self, entity: dict, entity_type: str) -> str
def extract_entity_features(self, entity: dict, entity_type: str) -> dict
async def get_entity_by_embedding_similarity(self, query_embedding: list[float], entity_type: str = None, top_k: int = 5) -> list[tuple[dict, float]]
def get_entity_cache_key(self, entity: dict) -> str
```

### 2. Entity-Specific Preprocessing (`src/game_loop/embeddings/entity_preprocessing.py`)

**Purpose**: Optimize text preprocessing for different entity types.

**Key Components**:
- Entity type registry with specialized preprocessors
- Context enrichment templates for each entity type
- Feature extraction for different entity types
- Standardized entity text representation

**Functions to Implement**:
```python
def preprocess_character(character_entity: dict) -> str
def preprocess_location(location_entity: dict) -> str
def preprocess_item(item_entity: dict) -> str
def preprocess_event(event_entity: dict) -> str
def extract_salient_features(entity: dict, entity_type: str) -> list[str]
def create_entity_context(entity: dict, entity_type: str) -> str
def build_entity_embedding_template(entity_type: str) -> str
```

### 3. Entity Embedding Registry (`src/game_loop/embeddings/entity_registry.py`)

**Purpose**: Track and manage entity embeddings for efficient retrieval.

**Key Components**:
- Registry for mapping entity IDs to embeddings
- Index structures for fast similarity search
- Entity version tracking for embedding updates
- Metadata storage alongside embeddings

**Classes to Implement**:
```python
class EntityEmbeddingRegistry:
    def __init__(self, dimension: int = 384)
    async def register_entity(self, entity_id: str, entity_type: str, embedding: list[float], metadata: dict = None) -> None
    async def get_entity_embedding(self, entity_id: str) -> Optional[list[float]]
    async def find_similar_entities(self, query_embedding: list[float], entity_type: str = None, top_k: int = 5) -> list[tuple[str, float]]
    async def update_entity_embedding(self, entity_id: str, new_embedding: list[float], metadata: dict = None) -> None
    async def remove_entity(self, entity_id: str) -> bool
    def get_all_entity_ids(self, entity_type: str = None) -> list[str]
    async def export_registry(self, file_path: Path) -> None
    async def import_registry(self, file_path: Path) -> None
```

### 4. Entity Similarity Search (`src/game_loop/embeddings/similarity.py`)

**Purpose**: Perform efficient similarity search using entity embeddings.

**Key Components**:
- Multiple similarity metrics (cosine, dot product, euclidean)
- Optimized search algorithms for entity context
- Contextual boosting based on game state
- Hybrid search combining embedding and keyword matching

**Functions to Implement**:
```python
def cosine_similarity(vec1: list[float], vec2: list[float]) -> float
def euclidean_distance(vec1: list[float], vec2: list[float]) -> float
def dot_product(vec1: list[float], vec2: list[float]) -> float
async def search_entities(query_embedding: list[float], entities_embeddings: dict, top_k: int = 5, metric: str = "cosine") -> list[tuple[str, float]]
def create_index_from_embeddings(embeddings: dict, index_type: str = "flat") -> Any
async def query_index(index: Any, query_embedding: list[float], top_k: int = 5) -> list[tuple[str, float]]
def boost_by_context(similarities: list[tuple[str, float]], context_factors: dict) -> list[tuple[str, float]]
```

### 5. Embedding Analytics and Visualization (`src/game_loop/embeddings/analytics.py`)

**Purpose**: Analyze and visualize entity embeddings for debugging and tuning.

**Key Components**:
- Embedding statistics generation
- Clustering and dimensionality reduction
- Distance matrix calculation
- Simple visualization utilities

**Functions to Implement**:
```python
def compute_embedding_stats(embeddings: list[list[float]]) -> dict
def reduce_dimensions(embeddings: list[list[float]], method: str = "pca", dimensions: int = 2) -> list[list[float]]
def cluster_embeddings(embeddings: list[list[float]], method: str = "kmeans", n_clusters: int = 5) -> list[int]
def calculate_distance_matrix(embeddings: list[list[float]], metric: str = "cosine") -> list[list[float]]
async def generate_embedding_report(entity_registry: EntityEmbeddingRegistry, output_path: Path) -> None
def visualize_embeddings(reduced_embeddings: list[list[float]], labels: list[str] = None, output_path: Path = None) -> None
```

### 6. Integration Updates

**GameStateManager Integration**:
- Add EntityEmbeddingGenerator to GameStateManager
- Initialize EntityEmbeddingRegistry during startup
- Add entity registration during entity creation/update

**Database Integration**:
- Create schema for storing entity embeddings
- Add entity embedding sync with database
- Implement batch loading/saving of embeddings

**Entity Search API**:
- Add search endpoints for finding similar entities
- Implement hybrid search combining keyword and semantic search
- Add entity recommendation based on context

## File Structure

```
src/game_loop/embeddings/
├── __init__.py
├── service.py             # Core EmbeddingService class (from commit 13)
├── entity_embeddings.py   # EntityEmbeddingGenerator class
├── entity_preprocessing.py # Entity-specific preprocessing
├── entity_registry.py     # Registry for entity embeddings
├── similarity.py          # Similarity search functions
├── analytics.py           # Embedding analytics and visualization
├── preprocessing.py       # General preprocessing (from commit 13)
├── cache.py               # Caching system (from commit 13)
├── config.py              # Configuration models (enhanced)
└── exceptions.py          # Error handling and exceptions (enhanced)
```

## Testing Strategy

### Unit Tests (`tests/unit/embeddings/`)

1. **test_entity_embeddings.py**:
   - Test entity embedding generation for different entity types
   - Test entity preprocessing functions
   - Test entity metadata extraction
   - Test entity cache key generation

2. **test_entity_registry.py**:
   - Test entity registration and retrieval
   - Test similar entity finding
   - Test entity version tracking
   - Test import/export functionality

3. **test_similarity.py**:
   - Test similarity metrics (cosine, euclidean, dot product)
   - Test search algorithms with various parameters
   - Test contextual boosting functionality
   - Test hybrid search capabilities

4. **test_analytics.py**:
   - Test dimension reduction methods
   - Test clustering algorithms
   - Test statistics generation
   - Test visualization functions

### Integration Tests (`tests/integration/embeddings/`)

1. **test_entity_embedding_integration.py**:
   - Test end-to-end entity embedding generation
   - Test integration with the base embedding service
   - Test with realistic entity data

2. **test_embedding_search_integration.py**:
   - Test search across different entity types
   - Test performance with large entity sets
   - Test accuracy of similarity results

3. **test_game_state_integration.py**:
   - Test integration with game state management
   - Test entity updates reflection in embeddings
   - Test search during game flow

## Verification Criteria

### Functional Verification
- [x] Generate embeddings for various entity types successfully
- [x] Verify entity preprocessing improves embedding quality
- [x] Confirm registry correctly tracks entity embeddings
- [x] Test entity similarity search returns relevant results
- [x] Validate analytics functions provide useful insights

### Performance Verification
- [x] Entity embedding generation < 3s for single entity
- [x] Batch processing shows linear scaling up to 100 entities
- [x] Similarity search completes in < 100ms for 1000 entities
- [x] Registry operations (add, get, find) complete in < 50ms
- [x] Memory usage stays within acceptable limits (< 200MB for registry)

### Accuracy Verification
- [x] Similar entities have high similarity scores (> 0.7)
- [x] Dissimilar entities have low similarity scores (< 0.3)
- [x] Entity type-specific preprocessing improves accuracy by at least 10%
- [x] Contextual boosting improves relevance in game context

## Dependencies

### New Dependencies
- Optional: `scikit-learn` for advanced analytics and dimension reduction
- Optional: `matplotlib` for embedding visualization
- No required new dependencies

### Configuration Updates
- Add entity embedding section to main configuration
- Add entity type registry configuration
- Add similarity metrics configuration

## Integration Points

1. **With EmbeddingService**: Extend the base service for entity-specific functionality
2. **With GameStateManager**: Provide entity embedding and search capabilities
3. **With Database**: Store and retrieve entity embeddings
4. **With Entity System**: Process entity updates to maintain embeddings

## Migration Considerations

- This builds upon commit 13, which must be implemented first
- No migration needed for existing data, embeddings will be generated on-demand
- Optional import/export functionality for pre-computed embeddings

## Code Quality Requirements

- [x] All code passes black, ruff, and mypy linting
- [x] Comprehensive docstrings for all public methods
- [x] Type hints for all function parameters and return values
- [x] Error handling for all entity processing edge cases
- [x] Logging for debugging and monitoring
- [x] Performance annotations for resource-intensive operations

## Documentation Updates

- [x] Update README.md with entity embedding functionality
- [x] Document entity type-specific preprocessing strategies
- [x] Add tutorial for entity similarity search
- [x] Create guide for embedding analytics and visualization
- [x] Document entity embedding performance considerations

## Future Considerations

This entity embedding generator will serve as the foundation for:
- **Commit 15**: Embedding Database Integration (storing embeddings in PostgreSQL)
- **Commit 16**: Semantic Search Implementation (comprehensive search API)
- **Commit 17**: Search Integration with Game Loop (connecting search to game mechanics)
- **Future**: Dynamic entity relationship discovery based on embeddings
- **Future**: Adaptive entity descriptions based on semantic similarity

The design should be flexible enough to support these future enhancements while maintaining backward compatibility.
