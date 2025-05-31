# Commit 13: Embedding Service Implementation

## Overview

This commit implements the core embedding service infrastructure for the game loop system. The embedding service will handle text preprocessing, embedding generation using Ollama, caching mechanisms, and error handling with retry logic. This foundation will enable semantic search capabilities that will be integrated in subsequent commits.

## Goals

1. Create a robust EmbeddingService class for text-to-vector conversion
2. Implement text preprocessing functions for optimal embedding quality
3. Add embedding generation using Ollama integration
4. Create caching system for frequently used embeddings
5. Implement retry logic and comprehensive error handling
6. Establish the foundation for semantic search functionality

## Implementation Tasks

### 1. Core EmbeddingService Class (`src/game_loop/embeddings/service.py`)

**Purpose**: Central service for generating and managing text embeddings.

**Key Components**:
- `EmbeddingService` class with async methods
- Integration with existing `OllamaClient`
- Configuration management for embedding parameters
- Batch processing capabilities
- Memory and disk caching systems

**Methods to Implement**:
```python
async def generate_embedding(self, text: str, entity_type: str = "general") -> list[float]
async def generate_embeddings_batch(self, texts: list[str], entity_types: list[str] = None) -> list[list[float]]
async def get_embedding_dimension(self) -> int
def preprocess_text(self, text: str, entity_type: str = "general") -> str
async def _generate_with_retry(self, text: str, max_retries: int = 3) -> list[float]
```

### 2. Text Preprocessing Functions (`src/game_loop/embeddings/preprocessing.py`)

**Purpose**: Standardize and optimize text for embedding generation.

**Key Components**:
- Entity-type specific preprocessing strategies
- Text normalization and cleaning
- Context enrichment for better embeddings
- Chunking for long texts

**Functions to Implement**:
```python
def preprocess_for_embedding(text: str, entity_type: str = "general", max_length: int = 512) -> str
def normalize_text(text: str) -> str
def enrich_context(text: str, entity_type: str, additional_context: dict = None) -> str
def chunk_text(text: str, max_length: int = 512, overlap: int = 50) -> list[str]
def clean_text(text: str) -> str
```

### 3. Caching System (`src/game_loop/embeddings/cache.py`)

**Purpose**: Improve performance by caching embeddings and reducing API calls.

**Key Components**:
- In-memory LRU cache for recent embeddings
- Persistent disk cache for long-term storage
- Cache invalidation strategies
- Cache statistics and monitoring

**Classes to Implement**:
```python
class EmbeddingCache:
    def __init__(self, memory_size: int = 1000, disk_cache_dir: Path = None)
    async def get(self, text_hash: str) -> Optional[list[float]]
    async def set(self, text_hash: str, embedding: list[float]) -> None
    def get_cache_stats(self) -> dict
    async def clear_cache(self, cache_type: str = "all") -> None
```

### 4. Configuration Models (`src/game_loop/embeddings/config.py`)

**Purpose**: Configuration for embedding service parameters and model settings.

**Key Components**:
```python
class EmbeddingConfig(BaseModel):
    model_name: str = "nomic-embed-text"
    max_text_length: int = 512
    batch_size: int = 10
    cache_enabled: bool = True
    cache_size: int = 1000
    retry_attempts: int = 3
    retry_delay: float = 1.0
    preprocessing_enabled: bool = True
```

### 5. Error Handling and Retry Logic (`src/game_loop/embeddings/exceptions.py`)

**Purpose**: Robust error handling for embedding generation failures.

**Key Components**:
```python
class EmbeddingError(Exception): pass
class EmbeddingGenerationError(EmbeddingError): pass
class EmbeddingCacheError(EmbeddingError): pass
class EmbeddingConfigError(EmbeddingError): pass

# Retry decorators and utilities
async def with_retry(func, max_retries: int = 3, delay: float = 1.0)
```

### 6. Integration Updates

**GameStateManager Integration**:
- Add optional EmbeddingService to GameStateManager constructor
- Update initialization to create EmbeddingService instance

**Configuration Integration**:
- Add embedding configuration to main config system
- Update CLI and YAML config support for embedding parameters

**OllamaClient Enhancement**:
- Ensure embedding endpoint support in OllamaClient
- Add embedding-specific error handling

## File Structure

```
src/game_loop/embeddings/
├── __init__.py
├── service.py          # Core EmbeddingService class
├── preprocessing.py    # Text preprocessing functions
├── cache.py           # Caching system
├── config.py          # Configuration models
└── exceptions.py      # Error handling and exceptions
```

## Testing Strategy

### Unit Tests (`tests/unit/embeddings/`)

1. **test_service.py**:
   - Test embedding generation with various inputs
   - Test batch processing functionality
   - Test retry logic with simulated failures
   - Test error handling for invalid inputs

2. **test_preprocessing.py**:
   - Test text normalization and cleaning
   - Test entity-type specific preprocessing
   - Test context enrichment functionality
   - Test text chunking for long inputs

3. **test_cache.py**:
   - Test cache hit/miss scenarios
   - Test cache persistence across sessions
   - Test cache size limits and eviction
   - Test cache statistics and monitoring

### Integration Tests (`tests/integration/embeddings/`)

1. **test_ollama_integration.py**:
   - Test actual embedding generation with Ollama
   - Test different model configurations
   - Test performance with various text sizes

2. **test_cache_integration.py**:
   - Test cache performance improvements
   - Test cache persistence with real embeddings
   - Test cache invalidation scenarios

## Verification Criteria

### Functional Verification
- [ ] Generate embeddings for test texts successfully
- [ ] Verify caching improves performance on repeated requests
- [ ] Test retry logic with simulated Ollama failures
- [ ] Confirm text preprocessing improves embedding quality
- [ ] Verify batch processing works efficiently

### Performance Verification
- [ ] Embedding generation completes within reasonable time (< 2s for single embedding)
- [ ] Cache hit rate > 80% for repeated text patterns
- [ ] Batch processing shows performance improvement over individual calls
- [ ] Memory usage stays within acceptable limits (< 100MB for cache)

### Error Handling Verification
- [ ] Graceful handling of Ollama service unavailability
- [ ] Proper error messages for invalid input text
- [ ] Retry logic activates and succeeds after transient failures
- [ ] Cache errors don't prevent embedding generation

## Dependencies

### New Dependencies
- No new external dependencies required (using existing Ollama client)
- May add `diskcache` for persistent caching if needed

### Configuration Updates
- Add embedding section to main configuration
- Update YAML configuration templates
- Add CLI parameters for embedding configuration

## Integration Points

1. **With OllamaClient**: Use existing client for embedding generation
2. **With ConfigManager**: Integrate embedding configuration
3. **With GameStateManager**: Optional embedding service integration
4. **With Future Components**: Foundation for semantic search and entity embedding

## Migration Considerations

- This is a new feature, no migration required
- Configuration is optional with sensible defaults
- Service can be disabled if Ollama embedding model unavailable

## Code Quality Requirements

- [ ] All code passes black, ruff, and mypy linting
- [ ] Comprehensive docstrings for all public methods
- [ ] Type hints for all function parameters and return values
- [ ] Error handling for all external service calls
- [ ] Logging for debugging and monitoring

## Documentation Updates

- [ ] Update README.md with embedding service overview
- [ ] Add embedding configuration documentation
- [ ] Document preprocessing strategies for different entity types
- [ ] Add troubleshooting guide for embedding issues

## Future Considerations

This embedding service will serve as the foundation for:
- **Commit 14**: Entity Embedding Generator (specialized embedding for game entities)
- **Commit 15**: Embedding Database Integration (storing embeddings in PostgreSQL)
- **Commit 16**: Semantic Search Implementation (using embeddings for similarity search)
- **Commit 17**: Search Integration with Game Loop (connecting search to game mechanics)

The design should be flexible enough to support these future enhancements while maintaining backward compatibility.
