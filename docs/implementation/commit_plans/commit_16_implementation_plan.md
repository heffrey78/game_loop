# Commit 16: Semantic Search Implementation

## Overview

This commit builds upon the embedding database integration established in Commit 15 to implement a comprehensive semantic search system for game entities. The semantic search capability will enable sophisticated natural language queries against game entities, intelligent context retrieval, and similarity-based entity relationships. This system will form a core component of the game's AI-powered interaction mechanisms, allowing for more natural and contextually relevant gameplay experiences.

## Goals

1. Create a unified semantic search API
2. Implement various search strategies (exact match, similarity, hybrid)
3. Add relevance scoring and ranking mechanisms
4. Develop query preprocessing and optimization
5. Create search result post-processing and formatting
6. Implement search caching and performance optimizations
7. Integrate search capabilities with game systems

## Implementation Tasks

### 1. Semantic Search Service (`src/game_loop/search/semantic_search.py`)

**Purpose**: Provide a centralized service for all semantic search operations.

**Key Components**:
- Search API facade for different search strategies
- Query preprocessing and normalization
- Search strategy selection logic
- Result aggregation and ranking
- Search context management

**Methods to Implement**:
```python
class SemanticSearchService:
    def __init__(self, embedding_db_manager, embedding_registry, embedding_generator, cache_size=1000):
        self.db_manager = embedding_db_manager
        self.registry = embedding_registry
        self.generator = embedding_generator
        self._results_cache = {}  # LRU cache for search results
        self._cache_size = cache_size
        self._search_metrics = {}  # Track performance metrics

    async def search(self, query: str, entity_types: List[str] = None, strategy: str = "hybrid",
                    top_k: int = 10, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Primary search method supporting multiple strategies

        Strategies:
        - "semantic": Pure vector similarity search
        - "keyword": Traditional keyword matching
        - "hybrid": Combined semantic and keyword search
        - "exact": Exact match on entity attributes
        """

    async def semantic_search(self, query: str, entity_types: List[str] = None,
                              top_k: int = 10, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Perform semantic search using vector embeddings"""

    async def keyword_search(self, query: str, entity_types: List[str] = None,
                            top_k: int = 10) -> List[Dict[str, Any]]:
        """Perform keyword-based search"""

    async def hybrid_search(self, query: str, entity_types: List[str] = None,
                           top_k: int = 10, semantic_weight: float = 0.7) -> List[Dict[str, Any]]:
        """Combine results from semantic and keyword search with weighting"""

    async def exact_match_search(self, query: str, field: str, entity_types: List[str] = None) -> List[Dict[str, Any]]:
        """Find exact matches for a specific field"""

    def _preprocess_query(self, query: str) -> str:
        """Clean and normalize the search query"""

    def _select_search_strategy(self, query: str, strategy: str) -> str:
        """Determine the best search strategy based on the query and specified strategy"""

    async def _generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for the search query"""

    def _rank_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Rank and sort search results by relevance"""

    def _filter_by_threshold(self, results: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
        """Filter results that don't meet the minimum similarity threshold"""

    def _cache_results(self, query: str, results: List[Dict[str, Any]], strategy: str) -> None:
        """Cache search results for future queries"""

    def get_search_metrics(self) -> Dict[str, Any]:
        """Return performance metrics for recent searches"""
```

### 2. Query Processing Module (`src/game_loop/search/query_processor.py`)

**Purpose**: Handle query preprocessing, analysis, and transformation for optimal search.

**Key Components**:
- Query normalization and cleaning
- Entity type inference
- Query expansion techniques
- Query classification
- Intent detection

**Methods to Implement**:
```python
class QueryProcessor:
    def __init__(self, entity_registry, nlp_processor=None):
        self.entity_registry = entity_registry
        self.nlp_processor = nlp_processor
        self._entity_type_patterns = {}
        self._initialize_patterns()

    def normalize_query(self, query: str) -> str:
        """Clean and normalize the query text"""

    def extract_entity_types(self, query: str) -> List[str]:
        """Identify potential entity types mentioned in the query"""

    def expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms"""

    def classify_query_intent(self, query: str) -> str:
        """Classify the query as lookup, exploration, comparison, etc."""

    def extract_constraints(self, query: str) -> Dict[str, Any]:
        """Extract filtering constraints from the query"""

    def generate_query_variants(self, query: str) -> List[str]:
        """Generate variations of the query for better matching"""

    def estimate_query_complexity(self, query: str) -> float:
        """Estimate the complexity of a query to optimize search strategy"""

    def _initialize_patterns(self) -> None:
        """Initialize regex patterns for entity type detection"""

    def _detect_special_tokens(self, query: str) -> Dict[str, Any]:
        """Detect special tokens or commands in the query"""
```

### 3. Search Results Processor (`src/game_loop/search/results_processor.py`)

**Purpose**: Process, format, and enhance search results.

**Key Components**:
- Result deduplication and merging
- Relevance scoring and sorting
- Entity data enrichment
- Snippet generation
- Result grouping and categorization

**Methods to Implement**:
```python
class SearchResultsProcessor:
    def __init__(self, entity_registry):
        self.entity_registry = entity_registry

    def deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate entities from results"""

    def enrich_results(self, results: List[Dict[str, Any]], include_fields: List[str] = None) -> List[Dict[str, Any]]:
        """Add additional entity information to search results"""

    def calculate_relevance_scores(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Calculate and add relevance scores to results"""

    def generate_result_snippets(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Generate contextual snippets highlighting match relevance"""

    def group_results_by_type(self, results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group results by entity type"""

    def sort_results(self, results: List[Dict[str, Any]], sort_by: str = "relevance") -> List[Dict[str, Any]]:
        """Sort results by specified criterion"""

    def format_results(self, results: List[Dict[str, Any]], format_type: str = "detailed") -> Any:
        """Format results according to specified format type"""

    def paginate_results(self, results: List[Dict[str, Any]], page: int = 1, page_size: int = 10) -> Dict[str, Any]:
        """Paginate results for display"""

    def highlight_matching_terms(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Highlight terms in results that match the query"""
```

### 4. Search Cache Manager (`src/game_loop/search/cache_manager.py`)

**Purpose**: Manage caching of search results for performance optimization.

**Key Components**:
- LRU cache implementation for search results
- Query normalization for cache keys
- Cache invalidation strategies
- Cache hit monitoring and analytics
- Cache size management

**Methods to Implement**:
```python
class SearchCacheManager:
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = {}  # Will use OrderedDict for LRU behavior
        self._hit_stats = {"hits": 0, "misses": 0}
        self._last_cleanup = time.time()

    async def get(self, key: str) -> Optional[Any]:
        """Get an item from the cache if it exists and is not expired"""

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Add an item to the cache with optional custom TTL"""

    async def invalidate(self, key: str) -> bool:
        """Remove a specific item from the cache"""

    async def invalidate_by_pattern(self, pattern: str) -> int:
        """Remove items matching a pattern from the cache"""

    async def clear(self) -> None:
        """Clear the entire cache"""

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""

    def normalize_key(self, query: str, params: Dict[str, Any]) -> str:
        """Normalize a query and params into a consistent cache key"""

    async def _cleanup_expired(self) -> int:
        """Remove expired items from the cache"""

    def _enforce_size_limit(self) -> int:
        """Remove oldest items if cache exceeds max size"""
```

### 5. Entity Similarity Analysis (`src/game_loop/search/similarity.py`)

**Purpose**: Provide advanced similarity analysis between entities and for queries.

**Key Components**:
- Multiple similarity metrics implementation
- Entity-to-entity similarity computation
- Query-to-entity similarity analysis
- Similarity threshold management
- Performance optimization for similarity calculations

**Methods to Implement**:
```python
class EntitySimilarityAnalyzer:
    def __init__(self, embedding_registry, db_manager=None):
        self.registry = embedding_registry
        self.db_manager = db_manager
        self._similarity_cache = {}

    async def find_similar_entities(self, entity_id: str, top_k: int = 10,
                                   min_similarity: float = 0.7) -> List[Tuple[str, float]]:
        """Find entities similar to a given entity"""

    async def compute_entity_similarity(self, entity_id1: str, entity_id2: str) -> float:
        """Compute similarity between two specific entities"""

    async def compute_batch_similarities(self, entity_ids: List[str],
                                       reference_id: str) -> Dict[str, float]:
        """Compute similarities between multiple entities and a reference entity"""

    async def find_entity_clusters(self, entity_type: str = None,
                                  threshold: float = 0.8) -> List[List[str]]:
        """Find clusters of similar entities"""

    async def similarity_graph(self, entity_ids: List[str],
                             min_similarity: float = 0.7) -> Dict[str, List[Tuple[str, float]]]:
        """Generate a similarity graph between entities"""

    def cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""

    def euclidean_distance(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate euclidean distance between two embeddings"""

    def dot_product_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate dot product similarity between two embeddings"""

    def jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity between two sets of terms"""
```

### 6. Search API Integration (`src/game_loop/api/endpoints/search.py`)

**Purpose**: Expose search functionality through REST API endpoints.

**Key Components**:
- RESTful search endpoints
- Query parameter handling
- Search result formatting
- Pagination and filtering
- Error handling and rate limiting

**Methods to Implement**:
```python
class SearchEndpoints:
    def __init__(self, search_service):
        self.search_service = search_service

    async def search_entities(self, request: Request) -> Response:
        """Main search endpoint for entities"""

    async def search_by_similarity(self, request: Request) -> Response:
        """Search entities similar to a reference entity"""

    async def get_entity_context(self, request: Request) -> Response:
        """Get contextually relevant entities for a specific entity"""

    async def search_by_example(self, request: Request) -> Response:
        """Search using an example entity as the query"""

    async def get_search_suggestions(self, request: Request) -> Response:
        """Get search query suggestions based on partial input"""

    async def get_recent_searches(self, request: Request) -> Response:
        """Get user's recent searches"""

    def _parse_search_params(self, request: Request) -> Dict[str, Any]:
        """Parse and validate search parameters from request"""

    def _format_search_response(self, results: List[Dict[str, Any]],
                               format_type: str = "detailed") -> Dict[str, Any]:
        """Format search results for API response"""
```

### 7. Game Integration Module (`src/game_loop/search/game_integration.py`)

**Purpose**: Integrate search functionality with game systems.

**Key Components**:
- Game event listeners for search triggers
- Search result handlers for game actions
- Context generation from search results
- Search-based game mechanics
- Search analytics for gameplay insights

**Methods to Implement**:
```python
class SearchGameIntegrator:
    def __init__(self, search_service, game_state_manager):
        self.search_service = search_service
        self.game_state_manager = game_state_manager
        self._event_handlers = {}
        self._register_event_handlers()

    async def handle_player_search_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a player's search query within game context"""

    async def generate_contextual_search(self, current_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate search results based on current game context"""

    async def search_related_entities(self, entity_id: str, relation_type: str = "any") -> List[Dict[str, Any]]:
        """Find entities related to a specific entity by relationship type"""

    async def search_environment(self, location_id: str, query: str = None) -> List[Dict[str, Any]]:
        """Search for entities within a specific game location"""

    async def handle_search_triggered_event(self, search_result: Dict[str, Any],
                                          event_type: str) -> Dict[str, Any]:
        """Handle game events triggered by search results"""

    def _register_event_handlers(self) -> None:
        """Register handlers for search-related game events"""

    def _extract_search_context(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant search context from current game state"""

    def _apply_search_results_to_game_state(self, results: List[Dict[str, Any]],
                                          game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Update game state based on search results"""
```

## File Structure

```
src/game_loop/
├── search/
│   ├── __init__.py
│   ├── semantic_search.py      # Main semantic search service
│   ├── query_processor.py      # Query preprocessing and analysis
│   ├── results_processor.py    # Search results processing
│   ├── cache_manager.py        # Search cache management
│   ├── similarity.py           # Entity similarity analysis
│   ├── game_integration.py     # Integration with game systems
├── api/
│   ├── endpoints/
│   │   ├── search.py           # Search API endpoints
├── database/
│   ├── queries/
│   │   ├── search_queries.py   # Database queries for search
```

## Testing Strategy

### Unit Tests

1. **Semantic Search Service Tests** (`tests/unit/search/test_semantic_search.py`):
   - Test different search strategies
   - Test query preprocessing
   - Test result ranking
   - Test threshold filtering
   - Test performance metrics

2. **Query Processor Tests** (`tests/unit/search/test_query_processor.py`):
   - Test query normalization
   - Test entity type extraction
   - Test query expansion
   - Test intent classification
   - Test constraint extraction

3. **Results Processor Tests** (`tests/unit/search/test_results_processor.py`):
   - Test deduplication
   - Test result enrichment
   - Test relevance scoring
   - Test snippet generation
   - Test sorting and grouping

4. **Cache Manager Tests** (`tests/unit/search/test_cache_manager.py`):
   - Test cache hit/miss
   - Test TTL expiration
   - Test size limits
   - Test invalidation
   - Test key normalization

5. **Similarity Analysis Tests** (`tests/unit/search/test_similarity.py`):
   - Test similarity metrics
   - Test entity clustering
   - Test similarity graph generation
   - Test batch similarity computation
   - Test similarity threshold handling

### Integration Tests

1. **Search API Integration Tests** (`tests/integration/api/test_search_api.py`):
   - Test API endpoints
   - Test parameter handling
   - Test result formatting
   - Test pagination
   - Test error handling

2. **Search-Database Integration Tests** (`tests/integration/search/test_search_database_integration.py`):
   - Test search with database backend
   - Test query performance
   - Test result consistency
   - Test with large datasets
   - Test database query optimizations

3. **Game System Integration Tests** (`tests/integration/search/test_search_game_integration.py`):
   - Test search in game context
   - Test search-triggered events
   - Test contextual search generation
   - Test search impact on game state
   - Test search-based interactions

### Performance Tests

1. **Search Performance Benchmark** (`tests/performance/test_search_performance.py`):
   - Test search latency
   - Test scaling with entity count
   - Test cache efficiency
   - Test memory usage
   - Test concurrent search requests

2. **Query Complexity Benchmark** (`tests/performance/test_query_complexity.py`):
   - Test performance across query complexity levels
   - Test optimization strategies
   - Test preprocessing overhead
   - Test strategy selection efficiency
   - Test with realistic query patterns

## Verification Criteria

### Functional Verification
- [x] Different search strategies (semantic, keyword, hybrid) produce expected results
- [x] Query preprocessing correctly handles different query formats
- [x] Result ranking produces intuitive ordering of search results
- [x] Similarity calculation works accurately for different metrics
- [x] Caching correctly improves performance for repeated queries
- [x] Search API endpoints function as expected
- [x] Game integration correctly utilizes search results

### Performance Verification
- [x] Simple semantic search completes in < 100ms for database with 10,000 entities
- [x] Cache hit rate > 95% for repeated queries
- [x] Batch similarity calculation processes 100 entities in < 500ms
- [x] Memory usage remains stable during sustained search operations
- [x] Query preprocessing overhead < 10% of total search time
- [x] API endpoints handle 10+ concurrent search requests efficiently
- [x] Game integration adds < 50ms overhead to search operations

### Integration Verification
- [x] Works seamlessly with EmbeddingDatabaseManager from Commit 15
- [x] Integrates with EntityEmbeddingRegistry and EntityEmbeddingGenerator
- [x] Search API endpoints integrate with existing API framework
- [x] Game systems can trigger and consume search operations
- [x] Search results can influence game state appropriately
- [x] Search operations respect database performance constraints
- [x] Caching works with existing application cache infrastructure

## Dependencies

### New Dependencies
- None (uses existing PostgreSQL vector capabilities from Commit 15)

### Configuration Updates
- Add search strategy configuration
- Add cache size and TTL parameters
- Add similarity metric configuration
- Add search API rate limiting configuration
- Add game integration event mappings

## Integration Points

1. **With Database System**: Query entity embeddings for similarity search
2. **With EntityEmbeddingRegistry/Generator**: Generate query embeddings and access entity embeddings
3. **With Game State Manager**: Integrate search with game context and state
4. **With API Framework**: Expose search functionality through API
5. **With Caching System**: Integrate with application-wide caching

## Migration Considerations

- No schema changes required from Commit 15
- Consider implementing progressive feature flags for search capabilities
- Plan for backward compatibility with existing entity querying mechanisms
- Provide migration path for code using direct database queries

## Code Quality Requirements

- [x] All code passes black, ruff, and mypy linting
- [x] Comprehensive docstrings for all public methods
- [x] Type hints for all function parameters and return values
- [x] Error handling for all search operations
- [x] Performance annotations for resource-intensive operations
- [x] Thorough logging for search operations and errors
- [x] Runtime performance monitoring

## Documentation Updates

- [x] Create semantic search overview documentation
- [x] Document search API endpoints
- [x] Add search strategy selection guide
- [x] Create search integration guide for game developers
- [x] Document performance optimization techniques
- [x] Add examples of common search queries and usage patterns
- [x] Update architecture documentation with search system

## Future Considerations

This semantic search implementation will serve as the foundation for:
- **Commit 17**: Search Integration with Game Loop (deeper integration with gameplay mechanics)
- **Future**: Natural language command processing using search as foundation
- **Future**: Advanced search-based gameplay mechanics and puzzles
- **Future**: Search analytics and player behavior insights
- **Future**: Federated search across multiple game instances
- **Future**: AI-driven search improvements based on player usage patterns

The design should be flexible enough to support these future enhancements while maintaining backward compatibility and performance.
