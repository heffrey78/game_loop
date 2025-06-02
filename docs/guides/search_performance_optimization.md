# Search Performance Optimization Techniques

## Introduction

The semantic search system can be resource-intensive, particularly with large numbers of entities or complex queries. This document outlines techniques for optimizing search performance, reducing latency, and managing resource usage.

## Key Performance Metrics

When optimizing search performance, focus on these key metrics:

1. **Search latency:** Time from query submission to results return
2. **Memory usage:** RAM consumed during search operations
3. **CPU utilization:** Processing power required for search operations
4. **Cache efficiency:** How effectively the cache reduces database load
5. **Database load:** Query volume and complexity against the embedding database
6. **Concurrent search throughput:** Number of simultaneous searches handled efficiently

## Caching Strategies

### Result Caching

The `SearchCacheManager` provides built-in caching for search results:

```python
# Configure cache with appropriate size and TTL
cache_manager = SearchCacheManager(max_size=2000, ttl_seconds=300)

# Get cached result if available
cached_result = await cache_manager.get(cache_key)
if cached_result:
    return cached_result

# If not in cache, perform search and cache result
search_result = await perform_search(query, params)
await cache_manager.set(cache_key, search_result)
```

### Cache Tuning Recommendations

Adjust these parameters based on your game's specific needs:

| Game Type | Cache Size | TTL (seconds) | Notes |
|-----------|------------|--------------|-------|
| Small/Single Player | 500-1,000 | 600 | Emphasize memory efficiency |
| Medium Multiplayer | 2,000-5,000 | 300 | Balance between memory and hit rate |
| Large MMO | 10,000+ | 120-180 | Shorter TTL due to more frequent game state changes |

### Cache Invalidation Patterns

Implement these invalidation patterns to keep the cache consistent with game state:

```python
# Invalidate when an entity changes
async def on_entity_update(entity_id):
    await cache_manager.invalidate_by_pattern(f"*{entity_id}*")

# Invalidate location-related searches when location changes
async def on_location_change(location_id):
    await cache_manager.invalidate_by_pattern(f"*location:{location_id}*")

# Periodic full invalidation for globally changed state
async def daily_reset():
    await cache_manager.clear()
```

## Query Optimization

### Query Preprocessing

Optimize queries before execution:

1. **Normalize queries:** Remove redundant punctuation and normalize spacing
2. **Extract entity types:** Limit search to relevant entity types when possible
3. **Set appropriate limits:** Use the minimum necessary `top_k` value
4. **Use query complexity estimation:**

```python
query_complexity = query_processor.estimate_query_complexity(query)
if query_complexity > 0.8:
    # Use a more restricted search to compensate for complexity
    top_k = min(top_k, 20)
    timeout_ms = 200
else:
    timeout_ms = 100
```

### Strategy Selection

Choose the most efficient search strategy for each use case:

- **Keyword search:** Use for simple, specific queries when precise terms are known
- **Exact match:** Use when searching for specific entity attributes
- **Semantic search:** Reserve for natural language queries and conceptual searches
- **Hybrid search:** Use with appropriate weighting based on query characteristics

```python
# Example of adaptive strategy selection
def select_optimal_strategy(query, context):
    if query.count(' ') <= 1 and len(query) < 20:
        # Simple query, likely a specific term
        return "keyword"
    elif context.get("requires_precise_matching"):
        return "exact"
    elif query.count(' ') >= 5 or "?" in query:
        # Complex or question query
        return "semantic"
    else:
        # Default to hybrid with appropriate weighting
        return "hybrid"
```

## Database Optimization

### Vector Indexes

Ensure your PostgreSQL database has appropriate vector indexes:

```sql
-- Create vector index for vector similarity search
CREATE INDEX IF NOT EXISTS entity_embedding_vector_idx
ON entity_embeddings
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

### Query Batching

Batch related queries together to reduce database round trips:

```python
async def batch_entity_search(entity_ids):
    # Instead of querying each entity individually
    # Batch them into a single query
    query = """
    SELECT entity_id, embedding
    FROM entity_embeddings
    WHERE entity_id = ANY($1)
    """
    return await db.fetch_all(query, entity_ids)
```

### Connection Pooling

Configure appropriate database connection pools:

```python
# Configure connection pool size based on concurrent search volume
pool = await asyncpg.create_pool(
    dsn=DATABASE_URL,
    min_size=5,
    max_size=20
)
```

## Embedding Optimization

### Embedding Dimensionality

Balance accuracy vs. performance when selecting embedding dimensions:

| Dimension | Search Quality | Performance | Recommended Use Case |
|-----------|---------------|-------------|---------------------|
| 128-256 | Good | Very Fast | High performance requirements, simpler content |
| 384-512 | Better | Fast | General purpose, balanced approach |
| 768-1024 | Best | Slower | Complex content, quality critical |

### Embedding Quantization

Consider quantizing embeddings to reduce storage and improve performance:

```python
# Example of simple 8-bit quantization
def quantize_embedding(embedding, bits=8):
    scale = (2**bits - 1) / (max(embedding) - min(embedding))
    quantized = [int((x - min(embedding)) * scale) for x in embedding]
    return quantized, min(embedding), scale

def dequantize_embedding(quantized, min_val, scale):
    return [min_val + (q / scale) for q in quantized]
```

## Multi-Threading and Async Optimization

### Parallel Query Execution

Execute independent search operations in parallel:

```python
async def multi_search(query, entity_types):
    tasks = []
    for entity_type in entity_types:
        tasks.append(
            search_service.search(query, entity_types=[entity_type])
        )

    results = await asyncio.gather(*tasks)
    # Merge and process results
    return merge_results(results)
```

### Background Processing

Move non-critical search operations to background tasks:

```python
async def search_with_background_processing(query, context):
    # Perform critical search immediately
    primary_results = await search_service.search(query)

    # Queue additional processing in background
    asyncio.create_task(
        update_search_analytics(query, primary_results, context)
    )

    return primary_results
```

## Memory Management

### Result Pagination

Always use pagination for potentially large result sets:

```python
async def paginated_search(query, page=1, page_size=10):
    # Get total count and just the current page of results
    total_results = await search_service.count_results(query)

    # Calculate offsets
    offset = (page - 1) * page_size

    # Get just the needed page
    results = await search_service.search_with_pagination(
        query, offset=offset, limit=page_size
    )

    return {
        "results": results,
        "total": total_results,
        "page": page,
        "pages": math.ceil(total_results / page_size)
    }
```

### Selective Result Fields

Only retrieve and return necessary fields:

```python
async def minimal_search(query, include_fields=None):
    if not include_fields:
        include_fields = ["entity_id", "name", "type"]

    results = await search_service.search(
        query, include_fields=include_fields
    )

    return results
```

## Load Balancing and Rate Limiting

### API Rate Limiting

Implement appropriate rate limits for search API endpoints:

```python
class SearchRateLimiter:
    def __init__(self):
        self.limits = {
            "default": {"rate": 10, "per": 60},  # 10 per minute
            "premium": {"rate": 30, "per": 60},  # 30 per minute
        }
        self.user_counters = {}

    async def check_rate_limit(self, user_id, user_tier="default"):
        current_time = time.time()
        user_key = f"{user_id}:{user_tier}"

        if user_key not in self.user_counters:
            self.user_counters[user_key] = {
                "count": 0,
                "reset_at": current_time + self.limits[user_tier]["per"]
            }

        # Reset counter if time window expired
        if current_time > self.user_counters[user_key]["reset_at"]:
            self.user_counters[user_key] = {
                "count": 0,
                "reset_at": current_time + self.limits[user_tier]["per"]
            }

        # Check if limit reached
        if self.user_counters[user_key]["count"] >= self.limits[user_tier]["rate"]:
            return False

        # Increment counter
        self.user_counters[user_key]["count"] += 1
        return True
```

### Search Request Prioritization

Prioritize different types of search requests:

```python
class SearchPrioritizer:
    def __init__(self):
        self.priority_queues = {
            "high": asyncio.Queue(),
            "medium": asyncio.Queue(),
            "low": asyncio.Queue()
        }
        self.worker_task = asyncio.create_task(self._process_queues())

    async def enqueue_search(self, priority, search_func, *args, **kwargs):
        await self.priority_queues[priority].put((search_func, args, kwargs))

    async def _process_queues(self):
        while True:
            # First check high priority queue
            if not self.priority_queues["high"].empty():
                func, args, kwargs = await self.priority_queues["high"].get()
                await func(*args, **kwargs)
                continue

            # Then check medium priority
            if not self.priority_queues["medium"].empty():
                func, args, kwargs = await self.priority_queues["medium"].get()
                await func(*args, **kwargs)
                continue

            # Finally check low priority
            if not self.priority_queues["low"].empty():
                func, args, kwargs = await self.priority_queues["low"].get()
                await func(*args, **kwargs)
                continue

            # If all queues empty, wait a bit
            await asyncio.sleep(0.01)
```

## Monitoring and Performance Tracking

### Search Performance Metrics

Collect and analyze key search performance metrics:

```python
class SearchMetricsCollector:
    def __init__(self):
        self.metrics = {
            "total_searches": 0,
            "avg_latency_ms": 0,
            "cache_hit_rate": 0,
            "strategy_usage": {
                "semantic": 0,
                "keyword": 0,
                "hybrid": 0,
                "exact": 0
            },
            "error_rate": 0
        }

    async def record_search(self, strategy, latency_ms, cache_hit=False, error=False):
        self.metrics["total_searches"] += 1

        # Update latency
        old_avg = self.metrics["avg_latency_ms"]
        self.metrics["avg_latency_ms"] = (old_avg * (self.metrics["total_searches"] - 1) + latency_ms) / self.metrics["total_searches"]

        # Update strategy usage
        self.metrics["strategy_usage"][strategy] += 1

        # Update cache hit rate
        if cache_hit:
            self.metrics["cache_hit_rate"] = (
                self.metrics["cache_hit_rate"] * (self.metrics["total_searches"] - 1) + 1
            ) / self.metrics["total_searches"]

        # Update error rate
        if error:
            self.metrics["error_rate"] = (
                self.metrics["error_rate"] * (self.metrics["total_searches"] - 1) + 1
            ) / self.metrics["total_searches"]
```

## Optimization Decision Matrix

Use this decision matrix to guide optimization efforts:

| Symptom | Possible Cause | Recommended Optimizations |
|---------|---------------|--------------------------|
| High latency for all searches | Database performance | Improve indexes, optimize queries, increase connection pool |
| High latency for specific queries | Query complexity | Optimize query preprocessing, adjust strategy selection |
| Inconsistent latency | Cache inefficiency | Tune cache size and TTL, review invalidation logic |
| High memory usage | Result handling | Implement pagination, reduce field selection |
| High CPU usage | Vector operations | Consider embedding quantization, dimension reduction |
| Poor scaling under load | Concurrency issues | Implement request prioritization, rate limiting |

## Conclusion

Performance optimization should be an ongoing process guided by monitoring and measurement. Start with the techniques that address your specific bottlenecks and gradually implement additional optimizations as needed.

Remember that search performance directly impacts user experience, especially in real-time game scenarios where search operations may block gameplay. Balance performance optimization with search quality to provide the best experience for your players.

For more specific optimization recommendations, consult with the database and infrastructure teams, as many optimizations will depend on your specific deployment environment and scale.
