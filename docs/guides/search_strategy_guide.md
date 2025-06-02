# Search Strategy Selection Guide

## Introduction

The semantic search system offers multiple search strategies, each with its own strengths and ideal use cases. This guide helps game developers choose the right strategy for different scenarios and explains how to optimize search performance.

## Available Search Strategies

### 1. Semantic Search (`semantic`)

**Description:** Uses vector embeddings to match queries based on meaning rather than exact keywords.

**Strengths:**
- Understands natural language queries
- Finds conceptually related content even with different terminology
- Works well with complex, descriptive queries
- Handles synonyms and related concepts naturally

**Limitations:**
- May miss exact keyword matches if semantically dissimilar
- Computationally more intensive than keyword search
- Depends on quality of embedding model

**Best for:**
- Natural language player queries
- Exploratory searches
- Finding entities conceptually related to a theme
- Handling queries with ambiguous terminology

**Example Use Cases:**
- "Find something that can help me defeat fire enemies"
- "I need a tool for climbing mountains"
- "Show me items related to healing and recovery"

### 2. Keyword Search (`keyword`)

**Description:** Traditional search based on matching specific keywords in entity data.

**Strengths:**
- Fast and efficient
- Works well for known terminology
- Precise when exact terms are known
- Less computationally intensive

**Limitations:**
- Misses synonyms and related concepts
- Sensitive to spelling and phrasing
- May return irrelevant results with keyword overlap

**Best for:**
- Specific, well-defined queries
- When players know exact terminology
- Quick lookups of named entities
- When exact matching is required

**Example Use Cases:**
- "Silver sword"
- "Health potion"
- "Goblin village"

### 3. Hybrid Search (`hybrid`)

**Description:** Combines semantic and keyword approaches with configurable weighting.

**Strengths:**
- Balances meaning and keyword precision
- More robust than either strategy alone
- Adaptable to different query types
- Combines strengths of both approaches

**Limitations:**
- Slightly more complex to configure optimally
- May dilute strengths of individual strategies
- Requires tuning for best results

**Best for:**
- General-purpose search (default choice)
- Mixed queries with both concepts and specific terms
- When query intent is unclear
- Balancing recall and precision

**Example Use Cases:**
- "Powerful fire sword"
- "Ancient ruins with hidden treasure"
- "Fast healing items for combat"

### 4. Exact Match Search (`exact`)

**Description:** Finds entities with fields that exactly match the search query.

**Strengths:**
- Highest precision
- Very fast for database indexes
- Deterministic results
- Simple to implement and debug

**Limitations:**
- No tolerance for variations
- Misses related content
- Requires exact knowledge
- Limited flexibility

**Best for:**
- ID lookups
- Reference searches
- When perfect precision is required
- Admin tools and debug features

**Example Use Cases:**
- "Excalibur" (exact item name)
- "Level 5 Health Potion"
- "Castle Blackreach" (exact location name)

## Strategy Selection Guidelines

### Automatic Strategy Selection

When using `strategy="auto"`, the system automatically selects a strategy based on query characteristics:

1. **Short queries (1-2 words)**: Favors exact match or keyword search
2. **Queries with quotes**: Uses keyword search to respect exact phrases
3. **Longer queries (3+ words)**: Favors semantic search
4. **Medium-length queries**: Uses hybrid approach

### Manual Selection Considerations

When manually selecting a strategy, consider:

1. **Query Specificity**
   - Specific terminology → Keyword/Exact
   - Conceptual/descriptive → Semantic
   - Mix of both → Hybrid

2. **Response Time Requirements**
   - Need fastest response → Keyword/Exact
   - Can tolerate slight delay → Semantic/Hybrid

3. **Result Precision vs. Recall**
   - Need high precision → Exact/Keyword
   - Need high recall → Semantic
   - Need balance → Hybrid

4. **User Knowledge Level**
   - Expert users (know terminology) → Keyword/Exact
   - Novice users → Semantic
   - Mixed user base → Hybrid

## Performance Optimization

### Semantic Search Optimization
- Use appropriate embedding dimension for your content
- Consider dimensionality reduction for very large entity sets
- Use clustering for initial filtering
- Implement approximated nearest neighbor when entity count > 100,000

### Keyword Search Optimization
- Create proper database indexes on searchable fields
- Use trigram indexing for partial matching
- Consider full-text search extensions

### Hybrid Search Optimization
- Tune semantic_weight parameter (default 0.7)
- Consider parallel execution of strategies
- Adjust weighting based on query characteristics

## Setting Search Strategy

### Via API

```
GET /api/search?query=magic+sword&strategy=semantic
```

### Via Search Service

```python
results = await search_service.search(
    query="magic sword",
    entity_types=["item", "weapon"],
    strategy="semantic",
    top_k=10
)
```

### Dynamic Strategy Selection

You can implement custom strategy selection logic:

```python
def select_strategy(query):
    if len(query.split()) <= 2:
        return "keyword"
    elif "\"" in query:
        return "keyword"  # Use keyword for quoted phrases
    elif query.isupper():
        return "exact"    # All caps might indicate exact lookup
    elif len(query.split()) >= 5:
        return "semantic" # Longer queries benefit from semantic
    else:
        return "hybrid"   # Default for most queries
```

## Monitoring and Tuning

The search system tracks performance metrics for each strategy:

```python
metrics = search_service.get_search_metrics()
print(f"Average search time: {metrics['avg_search_time']}")
print(f"Semantic search avg: {metrics['avg_times_by_strategy']['semantic']}")
```

Use these metrics to tune strategy selection and adjust search parameters based on real-world usage patterns.
