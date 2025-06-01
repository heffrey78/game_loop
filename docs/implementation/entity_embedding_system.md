# Entity Embedding System

## Overview

The Entity Embedding System is a specialized component of the Game Loop embedding infrastructure that generates, manages, and utilizes embeddings specifically optimized for game entities such as characters, locations, items, and events.

Built on top of the core embedding service (commit 13), the entity embedding system provides enhanced semantic understanding by applying entity-specific preprocessing, maintaining a specialized registry, and offering efficient similarity search capabilities.

## Components

### EntityEmbeddingGenerator

The `EntityEmbeddingGenerator` class is responsible for generating optimized embeddings for different game entity types. It extends the base embedding service with entity-specific preprocessing and metadata handling.

```python
# Generate an embedding for a character entity
character = {
    "id": "char1",
    "name": "Wizard",
    "description": "A powerful magic user",
    "personality": "Wise and mysterious"
}
embedding = await entity_generator.generate_entity_embedding(character, "character")
```

Key features:
- Type-specific preprocessing strategies
- Entity feature extraction
- Customized caching based on entity IDs
- Batch processing capabilities

### EntityEmbeddingRegistry

The `EntityEmbeddingRegistry` class provides a central repository for storing, retrieving, and searching entity embeddings.

```python
# Register an entity embedding
await registry.register_entity(
    "char1", "character", embedding, {"name": "Wizard"}
)

# Find similar entities
similar_entities = await registry.find_similar_entities(embedding, top_k=5)
```

Key features:
- Fast similarity search
- Entity type filtering
- Metadata storage
- Import/export capabilities
- Version tracking

### Similarity Search

The similarity module provides functions for calculating and comparing embedding similarity between entities.

```python
# Search for similar entities
results = await search_entities(query_embedding, entities_embeddings, top_k=5)

# Calculate similarity between two embeddings
similarity = cosine_similarity(embedding1, embedding2)
```

Supported similarity metrics:
- Cosine similarity
- Euclidean distance
- Dot product
- Context-boosted similarity

### Analytics and Visualization

The analytics module offers tools for analyzing and visualizing entity embeddings.

```python
# Reduce embeddings to 2D for visualization
reduced = reduce_dimensions(embeddings, method="pca", dimensions=2)

# Generate a full embedding report
await generate_embedding_report(registry, Path("report.json"))
```

Key capabilities:
- Dimensionality reduction (PCA)
- Clustering
- Statistics generation
- Visualization generation

## Entity Type-Specific Preprocessing

The system applies different preprocessing strategies based on entity type to optimize embedding quality:

### Characters
- Emphasizes personality traits and motivations
- Includes relationship information
- Preserves character background and role

### Locations
- Prioritizes environment and atmosphere descriptions
- Includes landmark information
- Preserves connectivity to other locations

### Items
- Focuses on properties and usage information
- Includes origin and value data
- Emphasizes functional aspects

### Events
- Highlights timeline and consequences
- Includes participant information
- Emphasizes significance and impact

## Integration Points

The Entity Embedding System integrates with multiple game components:

1. **Game State Manager**: For entity registration and updates
2. **Database System**: For persistent storage of embeddings
3. **Search API**: For finding similar entities based on semantic meaning
4. **LLM Services**: For generating coherent responses based on entity similarity

## Future Development

The current implementation establishes the foundation for enhanced entity understanding. Future commits will build upon this foundation:

- **Commit 15**: Embedding Database Integration (storing embeddings in PostgreSQL)
- **Commit 16**: Semantic Search Implementation (comprehensive search API)
- **Commit 17**: Search Integration with Game Loop (connecting search to game mechanics)

## Usage Examples

### Basic Entity Embedding

```python
from game_loop.embeddings import EntityEmbeddingGenerator, EmbeddingService

# Initialize services
embedding_service = EmbeddingService(...)
entity_generator = EntityEmbeddingGenerator(embedding_service)

# Create an entity
character = {
    "id": "hero1",
    "name": "Hero",
    "description": "The main protagonist",
    "personality": "Brave and kind"
}

# Generate embedding
embedding = await entity_generator.generate_entity_embedding(character, "character")
```

### Entity Similarity Search

```python
from game_loop.embeddings import EntityEmbeddingRegistry

# Initialize registry
registry = EntityEmbeddingRegistry()

# Register entities
await registry.register_entity("hero1", "character", hero_embedding)
await registry.register_entity("villain1", "character", villain_embedding)
await registry.register_entity("sword1", "item", sword_embedding)

# Find similar entities to the hero
similar = await registry.find_similar_entities(hero_embedding, top_k=3)

# Find only similar characters
similar_chars = await registry.find_similar_entities(
    hero_embedding, entity_type="character", top_k=2
)
```

### Batch Processing

```python
# Process multiple entities in batch
entities = [character1, character2, location1]
entity_types = ["character", "character", "location"]

embeddings = await entity_generator.generate_entity_embeddings_batch(
    entities, entity_types
)
```

## Performance Considerations

- **Caching**: Entity embeddings should be cached to avoid regeneration
- **Batch Processing**: Use batch processing for multiple entities
- **Persistence**: Store embeddings in database for long-term use
- **Dimensionality**: Consider dimension reduction for large-scale visualization
