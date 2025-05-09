# Embedding Generation Pipeline

This document details the process of generating and managing vector embeddings throughout the Game Loop system. These embeddings are crucial for semantic search operations and contextual relevance determination.

## Overview

The embedding pipeline converts textual descriptions of game entities (locations, objects, NPCs, etc.) into fixed-length vector representations that capture semantic meaning. These vectors enable similarity searches and help the system find contextually relevant content.

```
Text Description → Preprocessing → Embedding Generation → Vector Storage → Semantic Search
```

## Architecture Components

### 1. Embedding Service

The Embedding Service is responsible for generating vectors from text using the Ollama API:

```python
class EmbeddingService:
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        self.ollama_url = ollama_url
        self.model = model

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate a vector embedding for the given text."""
        url = f"{self.ollama_url}/api/embeddings"

        # Clean and prepare text
        prepared_text = self._preprocess_text(text)

        # Request embedding from Ollama
        response = await httpx.post(
            url,
            json={
                "model": self.model,
                "prompt": prepared_text
            },
            timeout=30.0
        )

        if response.status_code != 200:
            raise EmbeddingError(f"Failed to generate embedding: {response.text}")

        data = response.json()
        return data["embedding"]

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for embedding generation."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Truncate if necessary (model has context limits)
        if len(text) > 8192:
            text = text[:8192]
        return text
```

### 2. Entity Embedding Generator

This component handles the embedding generation for specific entity types:

```python
class EntityEmbeddingGenerator:
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service

    async def generate_location_embedding(self, location: dict) -> list[float]:
        """Generate embedding for a location."""
        # Combine relevant fields for richer context
        text = f"{location['name']}. {location['short_desc']} {location['full_desc']}"
        if location.get('region_name'):
            text += f" This location is in the {location['region_name']} region."

        return await self.embedding_service.generate_embedding(text)

    async def generate_object_embedding(self, object: dict) -> list[float]:
        """Generate embedding for an object."""
        text = f"{object['name']}. {object['short_desc']} {object['full_desc']}"

        return await self.embedding_service.generate_embedding(text)

    # Similar methods for other entity types...
```

### 3. Embedding Manager

The Embedding Manager coordinates embedding generation and database storage:

```python
class EmbeddingManager:
    def __init__(
        self,
        db_session: AsyncSession,
        embedding_generator: EntityEmbeddingGenerator
    ):
        self.db_session = db_session
        self.embedding_generator = embedding_generator

    async def create_or_update_location_embedding(self, location_id: UUID) -> None:
        """Create or update the embedding for a location."""
        # Fetch the location data
        location = await self.db_session.get(Location, location_id)
        if not location:
            raise ValueError(f"Location {location_id} not found")

        # Generate embedding
        embedding = await self.embedding_generator.generate_location_embedding(location.to_dict())

        # Update the location with the new embedding
        location.location_embedding = embedding
        await self.db_session.commit()

    # Similar methods for other entity types...

    async def batch_process_embeddings(self, entity_type: str, batch_size: int = 50) -> None:
        """Batch process embeddings for an entity type."""
        # Implementation for batch processing...
```

## Integration Points

### 1. Content Creation/Modification Hooks

Embeddings are automatically generated at these key points:

1. **New Entity Creation**:
   - When a new location, object, NPC, or knowledge is created
   - When dynamic content is generated by the LLM

2. **Content Updates**:
   - When descriptions are modified
   - When significant state changes occur

```python
# Example hook in world creation service
async def create_new_location(self, location_data: dict) -> Location:
    """Create a new location and generate its embedding."""
    # Create location in database
    location = Location(**location_data)
    self.db_session.add(location)
    await self.db_session.flush()

    # Generate and store embedding
    await self.embedding_manager.create_or_update_location_embedding(location.id)

    return location
```

### 2. Batch Processing

For initial data loading or migrations, batch processing is used:

```python
# In a migration script or data loading utility
async def migrate_embeddings():
    """Update all embeddings in the database."""
    embedding_manager = get_embedding_manager()

    # Process in batches
    await embedding_manager.batch_process_embeddings("locations")
    await embedding_manager.batch_process_embeddings("objects")
    await embedding_manager.batch_process_embeddings("npcs")
    # And so on...
```

## Semantic Search Integration

The generated embeddings enable powerful semantic search capabilities:

```python
async def search_similar_locations(query_text: str, limit: int = 5) -> list[Location]:
    """Find locations similar to the query."""
    # Generate query embedding
    query_embedding = await embedding_service.generate_embedding(query_text)

    # Perform vector similarity search
    stmt = select(Location).order_by(
        Location.location_embedding.cosine_distance(query_embedding)
    ).limit(limit)

    result = await db_session.execute(stmt)
    return result.scalars().all()
```

## Performance Considerations

1. **Caching Strategy**:
   - Cache frequently used embeddings in memory
   - Use a time-based invalidation strategy

2. **Batch Processing**:
   - Generate embeddings in batches for bulk operations
   - Use background tasks for non-blocking operation

3. **Model Optimization**:
   - Use quantized embedding models for speed
   - Consider model size vs. accuracy tradeoffs

4. **Resource Management**:
   - Monitor memory usage during batch operations
   - Implement rate limiting for embedding generation

## Vector Dimensions

The system uses 384-dimension vectors as standard, which is the output dimension of the Nomic Embed Text model. This dimension provides a good balance of:

- Semantic representation power
- Storage efficiency
- Query performance

This ensures consistency with our PostgreSQL pgvector setup which expects 384-dimensional vectors.

## Error Handling and Recovery

The embedding generation pipeline includes robust error handling:

1. **Retries**: Automatic retry logic for transient failures
2. **Fallbacks**: Default embeddings when generation fails
3. **Logging**: Detailed error tracking for failed embeddings
4. **Validation**: Vector dimension and quality checks before storage

## Monitoring and Metrics

The following metrics are tracked for the embedding pipeline:

1. **Generation Time**: Average time to generate embeddings
2. **Success Rate**: Percentage of successful embedding generations
3. **Queue Length**: Number of pending embedding generation requests
4. **Vector Quality**: Distribution of vector norms and other quality metrics

These metrics help identify bottlenecks and ensure the system is performing optimally.
