"""
Embeddings module for Game Loop.

This module provides functionality for generating and managing vector
embeddings for game entities as described in the embedding_pipeline.md
document.

The embedding system is a key part of the architecture as shown in the
architecture diagram, enabling semantic search capabilities,
as well as contextual relevance determination.
"""

from game_loop.embeddings.analytics import (
    cluster_embeddings,
    compute_embedding_stats,
    reduce_dimensions,
    visualize_embeddings,
)
from game_loop.embeddings.cache import EmbeddingCache
from game_loop.embeddings.config import EmbeddingConfig
from game_loop.embeddings.entity_embeddings import (
    EntityEmbeddingGenerator as EntityEmbeddingGeneratorV2,
)
from game_loop.embeddings.entity_generator import EntityEmbeddingGenerator
from game_loop.embeddings.entity_preprocessing import (
    create_entity_context,
    extract_salient_features,
    preprocess_character,
    preprocess_event,
    preprocess_item,
    preprocess_location,
)
from game_loop.embeddings.entity_registry import EntityEmbeddingRegistry
from game_loop.embeddings.exceptions import (
    EmbeddingCacheError,
    EmbeddingConfigError,
    EmbeddingError,
    EmbeddingGenerationError,
    EmbeddingPreprocessingError,
    with_retry,
    with_retry_async,
)
from game_loop.embeddings.manager import EmbeddingManager
from game_loop.embeddings.preprocessing import (
    chunk_text,
    clean_text,
    enrich_context,
    normalize_text,
    preprocess_for_embedding,
)
from game_loop.embeddings.service import EmbeddingService
from game_loop.embeddings.similarity import (
    boost_by_context,
    cosine_similarity,
    dot_product,
    euclidean_distance,
    search_entities,
)

__all__ = [
    # Core service
    "EmbeddingService",
    "EmbeddingConfig",
    # Cache
    "EmbeddingCache",
    # Preprocessing
    "preprocess_for_embedding",
    "normalize_text",
    "enrich_context",
    "chunk_text",
    "clean_text",
    # Exceptions
    "EmbeddingError",
    "EmbeddingGenerationError",
    "EmbeddingCacheError",
    "EmbeddingConfigError",
    "EmbeddingPreprocessingError",
    "with_retry",
    "with_retry_async",
    # Entity handling
    "EntityEmbeddingGenerator",
    "EntityEmbeddingGeneratorV2",
    "EntityEmbeddingRegistry",
    "EmbeddingManager",
    # Entity preprocessing
    "preprocess_character",
    "preprocess_location",
    "preprocess_item",
    "preprocess_event",
    "extract_salient_features",
    "create_entity_context",
    # Similarity
    "cosine_similarity",
    "euclidean_distance",
    "dot_product",
    "search_entities",
    "boost_by_context",
    # Analytics
    "compute_embedding_stats",
    "reduce_dimensions",
    "cluster_embeddings",
    "visualize_embeddings",
]
