"""
Embeddings module for Game Loop.

This module provides functionality for generating and managing vector
embeddings for game entities as described in the embedding_pipeline.md
document.

The embedding system is a key part of the architecture as shown in the
architecture diagram, enabling semantic search capabilities,
as well as contextual relevance determination.
"""

from game_loop.embeddings.cache import EmbeddingCache
from game_loop.embeddings.config import EmbeddingConfig
from game_loop.embeddings.entity_generator import EntityEmbeddingGenerator
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
    "EmbeddingManager",
]
