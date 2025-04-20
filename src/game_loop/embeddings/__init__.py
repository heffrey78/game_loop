"""
Embeddings module for Game Loop.

This module provides functionality for generating and managing vector embeddings
for game entities as described in the embedding_pipeline.md document.

The embedding system is a key part of the architecture as shown in the
architecture diagram, enabling semantic search capabilities,
as well as contextual relevance determination."""

from game_loop.embeddings.entity_generator import EntityEmbeddingGenerator
from game_loop.embeddings.manager import EmbeddingManager
from game_loop.embeddings.service import EmbeddingError, EmbeddingService

__all__ = [
    "EmbeddingService",
    "EmbeddingError",
    "EntityEmbeddingGenerator",
    "EmbeddingManager",
]
