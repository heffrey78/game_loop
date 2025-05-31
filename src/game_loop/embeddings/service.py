"""
Service for generating and managing embeddings using the OllamaClient.
"""

import asyncio
import logging
from typing import Any

from ..config.manager import ConfigManager
from ..llm.ollama.client import OllamaClient, OllamaEmbeddingConfig
from .cache import EmbeddingCache, create_text_hash
from .config import EmbeddingConfig
from .exceptions import (
    EmbeddingGenerationError,
    with_retry_async,
)
from .preprocessing import preprocess_for_embedding

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating and managing embeddings."""

    def __init__(
        self,
        config_manager: ConfigManager | None = None,
        embedding_config: EmbeddingConfig | None = None,
    ):
        """
        Initialize the embedding service.

        Args:
            config_manager: Configuration manager instance
            embedding_config: Embedding-specific configuration
        """
        self.config = config_manager or ConfigManager()
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.client: OllamaClient | None = None
        # Initialize cache
        self.cache = EmbeddingCache(
            memory_size=self.embedding_config.cache_size,
            enable_disk_cache=self.embedding_config.disk_cache_enabled,
        )

    async def _get_client(self) -> OllamaClient:
        """Get or create an Ollama client."""
        if self.client is None:
            self.client = OllamaClient(
                base_url=self.config.config.llm.base_url,
                timeout=self.config.config.llm.timeout,
            )
        return self.client

    async def generate_embedding(
        self, text: str, entity_type: str = "general"
    ) -> list[float]:
        """
        Generate an embedding for the given text.

        Args:
            text: The text to embed
            entity_type: Type of entity for specialized preprocessing

        Returns:
            List of embedding values

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not text or not text.strip():
            raise EmbeddingGenerationError("Cannot generate embedding for empty text")

        # Preprocess text if enabled
        processed_text = text
        if self.embedding_config.preprocessing_enabled:
            processed_text = preprocess_for_embedding(
                text, entity_type, self.embedding_config.max_text_length
            )

        # Create cache key
        cache_key = create_text_hash(
            processed_text, self.embedding_config.model_name, entity_type
        )

        # Check cache first if enabled
        if self.embedding_config.cache_enabled:
            cached_embedding = await self.cache.get(cache_key)
            if cached_embedding is not None:
                logger.debug(f"Cache hit for {entity_type} embedding")
                return cached_embedding

        # Generate embedding with retry logic
        try:
            embedding = await self._generate_with_retry(
                processed_text, self.embedding_config.retry_attempts
            )

            # Cache the result if caching is enabled
            if self.embedding_config.cache_enabled:
                await self.cache.set(cache_key, embedding)

            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding for {entity_type}: {e}")
            raise EmbeddingGenerationError(f"Embedding generation failed: {e}") from e

    async def generate_embeddings_batch(
        self,
        texts: list[str],
        entity_types: list[str] | None = None,
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            entity_types: List of entity types (one per text)

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If any embedding generation fails
        """
        if not texts:
            return []

        # Default entity types if not provided
        if entity_types is None:
            entity_types = ["general"] * len(texts)
        elif len(entity_types) != len(texts):
            raise EmbeddingGenerationError(
                f"Number of entity_types ({len(entity_types)}) must match "
                f"number of texts ({len(texts)})"
            )

        # Process in batches to avoid overwhelming the system
        batch_size = self.embedding_config.batch_size
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_types = entity_types[i : i + batch_size]

            # Generate embeddings for this batch
            batch_tasks = [
                self.generate_embedding(text, entity_type)
                for text, entity_type in zip(batch_texts, batch_types, strict=True)
            ]

            try:
                batch_embeddings = await asyncio.gather(*batch_tasks)
                all_embeddings.extend(batch_embeddings)

                # Small delay between batches to avoid overloading
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Failed to generate batch embeddings: {e}")
                raise EmbeddingGenerationError(f"Batch embedding failed: {e}") from e

        return all_embeddings

    async def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by the current model.

        Returns:
            Dimension of embedding vectors

        Raises:
            EmbeddingError: If dimension cannot be determined
        """
        try:
            # Generate a test embedding to determine dimension
            test_embedding = await self._generate_with_retry("test", max_retries=1)
            return len(test_embedding)

        except Exception as e:
            # Fall back to configuration value if available
            if hasattr(self.config.config.llm, "embedding_dimensions"):
                dimensions = self.config.config.llm.embedding_dimensions
                if dimensions is not None:
                    return dimensions

            logger.error(f"Failed to determine embedding dimension: {e}")
            raise EmbeddingGenerationError(
                f"Cannot determine embedding dimension: {e}"
            ) from e

    def preprocess_text(self, text: str, entity_type: str = "general") -> str:
        """
        Preprocess text for embedding generation.

        Args:
            text: Input text to preprocess
            entity_type: Type of entity for specialized preprocessing

        Returns:
            Preprocessed text
        """
        if not self.embedding_config.preprocessing_enabled:
            return text

        return preprocess_for_embedding(
            text, entity_type, self.embedding_config.max_text_length
        )

    async def _generate_with_retry(
        self, text: str, max_retries: int = 3
    ) -> list[float]:
        """
        Generate embedding with retry logic.

        Args:
            text: Text to generate embedding for
            max_retries: Maximum number of retry attempts

        Returns:
            Embedding vector

        Raises:
            EmbeddingGenerationError: If all retries fail
        """

        async def _generate() -> list[float]:
            # Create embedding config
            embedding_config = OllamaEmbeddingConfig(
                model=self.embedding_config.model_name,
                dimensions=getattr(
                    self.config.config.llm, "embedding_dimensions", None
                ),
            )

            # Get client and generate embedding
            client = await self._get_client()
            embedding = await client.generate_embeddings(text, embedding_config)

            if not embedding:
                raise EmbeddingGenerationError("Empty embedding returned from model")

            return embedding

        return await with_retry_async(
            _generate,
            max_retries=max_retries,
            delay=self.embedding_config.retry_delay,
        )

    async def close(self) -> None:
        """Close the client connection and cleanup resources."""
        if self.client is not None:
            await self.client.close()
            self.client = None

    async def clear_cache(self, cache_type: str = "all") -> None:
        """
        Clear the embedding cache.

        Args:
            cache_type: Type of cache to clear ("memory", "disk", or "all")
        """
        await self.cache.clear_cache(cache_type)

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get cache performance statistics.

        Returns:
            Dictionary with cache statistics
        """
        return self.cache.get_cache_stats()

    async def check_model_availability(self, model: str | None = None) -> bool:
        """
        Check if the specified model is available.

        Args:
            model: The model to check (defaults to config)

        Returns:
            True if the model is available
        """
        model_name = model or self.embedding_config.model_name
        client = await self._get_client()
        return await client.check_model_availability(model_name)
