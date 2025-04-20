"""
Service for generating and managing embeddings using the OllamaClient.
"""

import asyncio
import logging

from ..llm.config import ConfigManager
from ..llm.ollama.client import OllamaClient, OllamaEmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Exception raised for errors in embedding generation or processing."""

    pass


class EmbeddingService:
    """Service for generating and managing embeddings."""

    def __init__(self, config_manager: ConfigManager | None = None):
        """Initialize the embedding service."""
        self.config = config_manager or ConfigManager()
        self.client: OllamaClient | None = None
        self._embedding_cache: dict[str, list[float]] = {}

    async def _get_client(self) -> OllamaClient:
        """Get or create an Ollama client."""
        if self.client is None:
            self.client = OllamaClient(
                base_url=self.config.llm_config.base_url,
                timeout=self.config.llm_config.timeout,
            )
        return self.client

    async def generate_embedding(
        self, text: str, model: str | None = None, use_cache: bool = True
    ) -> list[float]:
        """
        Generate an embedding for the given text.

        Args:
            text: The text to embed
            model: The model to use (defaults to config)
            use_cache: Whether to use cached embeddings

        Returns:
            List of embedding values
        """
        # Check cache first if enabled
        cache_key = f"{model or self.config.llm_config.embedding_model}:{text}"
        if use_cache and cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        # Create embedding config
        embedding_config = OllamaEmbeddingConfig(
            model=model or self.config.llm_config.embedding_model,
            dimensions=self.config.llm_config.embedding_dimensions,
        )

        # Get client and generate embedding
        client = await self._get_client()
        try:
            embedding = await client.generate_embeddings(text, embedding_config)

            # Cache the result if caching is enabled
            if use_cache:
                self._embedding_cache[cache_key] = embedding

            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    async def generate_batch_embeddings(
        self, texts: list[str], model: str | None = None, use_cache: bool = True
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            model: The model to use (defaults to config)
            use_cache: Whether to use cached embeddings

        Returns:
            List of embedding vectors
        """
        # Process embeddings concurrently
        tasks = [self.generate_embedding(text, model, use_cache) for text in texts]
        return await asyncio.gather(*tasks)

    async def close(self) -> None:
        """Close the client connection."""
        if self.client is not None:
            await self.client.close()
            self.client = None

    async def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()

    async def check_model_availability(self, model: str | None = None) -> bool:
        """
        Check if the specified model is available.

        Args:
            model: The model to check (defaults to config)

        Returns:
            True if the model is available
        """
        model_name = model or self.config.llm_config.embedding_model
        client = await self._get_client()
        return await client.check_model_availability(model_name)
