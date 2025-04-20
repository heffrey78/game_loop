"""
Embedding service for the Game Loop application.

This module implements the vector embedding generation capabilities
as specified in the embedding_pipeline.md document. It provides services
for converting text descriptions into vector embeddings that can be stored
in the PostgreSQL database (using pgvector) for semantic search operations.

The embedding service utilizes the Ollama API as described in the tech stack
for generating embeddings, which are then used throughout the game loop system.
"""

import logging
import re

import httpx

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Exception raised when embedding generation fails."""

    pass


class EmbeddingService:
    """
    Service for generating vector embeddings from text.

    This service connects to the Ollama API to generate embeddings
    as specified in the tech stack document. These embeddings are used
    for semantic search operations throughout the game loop system.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
    ):
        """
        Initialize the embedding service.

        Args:
            ollama_url: URL of the Ollama API service
            model: Name of the embedding model to use
        """
        self.ollama_url = ollama_url
        self.model = model
        logger.info(f"Initialized embedding service with model {model} at {ollama_url}")

    async def generate_embedding(self, text: str) -> list[float]:
        """
        Generate a vector embedding for the given text.

        This method sends the text to the Ollama API and retrieves
        a vector embedding that represents the semantic meaning of the text.

        Args:
            text: Text to generate embedding for

        Returns:
            Vector embedding as a list of floats (384-dimensional vector)

        Raises:
            EmbeddingError: If embedding generation fails
        """
        url = f"{self.ollama_url}/api/embeddings"

        # Clean and prepare text as per embedding_pipeline.md
        prepared_text = self._preprocess_text(text)

        try:
            # Request embedding from Ollama as specified in tech stack
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    json={"model": self.model, "prompt": prepared_text},
                    timeout=30.0,
                )

            if response.status_code != 200:
                raise EmbeddingError(f"Failed to generate embedding: {response.text}")

            data = response.json()

            # Validate the embedding
            embedding = data.get("embedding")
            if not embedding or not isinstance(embedding, list):
                raise EmbeddingError(f"Invalid embedding response: {data}")

            # Ensure we got the expected dimension (384 as per embedding_pipeline.md)
            if len(embedding) != 384:
                logger.warning(
                    f"Unexpected embedding dimension: {len(embedding)} (expected 384)"
                )

            return [float(val) for val in embedding]  # Ensure all values are floats

        except httpx.RequestError as e:
            raise EmbeddingError(f"Request to Ollama API failed: {e}") from e
        except Exception as e:
            raise EmbeddingError(f"Unexpected error generating embedding: {e}") from e

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for embedding generation.

        This method cleans and prepares the text for optimal embedding generation
        as specified in the embedding_pipeline.md document.

        Args:
            text: Raw text to preprocess

        Returns:
            Preprocessed text ready for embedding generation
        """
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text.strip())

        # Truncate if necessary (model has context limits)
        # As specified in embedding_pipeline.md
        if len(text) > 8192:
            text = text[:8192]

        return text

    async def check_connectivity(self) -> bool:
        """
        Check if the Ollama service is available and the model is ready.

        Returns:
            True if service is available, False otherwise
        """
        try:
            # Make a minimal request to check connectivity
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.ollama_url}/api/version", timeout=5.0
                )

            if response.status_code == 200:
                logger.info("Ollama service is available")
                return True

            logger.error(
                f"Ollama service check failed with status {response.status_code}"
            )
            return False

        except Exception as e:
            logger.error(f"Ollama service connectivity check failed: {e}")
            return False

    async def batch_generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in batch.

        This can be more efficient for processing multiple items
        as described in the Performance Considerations section
        of embedding_pipeline.md.

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            List of vector embeddings

        Raises:
            EmbeddingError: If batch embedding generation fails
        """
        embeddings = []

        for text in texts:
            embedding = await self.generate_embedding(text)
            embeddings.append(embedding)

        return embeddings
