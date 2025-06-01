"""
Entity Embedding Registry for managing and searching entity embeddings.

This module provides a registry for storing, retrieving, and searching
entity embeddings based on their semantic similarity.
"""

import logging
import pickle
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class EntityEmbeddingRegistry:
    """Registry for managing and searching entity embeddings."""

    def __init__(self, dimension: int = 384):
        """
        Initialize the entity embedding registry.

        Args:
            dimension: Dimensionality of the entity embeddings
        """
        self.dimension = dimension
        self.embeddings: dict[str, list[float]] = (
            {}
        )  # Map of entity_id -> embedding vector
        self.metadata: dict[str, dict] = {}  # Map of entity_id -> metadata dict
        self.entity_types: dict[str, str] = {}  # Map of entity_id -> entity_type
        self._version_map: dict[str, int] = {}  # Map of entity_id -> version
        self._last_updated: dict[str, float] = {}  # Map of entity_id -> timestamp
        self._index = None  # Will store an index for fast similarity search

    async def register_entity(
        self,
        entity_id: str,
        entity_type: str,
        embedding: list[float],
        metadata: dict | None = None,
    ) -> None:
        """
        Register an entity embedding in the registry.

        Args:
            entity_id: Unique identifier for the entity
            entity_type: Type of the entity (character, location, etc.)
            embedding: The embedding vector
            metadata: Additional metadata to store with the embedding

        Raises:
            ValueError: If embedding dimension doesn't match registry dimension
        """
        if len(embedding) != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, "
                f"got {len(embedding)}"
            )

        # Store the embedding and metadata
        self.embeddings[entity_id] = embedding
        self.entity_types[entity_id] = entity_type.lower()
        self.metadata[entity_id] = metadata or {}
        self._last_updated[entity_id] = time.time()

        # Update version if provided in metadata
        if metadata and "version" in metadata:
            self._version_map[entity_id] = metadata["version"]

        # Reset the search index since data changed
        self._index = None

        logger.debug(f"Registered entity {entity_id} of type {entity_type} in registry")

    async def get_entity_embedding(self, entity_id: str) -> list[float] | None:
        """
        Retrieve an entity's embedding by ID.

        Args:
            entity_id: Unique identifier for the entity

        Returns:
            The entity embedding, or None if not found
        """
        return self.embeddings.get(entity_id)

    async def find_similar_entities(
        self,
        query_embedding: list[float],
        entity_type: str | None = None,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Find entities similar to the given embedding.

        Args:
            query_embedding: The embedding to compare against
            entity_type: Optional filter for entity type
            top_k: Number of results to return

        Returns:
            List of (entity_id, similarity_score) tuples

        Raises:
            ValueError: If query_embedding dimension doesn't match registry dimension
        """
        if len(query_embedding) != self.dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.dimension}, "
                f"got {len(query_embedding)}"
            )

        # Convert query to numpy array for efficient computation
        query_np = np.array(query_embedding)

        # Calculate similarities for all embeddings
        similarities = []
        for entity_id, embedding in self.embeddings.items():
            # Filter by entity type if specified
            if entity_type and self.entity_types.get(entity_id) != entity_type.lower():
                continue

            # Calculate cosine similarity
            embedding_np = np.array(embedding)
            similarity = self._cosine_similarity(query_np, embedding_np)
            similarities.append((entity_id, similarity))

        # Sort by similarity (descending) and return top_k results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    async def update_entity_embedding(
        self,
        entity_id: str,
        new_embedding: list[float],
        metadata: dict | None = None,
    ) -> None:
        """
        Update an existing entity's embedding.

        Args:
            entity_id: Unique identifier for the entity
            new_embedding: The new embedding vector
            metadata: Updated metadata (optional)

        Raises:
            ValueError: If entity_id doesn't exist or embedding dimension doesn't match
        """
        if entity_id not in self.embeddings:
            raise ValueError(f"Entity {entity_id} not found in registry")

        if len(new_embedding) != self.dimension:
            raise ValueError(
                f"New embedding dimension mismatch: expected {self.dimension}, "
                f"got {len(new_embedding)}"
            )

        # Store the updated embedding and metadata
        self.embeddings[entity_id] = new_embedding
        self._last_updated[entity_id] = time.time()

        # Update metadata if provided
        if metadata:
            self.metadata[entity_id].update(metadata)
            if "version" in metadata:
                self._version_map[entity_id] = metadata["version"]

        # Reset the search index since data changed
        self._index = None

        logger.debug(f"Updated entity {entity_id} in registry")

    async def remove_entity(self, entity_id: str) -> bool:
        """
        Remove an entity from the registry.

        Args:
            entity_id: Unique identifier for the entity

        Returns:
            True if entity was removed, False if not found
        """
        if entity_id not in self.embeddings:
            return False

        # Remove all data for this entity
        self.embeddings.pop(entity_id)
        self.metadata.pop(entity_id, None)
        self.entity_types.pop(entity_id, None)
        self._version_map.pop(entity_id, None)
        self._last_updated.pop(entity_id, None)

        # Reset the search index since data changed
        self._index = None

        logger.debug(f"Removed entity {entity_id} from registry")
        return True

    def get_all_entity_ids(self, entity_type: str | None = None) -> list[str]:
        """
        Get all entity IDs in the registry, optionally filtered by type.

        Args:
            entity_type: Optional filter for entity type

        Returns:
            List of entity IDs
        """
        if entity_type:
            return [
                entity_id
                for entity_id, e_type in self.entity_types.items()
                if e_type == entity_type.lower()
            ]
        return list(self.embeddings.keys())

    async def export_registry(self, file_path: Path) -> None:
        """
        Export the registry to a file.

        Args:
            file_path: Path to save the registry data

        Raises:
            IOError: If writing to file fails
        """
        export_data = {
            "dimension": self.dimension,
            "embeddings": self.embeddings,
            "metadata": self.metadata,
            "entity_types": self.entity_types,
            "version_map": self._version_map,
            "last_updated": self._last_updated,
        }

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "wb") as f:
                pickle.dump(export_data, f)
            logger.info(
                f"Registry exported to {file_path} with {len(self.embeddings)} entities"
            )
        except Exception as e:
            logger.error(f"Failed to export registry: {e}")
            raise OSError(f"Failed to export registry: {e}") from e

    async def import_registry(self, file_path: Path) -> None:
        """
        Import registry data from a file.

        Args:
            file_path: Path to load the registry data from

        Raises:
            IOError: If reading from file fails
            ValueError: If imported data format is invalid
        """
        try:
            with open(file_path, "rb") as f:
                import_data = pickle.load(f)

            # Validate data format
            required_keys = ["dimension", "embeddings", "metadata", "entity_types"]
            if not all(key in import_data for key in required_keys):
                raise ValueError("Invalid registry data format")

            # Set registry properties
            self.dimension = import_data["dimension"]
            self.embeddings = import_data["embeddings"]
            self.metadata = import_data["metadata"]
            self.entity_types = import_data["entity_types"]
            self._version_map = import_data.get("version_map", {})
            self._last_updated = import_data.get("last_updated", {})

            # Reset the search index
            self._index = None

            logger.info(
                f"Registry imported from {file_path} with "
                f"{len(self.embeddings)} entities"
            )
        except Exception as e:
            logger.error(f"Failed to import registry: {e}")
            if isinstance(e, ValueError):
                raise
            raise OSError(f"Failed to import registry: {e}") from e

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (between -1 and 1)
        """
        # Avoid division by zero
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))
