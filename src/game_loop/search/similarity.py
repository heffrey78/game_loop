"""
Entity similarity analysis module for semantic search.

This module provides tools for analyzing similarity between entities
and performing advanced similarity operations.
"""

import logging
import math
from collections import defaultdict

from ..database.managers.embedding_manager import EmbeddingDatabaseManager
from ..embeddings.entity_registry import EntityEmbeddingRegistry

logger = logging.getLogger(__name__)


class EntitySimilarityAnalyzer:
    """Provide advanced similarity analysis between entities and for queries."""

    def __init__(
        self,
        embedding_registry: EntityEmbeddingRegistry,
        db_manager: EmbeddingDatabaseManager | None = None,
    ):
        """
        Initialize the entity similarity analyzer.

        Args:
            embedding_registry: Registry for entity embeddings
            db_manager: Optional database manager for larger-scale similarity operations
        """
        self.registry = embedding_registry
        self.db_manager = db_manager
        self._similarity_cache: dict[str, float] = {}

    async def find_similar_entities(
        self, entity_id: str, top_k: int = 10, min_similarity: float = 0.7
    ) -> list[tuple[str, float]]:
        """
        Find entities similar to a given entity.

        Args:
            entity_id: ID of the reference entity
            top_k: Maximum number of similar entities to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of (entity_id, similarity) tuples
        """
        # Get embedding for the reference entity
        reference_embedding = await self._get_entity_embedding(entity_id)
        if not reference_embedding:
            logger.warning(f"No embedding found for entity: {entity_id}")
            return []

        # Get entity type to restrict search (optional)
        entity_type = await self._get_entity_type(entity_id)

        # Get all embeddings for comparison
        embeddings = await self._get_comparable_embeddings(entity_type)

        # Remove the reference entity from comparison set
        if entity_id in embeddings:
            del embeddings[entity_id]

        # Calculate similarities
        similarities = []
        for other_id, embedding in embeddings.items():
            similarity = self.cosine_similarity(reference_embedding, embedding)
            if similarity >= min_similarity:
                similarities.append((other_id, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    async def compute_entity_similarity(
        self, entity_id1: str, entity_id2: str
    ) -> float:
        """
        Compute similarity between two specific entities.

        Args:
            entity_id1: First entity ID
            entity_id2: Second entity ID

        Returns:
            Similarity score (0-1)
        """
        # Check cache first
        cache_key = f"{entity_id1}:{entity_id2}"
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]

        # Reverse order cache check
        cache_key_reverse = f"{entity_id2}:{entity_id1}"
        if cache_key_reverse in self._similarity_cache:
            return self._similarity_cache[cache_key_reverse]

        # Get embeddings
        embedding1 = await self._get_entity_embedding(entity_id1)
        embedding2 = await self._get_entity_embedding(entity_id2)

        if not embedding1 or not embedding2:
            logger.warning(
                f"Missing embeddings for comparison: {entity_id1}, {entity_id2}"
            )
            return 0.0

        # Calculate similarity
        similarity = self.cosine_similarity(embedding1, embedding2)

        # Cache the result
        self._similarity_cache[cache_key] = similarity

        return similarity

    async def compute_batch_similarities(
        self, entity_ids: list[str], reference_id: str
    ) -> dict[str, float]:
        """
        Compute similarities between multiple entities and a reference entity.

        Args:
            entity_ids: List of entity IDs to compare
            reference_id: Reference entity ID

        Returns:
            Dictionary mapping entity IDs to similarity scores
        """
        # Get reference embedding
        reference_embedding = await self._get_entity_embedding(reference_id)
        if not reference_embedding:
            logger.warning(f"No embedding found for reference entity: {reference_id}")
            return {}

        # Get all comparison embeddings
        comparison_embeddings = {}
        for entity_id in entity_ids:
            if entity_id != reference_id:  # Skip self-comparison
                embedding = await self._get_entity_embedding(entity_id)
                if embedding:
                    comparison_embeddings[entity_id] = embedding

        # Calculate similarities
        similarities = {}
        for entity_id, embedding in comparison_embeddings.items():
            similarity = self.cosine_similarity(reference_embedding, embedding)
            similarities[entity_id] = similarity

            # Cache the result for later use
            cache_key = f"{reference_id}:{entity_id}"
            self._similarity_cache[cache_key] = similarity

        return similarities

    async def find_entity_clusters(
        self, entity_type: str | None = None, threshold: float = 0.8
    ) -> list[list[str]]:
        """
        Find clusters of similar entities.

        Args:
            entity_type: Optional entity type to restrict clustering
            threshold: Similarity threshold for cluster membership

        Returns:
            List of entity clusters (lists of entity IDs)
        """
        # Get embeddings to cluster
        embeddings = await self._get_comparable_embeddings(entity_type)
        entity_ids = list(embeddings.keys())

        # If very few entities, no need for clustering
        if len(entity_ids) <= 2:
            return [entity_ids]

        # Simple clustering algorithm (hierarchical clustering)
        # In a production system, you might use a more efficient algorithm

        # Initialize each entity as its own cluster
        clusters = [[entity_id] for entity_id in entity_ids]

        # Iteratively merge clusters
        merged = True
        while merged and len(clusters) > 1:
            merged = False

            # Check each pair of clusters
            for i in range(len(clusters)):
                if i >= len(clusters):
                    continue  # Skip if we've merged this cluster

                for j in range(i + 1, len(clusters)):
                    if j >= len(clusters):
                        continue  # Skip if we've merged this cluster

                    # Check if clusters should be merged
                    if await self._should_merge_clusters(
                        clusters[i], clusters[j], embeddings, threshold
                    ):
                        # Merge clusters
                        clusters[i].extend(clusters[j])
                        clusters.pop(j)
                        merged = True
                        break  # Break inner loop to restart

                if merged:
                    break  # Break outer loop to restart

        return clusters

    async def similarity_graph(
        self, entity_ids: list[str], min_similarity: float = 0.7
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Generate a similarity graph between entities.

        Args:
            entity_ids: List of entity IDs to include in graph
            min_similarity: Minimum similarity threshold for edges

        Returns:
            Graph as adjacency list mapping entity IDs to [(entity_id, similarity)]
        """
        graph = defaultdict(list)

        # Get all embeddings
        embeddings = {}
        for entity_id in entity_ids:
            embedding = await self._get_entity_embedding(entity_id)
            if embedding:
                embeddings[entity_id] = embedding

        # Build graph by comparing all pairs
        for i, entity_id1 in enumerate(embeddings.keys()):
            embedding1 = embeddings[entity_id1]

            for _, entity_id2 in enumerate(list(embeddings.keys())[i + 1 :], i + 1):
                embedding2 = embeddings[entity_id2]

                # Calculate similarity
                similarity = self.cosine_similarity(embedding1, embedding2)

                # Add edge if similarity is above threshold
                if similarity >= min_similarity:
                    graph[entity_id1].append((entity_id2, similarity))
                    graph[entity_id2].append(
                        (entity_id1, similarity)
                    )  # Undirected graph

        return dict(graph)

    def cosine_similarity(
        self, embedding1: list[float], embedding2: list[float]
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First vector embedding
            embedding2: Second vector embedding

        Returns:
            Cosine similarity (0-1)
        """
        # Check if vectors are same length
        if len(embedding1) != len(embedding2):
            logger.warning(
                f"Vectors have different dimensions: {len(embedding1)} vs "
                f"{len(embedding2)}"
            )
            min_len = min(len(embedding1), len(embedding2))
            embedding1 = embedding1[:min_len]
            embedding2 = embedding2[:min_len]

        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2, strict=False))

        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in embedding1))
        magnitude2 = math.sqrt(sum(b * b for b in embedding2))

        # Prevent division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        # Calculate cosine similarity
        similarity = dot_product / (magnitude1 * magnitude2)

        # Ensure result is between 0 and 1
        return max(0.0, min(1.0, similarity))

    def euclidean_distance(
        self, embedding1: list[float], embedding2: list[float]
    ) -> float:
        """
        Calculate euclidean distance between two embeddings.

        Args:
            embedding1: First vector embedding
            embedding2: Second vector embedding

        Returns:
            Euclidean distance (lower means more similar)
        """
        # Check if vectors are same length
        if len(embedding1) != len(embedding2):
            min_len = min(len(embedding1), len(embedding2))
            embedding1 = embedding1[:min_len]
            embedding2 = embedding2[:min_len]

        # Calculate squared differences
        squared_diff = sum(
            (a - b) ** 2 for a, b in zip(embedding1, embedding2, strict=False)
        )

        # Return square root of sum
        return math.sqrt(squared_diff)

    def dot_product_similarity(
        self, embedding1: list[float], embedding2: list[float]
    ) -> float:
        """
        Calculate dot product similarity between two embeddings.

        Args:
            embedding1: First vector embedding
            embedding2: Second vector embedding

        Returns:
            Dot product similarity
        """
        # Check if vectors are same length
        if len(embedding1) != len(embedding2):
            min_len = min(len(embedding1), len(embedding2))
            embedding1 = embedding1[:min_len]
            embedding2 = embedding2[:min_len]

        # Calculate dot product
        return sum(a * b for a, b in zip(embedding1, embedding2, strict=False))

    def jaccard_similarity(self, set1: set[str], set2: set[str]) -> float:
        """
        Calculate Jaccard similarity between two sets of terms.

        Args:
            set1: First set of terms
            set2: Second set of terms

        Returns:
            Jaccard similarity (0-1)
        """
        if not set1 or not set2:
            return 0.0

        # Calculate intersection and union
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        # Prevent division by zero
        if union == 0:
            return 0.0

        return intersection / union

    async def _get_entity_embedding(self, entity_id: str) -> list[float] | None:
        """
        Get embedding for an entity from registry or database.

        Args:
            entity_id: Entity ID

        Returns:
            Vector embedding or None if not found
        """
        # Try registry first
        if hasattr(self.registry, "get_embedding"):
            embedding = self.registry.get_embedding(entity_id)
            if embedding:
                return embedding  # type: ignore[no-any-return]

        # Fall back to database if available
        if self.db_manager:
            try:
                embedding = await self.db_manager.get_embedding(entity_id)
                return embedding  # type: ignore[no-any-return]
            except Exception as e:
                logger.error(f"Error getting embedding from database: {e}")

        return None

    async def _get_entity_type(self, entity_id: str) -> str | None:
        """
        Get entity type for an entity.

        Args:
            entity_id: Entity ID

        Returns:
            Entity type or None if not found
        """
        # Try registry first
        if hasattr(self.registry, "get_entity_type"):
            return self.registry.get_entity_type(entity_id)  # type: ignore[no-any-return]

        # Try getting entity data
        if hasattr(self.registry, "get_entity"):
            entity = self.registry.get_entity(entity_id)
            if entity and "entity_type" in entity:
                return entity["entity_type"]  # type: ignore[no-any-return]

        # Fall back to database if available
        if self.db_manager and hasattr(self.db_manager, "get_entity_type"):
            try:
                return await self.db_manager.get_entity_type(entity_id)  # type: ignore[no-any-return]
            except Exception as e:
                logger.error(f"Error getting entity type from database: {e}")

        return None

    async def _get_comparable_embeddings(
        self, entity_type: str | None = None
    ) -> dict[str, list[float]]:
        """
        Get embeddings for comparison, optionally filtered by type.

        Args:
            entity_type: Optional entity type to filter by

        Returns:
            Dictionary mapping entity IDs to embeddings
        """
        # Try registry first
        if hasattr(self.registry, "get_all_embeddings"):
            all_embeddings = self.registry.get_all_embeddings()

            # Filter by entity type if specified
            if entity_type and hasattr(self.registry, "get_entity_type"):
                return {
                    entity_id: embedding
                    for entity_id, embedding in all_embeddings.items()
                    if self.registry.get_entity_type(entity_id) == entity_type
                }
            return all_embeddings  # type: ignore[no-any-return]

        # Fall back to database if available
        if self.db_manager:
            try:
                if entity_type:
                    return await self.db_manager.get_embeddings_by_entity_type(
                        entity_type
                    )
                else:
                    # We need to implement this method in the database manager
                    # to get all embeddings across entity types
                    return await self.db_manager.get_all_embeddings()
            except Exception as e:
                logger.error(f"Error getting embeddings from database: {e}")

        return {}

    async def _should_merge_clusters(
        self,
        cluster1: list[str],
        cluster2: list[str],
        embeddings: dict[str, list[float]],
        threshold: float,
    ) -> bool:
        """
        Determine if two clusters should be merged based on similarity.

        Args:
            cluster1: First cluster (list of entity IDs)
            cluster2: Second cluster (list of entity IDs)
            embeddings: Dictionary mapping entity IDs to embeddings
            threshold: Similarity threshold for merging

        Returns:
            True if clusters should be merged
        """
        # Calculate average similarity between clusters
        similarities = []

        # Sample at most 5 entities from each cluster to reduce computation
        sample1 = cluster1[:5] if len(cluster1) > 5 else cluster1
        sample2 = cluster2[:5] if len(cluster2) > 5 else cluster2

        for entity_id1 in sample1:
            for entity_id2 in sample2:
                if entity_id1 in embeddings and entity_id2 in embeddings:
                    sim = self.cosine_similarity(
                        embeddings[entity_id1], embeddings[entity_id2]
                    )
                    similarities.append(sim)

        if not similarities:
            return False

        # Use average similarity for decision
        avg_similarity = sum(similarities) / len(similarities)
        return avg_similarity >= threshold
