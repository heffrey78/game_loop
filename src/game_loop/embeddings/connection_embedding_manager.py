"""
Connection Embedding Manager for World Connection Management System.

This module generates and manages vector embeddings for connections to enable
semantic search and similarity analysis.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

import numpy as np

from game_loop.core.models.connection_models import GeneratedConnection
from game_loop.database.session_factory import DatabaseSessionFactory
from game_loop.embeddings.embedding_manager import EmbeddingManager

logger = logging.getLogger(__name__)


class ConnectionEmbeddingManager:
    """Generates and manages vector embeddings for connections."""

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        session_factory: DatabaseSessionFactory,
    ):
        """Initialize connection embedding manager."""
        self.embedding_manager = embedding_manager
        self.session_factory = session_factory
        self._embedding_cache: dict[UUID, list[float]] = {}

    async def generate_connection_embedding(
        self, connection: GeneratedConnection
    ) -> list[float]:
        """Generate embedding vector for connection."""
        try:
            # Check cache first
            if connection.connection_id in self._embedding_cache:
                return self._embedding_cache[connection.connection_id]

            # Create comprehensive text representation of the connection
            connection_text = self._create_connection_text(connection)

            # Generate embedding using the base embedding manager
            embedding = await self.embedding_manager.generate_embedding(
                text=connection_text
            )

            # Cache the embedding
            self._embedding_cache[connection.connection_id] = embedding

            logger.debug(
                f"Generated embedding for connection {connection.connection_id}"
            )
            return embedding

        except Exception as e:
            logger.error(f"Error generating connection embedding: {e}")
            # Return a default embedding vector
            return [0.0] * 1536  # Default OpenAI embedding size

    async def find_similar_connections(
        self,
        connection: GeneratedConnection,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[tuple[GeneratedConnection, float]]:
        """Find connections similar to the given connection."""
        try:
            # Generate embedding for the input connection
            target_embedding = await self.generate_connection_embedding(connection)

            # Get all cached embeddings
            similar_connections = []

            for cached_id, cached_embedding in self._embedding_cache.items():
                if cached_id == connection.connection_id:
                    continue  # Skip self

                # Calculate similarity
                similarity = self._calculate_cosine_similarity(
                    target_embedding, cached_embedding
                )

                if similarity >= threshold:
                    # Note: In a real implementation, we'd retrieve the connection from storage
                    # For now, we'll create a placeholder
                    similar_connection = self._create_placeholder_connection(cached_id)
                    similar_connections.append((similar_connection, similarity))

            # Sort by similarity (highest first)
            similar_connections.sort(key=lambda x: x[1], reverse=True)

            # Limit results
            return similar_connections[:limit]

        except Exception as e:
            logger.error(f"Error finding similar connections: {e}")
            return []

    async def cluster_connections_by_type(
        self, connection_type: str
    ) -> list[list[GeneratedConnection]]:
        """Cluster connections of same type for analysis."""
        try:
            # Get all connections of the specified type
            type_connections = []
            type_embeddings = []

            for conn_id, embedding in self._embedding_cache.items():
                # Note: In a real implementation, we'd query the storage to get connection details
                # For now, we'll simulate this
                placeholder_conn = self._create_placeholder_connection(conn_id)
                if placeholder_conn.properties.connection_type == connection_type:
                    type_connections.append(placeholder_conn)
                    type_embeddings.append(embedding)

            if len(type_connections) < 2:
                return [type_connections] if type_connections else []

            # Perform simple clustering based on similarity
            clusters = self._perform_simple_clustering(
                type_connections, type_embeddings
            )

            logger.info(
                f"Clustered {len(type_connections)} {connection_type} connections into {len(clusters)} clusters"
            )
            return clusters

        except Exception as e:
            logger.error(f"Error clustering connections by type {connection_type}: {e}")
            return []

    async def get_connection_similarity_matrix(
        self, connection_ids: list[UUID]
    ) -> dict[tuple[UUID, UUID], float]:
        """Generate similarity matrix for a set of connections."""
        try:
            similarity_matrix = {}

            # Get embeddings for all connections
            embeddings = {}
            for conn_id in connection_ids:
                if conn_id in self._embedding_cache:
                    embeddings[conn_id] = self._embedding_cache[conn_id]
                else:
                    logger.warning(f"No embedding found for connection {conn_id}")

            # Calculate pairwise similarities
            for i, conn_id1 in enumerate(connection_ids):
                for j, conn_id2 in enumerate(connection_ids):
                    if i >= j:  # Skip diagonal and duplicate pairs
                        continue

                    if conn_id1 in embeddings and conn_id2 in embeddings:
                        similarity = self._calculate_cosine_similarity(
                            embeddings[conn_id1], embeddings[conn_id2]
                        )
                        similarity_matrix[(conn_id1, conn_id2)] = similarity

            return similarity_matrix

        except Exception as e:
            logger.error(f"Error generating similarity matrix: {e}")
            return {}

    async def update_connection_embedding(
        self, connection: GeneratedConnection, force_regenerate: bool = False
    ) -> bool:
        """Update embedding for a connection."""
        try:
            if (
                force_regenerate
                or connection.connection_id not in self._embedding_cache
            ):
                embedding = await self.generate_connection_embedding(connection)
                connection.embedding_vector = embedding
                logger.info(
                    f"Updated embedding for connection {connection.connection_id}"
                )
                return True
            else:
                logger.debug(
                    f"Embedding already exists for connection {connection.connection_id}"
                )
                return False

        except Exception as e:
            logger.error(f"Error updating connection embedding: {e}")
            return False

    async def remove_connection_embedding(self, connection_id: UUID) -> bool:
        """Remove embedding from cache."""
        try:
            if connection_id in self._embedding_cache:
                del self._embedding_cache[connection_id]
                logger.debug(f"Removed embedding for connection {connection_id}")
                return True
            else:
                logger.warning(f"No embedding found for connection {connection_id}")
                return False

        except Exception as e:
            logger.error(f"Error removing connection embedding: {e}")
            return False

    async def get_embedding_statistics(self) -> dict[str, Any]:
        """Get statistics about connection embeddings."""
        try:
            stats = {
                "total_embeddings": len(self._embedding_cache),
                "cache_size_mb": self._calculate_cache_size(),
                "embedding_dimensions": (
                    len(next(iter(self._embedding_cache.values())))
                    if self._embedding_cache
                    else 0
                ),
            }

            # Analyze embedding patterns
            if self._embedding_cache:
                embeddings_array = np.array(list(self._embedding_cache.values()))
                stats["embedding_analysis"] = {
                    "mean_magnitude": float(
                        np.mean(np.linalg.norm(embeddings_array, axis=1))
                    ),
                    "std_magnitude": float(
                        np.std(np.linalg.norm(embeddings_array, axis=1))
                    ),
                    "min_value": float(np.min(embeddings_array)),
                    "max_value": float(np.max(embeddings_array)),
                }

            return stats

        except Exception as e:
            logger.error(f"Error getting embedding statistics: {e}")
            return {"error": str(e)}

    def _create_connection_text(self, connection: GeneratedConnection) -> str:
        """Create comprehensive text representation of connection for embedding."""
        try:
            # Build text from various connection attributes
            text_parts = []

            # Connection type and basic properties
            text_parts.append(
                f"Connection type: {connection.properties.connection_type}"
            )
            text_parts.append(f"Description: {connection.properties.description}")

            # Difficulty and travel information
            text_parts.append(f"Difficulty level: {connection.properties.difficulty}")
            text_parts.append(
                f"Travel time: {connection.properties.travel_time} seconds"
            )
            text_parts.append(f"Visibility: {connection.properties.visibility}")

            # Requirements and features
            if connection.properties.requirements:
                text_parts.append(
                    f"Requirements: {', '.join(connection.properties.requirements)}"
                )

            if connection.properties.special_features:
                text_parts.append(
                    f"Special features: {', '.join(connection.properties.special_features)}"
                )

            # Metadata information
            if "generation_purpose" in connection.metadata:
                text_parts.append(
                    f"Purpose: {connection.metadata['generation_purpose']}"
                )

            # Reversibility
            if connection.properties.reversible:
                text_parts.append("Bidirectional connection")
            else:
                text_parts.append("One-way connection")

            # Combine all parts
            connection_text = " | ".join(text_parts)

            return connection_text

        except Exception as e:
            logger.error(f"Error creating connection text: {e}")
            return f"Connection: {connection.properties.connection_type}"

    def _calculate_cosine_similarity(
        self, embedding1: list[float], embedding2: list[float]
    ) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0

    def _create_placeholder_connection(
        self, connection_id: UUID
    ) -> GeneratedConnection:
        """Create a placeholder connection for testing purposes."""
        from game_loop.core.models.connection_models import ConnectionProperties

        # This is a simplified placeholder - in real implementation,
        # we'd retrieve the actual connection from storage
        properties = ConnectionProperties(
            connection_type="passage",
            difficulty=2,
            travel_time=60,
            description="A placeholder connection",
            visibility="visible",
            requirements=[],
        )

        return GeneratedConnection(
            source_location_id=UUID("00000000-0000-0000-0000-000000000001"),
            target_location_id=UUID("00000000-0000-0000-0000-000000000002"),
            properties=properties,
            connection_id=connection_id,
        )

    def _perform_simple_clustering(
        self, connections: list[GeneratedConnection], embeddings: list[list[float]]
    ) -> list[list[GeneratedConnection]]:
        """Perform simple clustering based on embedding similarity."""
        try:
            if not connections:
                return []

            clusters = []
            used_indices = set()

            for i, embedding1 in enumerate(embeddings):
                if i in used_indices:
                    continue

                # Start a new cluster
                cluster = [connections[i]]
                used_indices.add(i)

                # Find similar connections for this cluster
                for j, embedding2 in enumerate(embeddings):
                    if j in used_indices:
                        continue

                    similarity = self._calculate_cosine_similarity(
                        embedding1, embedding2
                    )
                    if similarity > 0.8:  # High similarity threshold for clustering
                        cluster.append(connections[j])
                        used_indices.add(j)

                clusters.append(cluster)

            return clusters

        except Exception as e:
            logger.error(f"Error performing clustering: {e}")
            return [connections]  # Return all in one cluster as fallback

    def _calculate_cache_size(self) -> float:
        """Calculate approximate cache size in MB."""
        try:
            if not self._embedding_cache:
                return 0.0

            # Estimate size: UUID (16 bytes) + embedding (1536 * 4 bytes for float32)
            bytes_per_entry = 16 + (1536 * 4)
            total_bytes = len(self._embedding_cache) * bytes_per_entry
            mb_size = total_bytes / (1024 * 1024)

            return round(mb_size, 2)

        except Exception as e:
            logger.error(f"Error calculating cache size: {e}")
            return 0.0

    async def clear_embedding_cache(self) -> int:
        """Clear the embedding cache and return number of embeddings cleared."""
        try:
            count = len(self._embedding_cache)
            self._embedding_cache.clear()
            logger.info(f"Cleared {count} connection embeddings from cache")
            return count

        except Exception as e:
            logger.error(f"Error clearing embedding cache: {e}")
            return 0

    async def precompute_embeddings_batch(
        self, connections: list[GeneratedConnection]
    ) -> dict[UUID, bool]:
        """Precompute embeddings for a batch of connections."""
        try:
            results = {}

            for connection in connections:
                try:
                    await self.generate_connection_embedding(connection)
                    results[connection.connection_id] = True
                except Exception as e:
                    logger.error(
                        f"Error generating embedding for {connection.connection_id}: {e}"
                    )
                    results[connection.connection_id] = False

            successful = sum(1 for success in results.values() if success)
            logger.info(
                f"Successfully generated {successful}/{len(connections)} embeddings"
            )

            return results

        except Exception as e:
            logger.error(f"Error in batch embedding generation: {e}")
            return {}

    async def get_most_similar_connection_type(
        self, connection: GeneratedConnection, available_types: list[str]
    ) -> tuple[str, float] | None:
        """Find the most similar connection type based on embeddings."""
        try:
            # Generate embedding for the input connection
            target_embedding = await self.generate_connection_embedding(connection)

            best_match = None
            best_similarity = 0.0

            # Check against connections of each available type
            for conn_type in available_types:
                type_similarities = []

                for conn_id, embedding in self._embedding_cache.items():
                    # In real implementation, we'd check the actual connection type
                    # For now, we'll simulate by assuming some connections match each type
                    similarity = self._calculate_cosine_similarity(
                        target_embedding, embedding
                    )
                    type_similarities.append(similarity)

                if type_similarities:
                    avg_similarity = sum(type_similarities) / len(type_similarities)
                    if avg_similarity > best_similarity:
                        best_similarity = avg_similarity
                        best_match = conn_type

            if best_match and best_similarity > 0.5:
                return (best_match, best_similarity)
            else:
                return None

        except Exception as e:
            logger.error(f"Error finding most similar connection type: {e}")
            return None
