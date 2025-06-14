"""
Object Embedding Manager for specialized object embedding generation and management.

This module provides object-specific embedding functionality, including
context-aware embedding generation and object similarity calculations.
"""

from __future__ import annotations

import logging
from typing import Any

from game_loop.core.models.object_models import (
    GeneratedObject,
    ObjectInteractions,
    ObjectProperties,
)
from game_loop.embeddings.embedding_manager import EmbeddingManager
from game_loop.llm.client import OllamaClient

logger = logging.getLogger(__name__)


class ObjectEmbeddingManager:
    """Specialized embedding manager for objects with enhanced context awareness."""

    def __init__(
        self, base_embedding_manager: EmbeddingManager, llm_client: OllamaClient
    ):
        self.base_embedding_manager = base_embedding_manager
        self.llm_client = llm_client
        self._embedding_cache: dict[str, list[float]] = {}

    async def generate_object_embedding(
        self, generated_object: GeneratedObject
    ) -> list[float]:
        """Generate context-aware embedding for a complete object."""
        try:
            # Create comprehensive text representation
            embedding_text = self._create_comprehensive_object_text(generated_object)

            # Check cache first
            cache_key = self._create_cache_key(embedding_text)
            if cache_key in self._embedding_cache:
                logger.debug(
                    f"Using cached embedding for {generated_object.properties.name}"
                )
                return self._embedding_cache[cache_key]

            # Generate embedding
            embedding = await self.base_embedding_manager.generate_embedding(
                embedding_text
            )

            # Cache the result
            self._embedding_cache[cache_key] = embedding

            logger.debug(f"Generated embedding for {generated_object.properties.name}")
            return embedding

        except Exception as e:
            logger.error(f"Error generating object embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 1536  # Standard embedding dimension

    async def generate_property_embedding(
        self, properties: ObjectProperties
    ) -> list[float]:
        """Generate embedding focused on object properties."""
        try:
            property_text = self._create_property_text(properties)

            cache_key = self._create_cache_key(property_text)
            if cache_key in self._embedding_cache:
                return self._embedding_cache[cache_key]

            embedding = await self.base_embedding_manager.generate_embedding(
                property_text
            )
            self._embedding_cache[cache_key] = embedding

            return embedding

        except Exception as e:
            logger.error(f"Error generating property embedding: {e}")
            return [0.0] * 1536

    async def generate_interaction_embedding(
        self, interactions: ObjectInteractions
    ) -> list[float]:
        """Generate embedding focused on object interactions."""
        try:
            interaction_text = self._create_interaction_text(interactions)

            cache_key = self._create_cache_key(interaction_text)
            if cache_key in self._embedding_cache:
                return self._embedding_cache[cache_key]

            embedding = await self.base_embedding_manager.generate_embedding(
                interaction_text
            )
            self._embedding_cache[cache_key] = embedding

            return embedding

        except Exception as e:
            logger.error(f"Error generating interaction embedding: {e}")
            return [0.0] * 1536

    async def generate_search_embedding(
        self, search_query: str, context: dict[str, Any] | None = None
    ) -> list[float]:
        """Generate embedding for object search queries with optional context."""
        try:
            # Enhance query with context if provided
            enhanced_query = self._enhance_search_query(search_query, context)

            cache_key = self._create_cache_key(enhanced_query)
            if cache_key in self._embedding_cache:
                return self._embedding_cache[cache_key]

            embedding = await self.base_embedding_manager.generate_embedding(
                enhanced_query
            )
            self._embedding_cache[cache_key] = embedding

            logger.debug(f"Generated search embedding for query: {search_query}")
            return embedding

        except Exception as e:
            logger.error(f"Error generating search embedding: {e}")
            return [0.0] * 1536

    def calculate_object_similarity(
        self, embedding1: list[float], embedding2: list[float]
    ) -> float:
        """Calculate similarity between two object embeddings."""
        try:
            return self.base_embedding_manager.calculate_similarity(
                embedding1, embedding2
            )
        except Exception as e:
            logger.error(f"Error calculating object similarity: {e}")
            return 0.0

    async def find_similar_objects(
        self,
        target_object: GeneratedObject,
        candidate_objects: list[GeneratedObject],
        threshold: float = 0.7,
    ) -> list[tuple[GeneratedObject, float]]:
        """Find objects similar to the target object."""
        try:
            target_embedding = await self.generate_object_embedding(target_object)
            similar_objects = []

            for candidate in candidate_objects:
                candidate_embedding = await self.generate_object_embedding(candidate)
                similarity = self.calculate_object_similarity(
                    target_embedding, candidate_embedding
                )

                if similarity >= threshold:
                    similar_objects.append((candidate, similarity))

            # Sort by similarity (highest first)
            similar_objects.sort(key=lambda x: x[1], reverse=True)

            logger.info(
                f"Found {len(similar_objects)} similar objects to {target_object.properties.name}"
            )
            return similar_objects

        except Exception as e:
            logger.error(f"Error finding similar objects: {e}")
            return []

    async def cluster_objects_by_similarity(
        self, objects: list[GeneratedObject], similarity_threshold: float = 0.8
    ) -> list[list[GeneratedObject]]:
        """Group objects into clusters based on similarity."""
        try:
            if not objects:
                return []

            # Generate embeddings for all objects
            embeddings = []
            for obj in objects:
                embedding = await self.generate_object_embedding(obj)
                embeddings.append(embedding)

            # Simple clustering algorithm
            clusters = []
            used_indices = set()

            for i, obj in enumerate(objects):
                if i in used_indices:
                    continue

                # Start new cluster
                cluster = [obj]
                used_indices.add(i)

                # Find similar objects for this cluster
                for j, other_obj in enumerate(objects):
                    if j in used_indices:
                        continue

                    similarity = self.calculate_object_similarity(
                        embeddings[i], embeddings[j]
                    )
                    if similarity >= similarity_threshold:
                        cluster.append(other_obj)
                        used_indices.add(j)

                clusters.append(cluster)

            logger.info(f"Clustered {len(objects)} objects into {len(clusters)} groups")
            return clusters

        except Exception as e:
            logger.error(f"Error clustering objects: {e}")
            return [
                [obj] for obj in objects
            ]  # Fallback: each object in its own cluster

    def _create_comprehensive_object_text(
        self, generated_object: GeneratedObject
    ) -> str:
        """Create comprehensive text representation of an object for embedding."""
        parts = []

        # Basic properties
        props = generated_object.properties
        parts.append(f"Object: {props.name}")
        parts.append(f"Type: {props.object_type}")
        parts.append(f"Material: {props.material}")
        parts.append(f"Size: {props.size}, Weight: {props.weight}")
        parts.append(f"Durability: {props.durability}")

        if props.description:
            parts.append(f"Description: {props.description}")

        if props.special_properties:
            parts.append(f"Properties: {', '.join(props.special_properties)}")

        parts.append(f"Cultural significance: {props.cultural_significance}")

        if props.value > 0:
            parts.append(f"Value: {props.value}")

        # Interactions
        interactions = generated_object.interactions
        if interactions.available_actions:
            parts.append(f"Actions: {', '.join(interactions.available_actions)}")

        if interactions.examination_text:
            parts.append(f"Examination: {interactions.examination_text}")

        parts.append(f"Portable: {'yes' if interactions.portable else 'no'}")
        parts.append(f"Consumable: {'yes' if interactions.consumable else 'no'}")

        # Interaction results
        if interactions.interaction_results:
            for action, result in interactions.interaction_results.items():
                parts.append(f"{action.capitalize()}: {result}")

        # Generation metadata
        if generated_object.generation_metadata:
            metadata = generated_object.generation_metadata
            if "location_theme" in metadata:
                parts.append(f"Theme: {metadata['location_theme']}")
            if "generation_purpose" in metadata:
                parts.append(f"Purpose: {metadata['generation_purpose']}")

        return " ".join(parts)

    def _create_property_text(self, properties: ObjectProperties) -> str:
        """Create text representation focused on properties."""
        parts = [
            properties.name,
            properties.object_type,
            properties.material,
            f"{properties.size} {properties.weight}",
            properties.durability,
            properties.cultural_significance,
        ]

        if properties.description:
            parts.append(properties.description)

        if properties.special_properties:
            parts.extend(properties.special_properties)

        return " ".join(filter(None, parts))

    def _create_interaction_text(self, interactions: ObjectInteractions) -> str:
        """Create text representation focused on interactions."""
        parts = []

        if interactions.available_actions:
            parts.extend(interactions.available_actions)

        if interactions.examination_text:
            parts.append(interactions.examination_text)

        # Add interaction context
        parts.append("portable" if interactions.portable else "stationary")
        parts.append("consumable" if interactions.consumable else "permanent")

        # Add interaction results
        if interactions.interaction_results:
            parts.extend(interactions.interaction_results.values())

        return " ".join(filter(None, parts))

    def _enhance_search_query(self, query: str, context: dict[str, Any] | None) -> str:
        """Enhance search query with contextual information."""
        enhanced_parts = [query]

        if context:
            # Add location context
            if "location_theme" in context:
                enhanced_parts.append(f"in {context['location_theme']} setting")

            # Add purpose context
            if "search_purpose" in context:
                enhanced_parts.append(f"for {context['search_purpose']}")

            # Add player level context
            if "player_level" in context:
                level = context["player_level"]
                if level <= 3:
                    enhanced_parts.append("simple basic")
                elif level >= 7:
                    enhanced_parts.append("advanced powerful")

            # Add object type filters
            if "preferred_types" in context:
                enhanced_parts.extend(context["preferred_types"])

        return " ".join(enhanced_parts)

    def _create_cache_key(self, text: str) -> str:
        """Create cache key from text."""
        # Simple hash-based key
        return str(hash(text.lower().strip()))

    def clear_cache(self):
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        logger.info("Cleared object embedding cache")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get statistics about the embedding cache."""
        return {
            "cache_size": len(self._embedding_cache),
            "cache_keys": list(self._embedding_cache.keys())[
                :10
            ],  # First 10 keys for inspection
        }
