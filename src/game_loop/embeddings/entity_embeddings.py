"""
Entity Embedding Generator for optimized game entity embeddings.

This module provides specialized embedding generation for different game entity types
such as characters, locations, items, and events.
"""

import hashlib
import json
import logging

from .exceptions import EmbeddingGenerationError
from .service import EmbeddingService

logger = logging.getLogger(__name__)


class EntityEmbeddingGenerator:
    """Generate optimized embeddings for different game entity types."""

    def __init__(self, embedding_service: EmbeddingService):
        """
        Initialize the EntityEmbeddingGenerator.

        Args:
            embedding_service: The base embedding service for text embeddings
        """
        self.embedding_service = embedding_service
        self.entity_preprocessors = {
            "character": self._preprocess_character,
            "location": self._preprocess_location,
            "item": self._preprocess_item,
            "event": self._preprocess_event,
            "general": self._preprocess_general,
        }

    async def generate_entity_embedding(
        self, entity: dict, entity_type: str
    ) -> list[float]:
        """
        Generate an embedding for a specific game entity.

        Args:
            entity: The entity data dictionary
            entity_type: Type of entity (character, location, item, event)

        Returns:
            list of embedding values as floats

        Raises:
            EmbeddingGenerationError: If embedding generation fails
        """
        try:
            # Preprocess the entity into optimized text
            processed_text = self.preprocess_entity(entity, entity_type)

            # Generate embedding with the entity type as context
            embedding = await self.embedding_service.generate_embedding(
                processed_text, entity_type=entity_type
            )

            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding for entity: {e}")
            raise EmbeddingGenerationError(
                f"Entity embedding generation " f"failed: {e}"
            ) from e

    async def generate_entity_embeddings_batch(
        self, entities: list[dict], entity_types: list[str] | None = None
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple entities in a batch.

        Args:
            entities: list of entity dictionaries
            entity_types: list of entity types (must match entities length if provided)

        Returns:
            list of embedding vectors

        Raises:
            ValueError: If entity_types is provided but length doesn't match entities
            EmbeddingGenerationError: If batch embedding generation fails
        """
        if entity_types and len(entities) != len(entity_types):
            raise ValueError(
                "If entity_types is provided, it must match the length of entities"
            )

        # Use default entity type if not specified
        if not entity_types:
            entity_types = ["general"] * len(entities)

        try:
            # Preprocess all entities
            processed_texts = [
                self.preprocess_entity(entity, entity_type)
                for entity, entity_type in zip(entities, entity_types, strict=False)
            ]

            # Generate embeddings in batch
            embeddings = await self.embedding_service.generate_embeddings_batch(
                processed_texts, entity_types=entity_types
            )

            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate batch entity embeddings: {e}")
            raise EmbeddingGenerationError(
                f"Batch entity embedding generation failed: {e}"
            ) from e

    def preprocess_entity(self, entity: dict, entity_type: str) -> str:
        """
        Preprocess an entity into optimized text for embedding.

        Args:
            entity: Entity data dictionary
            entity_type: Type of entity (character, location, item, event)

        Returns:
            Processed text optimized for embedding generation
        """
        # Get the appropriate preprocessor function
        preprocessor = self.entity_preprocessors.get(
            entity_type.lower(), self.entity_preprocessors["general"]
        )

        # Extract entity features and apply preprocessing
        features = self.extract_entity_features(entity, entity_type)
        processed_text = preprocessor(features)

        return processed_text

    def extract_entity_features(self, entity: dict, entity_type: str) -> dict:
        """
        Extract relevant features from an entity based on its type.

        Args:
            entity: Entity data dictionary
            entity_type: Type of entity

        Returns:
            Dictionary of extracted features
        """
        # Common features for all entity types
        features = {
            "id": entity.get("id", ""),
            "name": entity.get("name", ""),
            "description": entity.get("description", ""),
        }

        # Entity type specific features
        if entity_type.lower() == "character":
            features.update(
                {
                    "personality": entity.get("personality", ""),
                    "background": entity.get("background", ""),
                    "motivations": entity.get("motivations", ""),
                    "relationships": entity.get("relationships", {}),
                }
            )
        elif entity_type.lower() == "location":
            features.update(
                {
                    "environment": entity.get("environment", ""),
                    "atmosphere": entity.get("atmosphere", ""),
                    "landmarks": entity.get("landmarks", []),
                    "connections": entity.get("connections", []),
                }
            )
        elif entity_type.lower() == "item":
            features.update(
                {
                    "properties": entity.get("properties", {}),
                    "usage": entity.get("usage", ""),
                    "origin": entity.get("origin", ""),
                    "value": entity.get("value", ""),
                }
            )
        elif entity_type.lower() == "event":
            features.update(
                {
                    "participants": entity.get("participants", []),
                    "timeline": entity.get("timeline", ""),
                    "consequences": entity.get("consequences", ""),
                    "significance": entity.get("significance", ""),
                }
            )

        return features

    async def get_entity_by_embedding_similarity(
        self,
        query_embedding: list[float],
        entity_type: str | None = None,
        top_k: int = 5,
    ) -> list[tuple[dict, float]]:
        """
        Find entities similar to the given embedding.

        This is a placeholder method that requires the EntityEmbeddingRegistry
        to be fully functional. Will be implemented when the registry is available.

        Args:
            query_embedding: The embedding to compare against
            entity_type: Optional filter for entity type
            top_k: Number of results to return

        Returns:
            list of (entity, similarity_score) tuples
        """
        # This will be implemented when integrated with the EntityEmbeddingRegistry
        logger.warning("get_entity_by_embedding_similarity not fully implemented yet")
        return []

    def get_entity_cache_key(self, entity: dict) -> str:
        """
        Generate a unique cache key for an entity.

        Args:
            entity: Entity data dictionary

        Returns:
            String hash key for the entity
        """
        # Extract identifying fields
        identifier = {
            "id": entity.get("id", ""),
            "name": entity.get("name", ""),
            "type": entity.get("type", "general"),
        }

        # Add version or updated timestamp if available
        if "version" in entity:
            identifier["version"] = entity["version"]
        if "updated_at" in entity:
            identifier["updated_at"] = entity["updated_at"]

        # Create a stable hash from the identifier
        serialized = json.dumps(identifier, sort_keys=True)
        return f"entity:{hashlib.md5(serialized.encode()).hexdigest()}"

    # Private preprocessing methods for each entity type

    def _preprocess_character(self, features: dict) -> str:
        """Preprocess character entity features for embedding."""
        template = (
            f"Character: {features['name']}\n"
            f"Description: {features['description']}\n"
            f"Personality: {features['personality']}\n"
            f"Background: {features['background']}\n"
            f"Motivations: {features['motivations']}\n"
        )

        # Add relationship information if available
        if features.get("relationships"):
            relationships_text = "\nRelationships:\n"
            for rel_name, rel_desc in features["relationships"].items():
                relationships_text += f"- {rel_name}: {rel_desc}\n"
            template += relationships_text

        return template.strip()

    def _preprocess_location(self, features: dict) -> str:
        """Preprocess location entity features for embedding."""
        template = (
            f"Location: {features['name']}\n"
            f"Description: {features['description']}\n"
            f"Environment: {features['environment']}\n"
            f"Atmosphere: {features['atmosphere']}\n"
        )

        # Add landmark information if available
        if features.get("landmarks"):
            landmarks_text = "\nLandmarks:\n"
            for landmark in features["landmarks"]:
                landmarks_text += f"- {landmark}\n"
            template += landmarks_text

        # Add connection information if available
        if features.get("connections"):
            connections_text = "\nConnections:\n"
            for connection in features["connections"]:
                connections_text += f"- {connection}\n"
            template += connections_text

        return template.strip()

    def _preprocess_item(self, features: dict) -> str:
        """Preprocess item entity features for embedding."""
        template = (
            f"Item: {features['name']}\n"
            f"Description: {features['description']}\n"
            f"Usage: {features['usage']}\n"
            f"Origin: {features['origin']}\n"
            f"Value: {features['value']}\n"
        )

        # Add property information if available
        if features.get("properties"):
            properties_text = "\nProperties:\n"
            for prop_name, prop_value in features["properties"].items():
                properties_text += f"- {prop_name}: {prop_value}\n"
            template += properties_text

        return template.strip()

    def _preprocess_event(self, features: dict) -> str:
        """Preprocess event entity features for embedding."""
        template = (
            f"Event: {features['name']}\n"
            f"Description: {features['description']}\n"
            f"Timeline: {features['timeline']}\n"
            f"Consequences: {features['consequences']}\n"
            f"Significance: {features['significance']}\n"
        )

        # Add participant information if available
        if features.get("participants"):
            participants_text = "\nParticipants:\n"
            for participant in features["participants"]:
                participants_text += f"- {participant}\n"
            template += participants_text

        return template.strip()

    def _preprocess_general(self, features: dict) -> str:
        """Default entity preprocessor for unknown entity types."""
        template = (
            f"Entity: {features['name']}\n"
            f"ID: {features['id']}\n"
            f"Description: {features['description']}\n"
        )

        # Add any other available fields
        for key, value in features.items():
            if key not in ["name", "id", "description"] and isinstance(value, str):
                template += f"{key.capitalize()}: {value}\n"

        return template.strip()

    async def generate_embedding(self, text: str) -> list[float]:
        """
        Generate an embedding for raw text (compatibility method).

        Args:
            text: Text to embed

        Returns:
            list of embedding values as floats
        """
        # Convert text to a general entity for processing
        entity = {"name": text, "description": text}
        return await self.generate_entity_embedding(entity, "general")
