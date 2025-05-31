"""
Entity Embedding Generator for Game Loop.

This module provides specialized embedding generation for different game entity types
as specified in the embedding_pipeline.md document. It builds on the core embedding
service to provide context-aware embedding generation for game entities.

The entity embedding generator is part of the Embedding System in the architecture
diagram and handles the entity-specific text preprocessing and embedding generation.
"""

import logging
from typing import Any

from game_loop.embeddings.exceptions import EmbeddingError
from game_loop.embeddings.service import EmbeddingService

logger = logging.getLogger(__name__)


class EntityEmbeddingGenerator:
    """
    Handles the generation of embeddings for specific entity types.

    This component preprocesses entity data to create meaningful text
    that captures the entity's semantic content before generating embeddings.
    It's a key part of the Embedding System in the architecture diagram.
    """

    def __init__(self, embedding_service: EmbeddingService):
        """
        Initialize the entity embedding generator.

        Args:
            embedding_service: The embedding service to use for vector generation
        """
        self.embedding_service = embedding_service
        logger.info("Initialized entity embedding generator")

    async def generate_location_embedding(
        self, location: dict[str, Any]
    ) -> list[float]:
        """
        Generate embedding for a location entity.

        Combines relevant location fields to create a rich text representation
        that captures the location's semantic meaning.

        Args:
            location: Dictionary containing location data

        Returns:
            Vector embedding for the location

        Raises:
            EmbeddingError: If embedding generation fails
            ValueError: If location data is invalid
        """
        if not location.get("name") or not location.get("short_desc"):
            raise ValueError("Location must have name and short_desc")

        # Combine relevant fields for richer context as described in
        # embedding_pipeline.md
        text_parts = [location["name"], location["short_desc"]]

        if location.get("full_desc"):
            text_parts.append(location["full_desc"])

        if location.get("region_name"):
            text_parts.append(
                f"This location is in the {location['region_name']} region."
            )

        if location.get("location_type"):
            text_parts.append(f"This is a {location['location_type']} location.")

        # Join all parts with proper spacing
        text = ". ".join(text_parts)

        try:
            return await self.embedding_service.generate_embedding(text)
        except EmbeddingError as e:
            logger.error(
                f"Failed to generate embedding for location '{location['name']}': {e}"
            )
            raise

    async def generate_object_embedding(
        self, object_data: dict[str, Any]
    ) -> list[float]:
        """
        Generate embedding for an object entity.

        Combines relevant object fields to create a rich text representation
        that captures the object's semantic meaning.

        Args:
            object_data: Dictionary containing object data

        Returns:
            Vector embedding for the object

        Raises:
            EmbeddingError: If embedding generation fails
            ValueError: If object data is invalid
        """
        if not object_data.get("name") or not object_data.get("short_desc"):
            raise ValueError("Object must have name and short_desc")

        # Combine relevant fields for richer context
        text_parts = [object_data["name"], object_data["short_desc"]]

        if object_data.get("full_desc"):
            text_parts.append(object_data["full_desc"])

        if object_data.get("object_type"):
            text_parts.append(f"This is a {object_data['object_type']}.")

        # Include properties if available
        if object_data.get("properties") and isinstance(
            object_data["properties"], dict
        ):
            properties = object_data["properties"]
            for key, value in properties.items():
                if isinstance(value, str | int | float | bool):
                    text_parts.append(f"It has {key}: {value}.")

        # Join all parts with proper spacing
        text = ". ".join(text_parts)

        try:
            return await self.embedding_service.generate_embedding(text)
        except EmbeddingError as e:
            logger.error(
                f"Failed to generate embedding for object '{object_data['name']}': {e}"
            )
            raise

    async def generate_npc_embedding(self, npc: dict[str, Any]) -> list[float]:
        """
        Generate embedding for an NPC entity.

        Combines relevant NPC fields to create a rich text representation
        that captures the NPC's semantic meaning.

        Args:
            npc: Dictionary containing NPC data

        Returns:
            Vector embedding for the NPC

        Raises:
            EmbeddingError: If embedding generation fails
            ValueError: If NPC data is invalid
        """
        if not npc.get("name") or not npc.get("short_desc"):
            raise ValueError("NPC must have name and short_desc")

        # Combine relevant fields for richer context
        text_parts = [npc["name"], npc["short_desc"]]

        if npc.get("full_desc"):
            text_parts.append(npc["full_desc"])

        if npc.get("npc_type"):
            text_parts.append(f"This is a {npc['npc_type']}.")

        # Include personality traits if available
        if npc.get("personality") and isinstance(npc["personality"], dict):
            personality = npc["personality"]
            traits = []
            for trait, value in personality.items():
                if isinstance(value, str | int | float):
                    traits.append(f"{trait}: {value}")

            if traits:
                text_parts.append(f"Personality traits: {', '.join(traits)}.")

        # Include knowledge if available (limited to avoid too much text)
        if npc.get("knowledge") and isinstance(npc["knowledge"], dict):
            knowledge = list(npc["knowledge"].keys())[:5]  # Limit to 5 knowledge items
            if knowledge:
                text_parts.append(f"Knows about: {', '.join(knowledge)}.")

        # Join all parts with proper spacing
        text = ". ".join(text_parts)

        try:
            return await self.embedding_service.generate_embedding(text)
        except EmbeddingError as e:
            logger.error(f"Failed to generate embedding for NPC '{npc['name']}': {e}")
            raise

    async def generate_quest_embedding(self, quest: dict[str, Any]) -> list[float]:
        """
        Generate embedding for a quest entity.

        Combines relevant quest fields to create a rich text representation
        that captures the quest's semantic meaning.

        Args:
            quest: Dictionary containing quest data

        Returns:
            Vector embedding for the quest

        Raises:
            EmbeddingError: If embedding generation fails
            ValueError: If quest data is invalid
        """
        if not quest.get("title") or not quest.get("description"):
            raise ValueError("Quest must have title and description")

        # Combine relevant fields for richer context
        text_parts = [quest["title"], quest["description"]]

        if quest.get("quest_type"):
            text_parts.append(f"This is a {quest['quest_type']} quest.")

        # Include steps overview (limited to avoid too much text)
        if quest.get("steps") and isinstance(quest["steps"], list):
            steps = quest["steps"]
            if len(steps) > 0:
                step_descs = []
                for _, step in enumerate(steps[:3]):  # Limit to first 3 steps
                    if isinstance(step, dict) and step.get("description"):
                        step_descs.append(step["description"])
                    elif isinstance(step, str):
                        step_descs.append(step)

                if step_descs:
                    text_parts.append(f"Quest steps include: {'. '.join(step_descs)}")

        # Include rewards if available
        if quest.get("rewards") and isinstance(quest["rewards"], dict):
            rewards = []
            for reward_type, value in quest["rewards"].items():
                rewards.append(f"{reward_type}: {value}")

            if rewards:
                text_parts.append(f"Rewards: {', '.join(rewards)}")

        # Join all parts with proper spacing
        text = ". ".join(text_parts)

        try:
            return await self.embedding_service.generate_embedding(text)
        except EmbeddingError as e:
            logger.error(
                f"Failed to generate embedding for quest '{quest['title']}': {e}"
            )
            raise

    async def generate_knowledge_embedding(
        self, knowledge: dict[str, Any]
    ) -> list[float]:
        """
        Generate embedding for a knowledge item.

        Creates a text representation that captures the knowledge content
        and its key for embedding generation.

        Args:
            knowledge: Dictionary containing knowledge data

        Returns:
            Vector embedding for the knowledge

        Raises:
            EmbeddingError: If embedding generation fails
            ValueError: If knowledge data is invalid
        """
        if not knowledge.get("key") or not knowledge.get("value"):
            raise ValueError("Knowledge must have key and value")

        # Combine key and value for context
        text = f"{knowledge['key']}: {knowledge['value']}"

        try:
            return await self.embedding_service.generate_embedding(text)
        except EmbeddingError as e:
            logger.error(
                f"Failed to generate embedding for knowledge '{knowledge['key']}': {e}"
            )
            raise

    async def generate_rule_embedding(self, rule: dict[str, Any]) -> list[float]:
        """
        Generate embedding for a game rule.

        Creates a text representation that captures the rule's purpose and logic
        for embedding generation.

        Args:
            rule: Dictionary containing rule data

        Returns:
            Vector embedding for the rule

        Raises:
            EmbeddingError: If embedding generation fails
            ValueError: If rule data is invalid
        """
        if not rule.get("name") or not rule.get("description"):
            raise ValueError("Rule must have name and description")

        # Combine relevant fields for context
        text_parts = [rule["name"], rule["description"]]

        if rule.get("rule_type"):
            text_parts.append(f"This is a {rule['rule_type']} rule.")

        # Include simplified logic if available
        if rule.get("logic_summary"):
            text_parts.append(rule["logic_summary"])

        # Join all parts with proper spacing
        text = ". ".join(text_parts)

        try:
            return await self.embedding_service.generate_embedding(text)
        except EmbeddingError as e:
            logger.error(f"Failed to generate embedding for rule '{rule['name']}': {e}")
            raise
