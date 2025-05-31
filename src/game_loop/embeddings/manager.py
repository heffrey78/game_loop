"""
Embedding Manager for Game Loop.

This module implements the EmbeddingManager, which coordinates embedding generation
and database storage operations as specified in the embedding_pipeline.md document.

The EmbeddingManager is responsible for:
1. Creating and updating embeddings for game entities
2. Coordinating between the database and the embedding generators
3. Handling batch processing of embeddings
4. Managing embedding lifecycle (creation, updates, deletion)
"""

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any
from uuid import UUID

from game_loop.database.db_connection import get_connection
from game_loop.embeddings.entity_generator import EntityEmbeddingGenerator
from game_loop.embeddings.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manager for entity embedding generation and storage.

    This component coordinates between the entity embedding generator and
    the database, ensuring that embeddings are properly generated, stored,
    and updated when entity data changes.

    It's a key part of the Embedding System in the architecture diagram,
    serving as the bridge between the vector generation capabilities and
    the database's vector storage.
    """

    def __init__(self, embedding_generator: EntityEmbeddingGenerator):
        """
        Initialize the embedding manager.

        Args:
            embedding_generator: The entity embedding generator to use
        """
        self.embedding_generator = embedding_generator
        logger.info("Initialized embedding manager")

    async def create_or_update_location_embedding(self, location_id: UUID) -> bool:
        """
        Create or update the embedding for a location.

        This method fetches the location data from the database,
        generates a new embedding, and updates the database with
        the new embedding vector.

        Args:
            location_id: UUID of the location to update

        Returns:
            True if successful, False otherwise

        Raises:
            ValueError: If location not found
        """
        try:
            async with get_connection() as conn:
                # Fetch the location data - join with regions to get region_name
                location = await conn.fetchrow(
                    """
                    SELECT
                        l.id,
                        l.name,
                        l.short_desc,
                        l.full_desc,
                        l.location_type,
                        r.name as region_name
                    FROM
                        locations l
                    LEFT JOIN
                        regions r ON l.region_id = r.id
                    WHERE
                        l.id = $1
                """,
                    location_id,
                )

                if not location:
                    raise ValueError(f"Location {location_id} not found")

                # Convert to dict for the generator
                location_dict = dict(location)

                # Generate embedding
                embedding = await self.embedding_generator.generate_location_embedding(
                    location_dict
                )

                # Update the location with the new embedding
                await conn.execute(
                    """
                    UPDATE locations
                    SET location_embedding = $1
                    WHERE id = $2
                """,
                    embedding,
                    location_id,
                )

                logger.info(f"Updated embedding for location {location_id}")
                return True

        except EmbeddingError as e:
            logger.error(
                f"Failed to generate embedding for location {location_id}: {e}"
            )
            return False
        except Exception as e:
            logger.error(f"Error updating location embedding {location_id}: {e}")
            return False

    async def create_or_update_object_embedding(self, object_id: UUID) -> bool:
        """
        Create or update the embedding for an object.

        This method fetches the object data from the database,
        generates a new embedding, and updates the database with
        the new embedding vector.

        Args:
            object_id: UUID of the object to update

        Returns:
            True if successful, False otherwise

        Raises:
            ValueError: If object not found
        """
        try:
            async with get_connection() as conn:
                # Fetch the object data
                object_data = await conn.fetchrow(
                    """
                    SELECT
                        id,
                        name,
                        short_desc,
                        full_desc,
                        object_type,
                        properties_json as properties
                    FROM
                        objects
                    WHERE
                        id = $1
                """,
                    object_id,
                )

                if not object_data:
                    raise ValueError(f"Object {object_id} not found")

                # Convert to dict for the generator
                object_dict = dict(object_data)

                # Generate embedding
                embedding = await self.embedding_generator.generate_object_embedding(
                    object_dict
                )

                # Update the object with the new embedding
                await conn.execute(
                    """
                    UPDATE objects
                    SET object_embedding = $1
                    WHERE id = $2
                """,
                    embedding,
                    object_id,
                )

                logger.info(f"Updated embedding for object {object_id}")
                return True

        except EmbeddingError as e:
            logger.error(f"Failed to generate embedding for object {object_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error updating object embedding {object_id}: {e}")
            return False

    async def create_or_update_npc_embedding(self, npc_id: UUID) -> bool:
        """
        Create or update the embedding for an NPC.

        This method fetches the NPC data from the database,
        generates a new embedding, and updates the database with
        the new embedding vector.

        Args:
            npc_id: UUID of the NPC to update

        Returns:
            True if successful, False otherwise

        Raises:
            ValueError: If NPC not found
        """
        try:
            async with get_connection() as conn:
                # Fetch the NPC data
                npc_data = await conn.fetchrow(
                    """
                    SELECT
                        id,
                        name,
                        short_desc,
                        full_desc,
                        npc_type,
                        personality_json as personality,
                        knowledge_json as knowledge
                    FROM
                        npcs
                    WHERE
                        id = $1
                """,
                    npc_id,
                )

                if not npc_data:
                    raise ValueError(f"NPC {npc_id} not found")

                # Convert to dict for the generator
                npc_dict = dict(npc_data)

                # Generate embedding
                embedding = await self.embedding_generator.generate_npc_embedding(
                    npc_dict
                )

                # Update the NPC with the new embedding
                await conn.execute(
                    """
                    UPDATE npcs
                    SET npc_embedding = $1
                    WHERE id = $2
                """,
                    embedding,
                    npc_id,
                )

                logger.info(f"Updated embedding for NPC {npc_id}")
                return True

        except EmbeddingError as e:
            logger.error(f"Failed to generate embedding for NPC {npc_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error updating NPC embedding {npc_id}: {e}")
            return False

    async def create_or_update_quest_embedding(self, quest_id: UUID) -> bool:
        """
        Create or update the embedding for a quest.

        This method fetches the quest data from the database,
        generates a new embedding, and updates the database with
        the new embedding vector.

        Args:
            quest_id: UUID of the quest to update

        Returns:
            True if successful, False otherwise

        Raises:
            ValueError: If quest not found
        """
        try:
            async with get_connection() as conn:
                # Fetch the quest data
                quest_data = await conn.fetchrow(
                    """
                    SELECT
                        id,
                        title,
                        description,
                        quest_type,
                        steps_json as steps,
                        rewards_json as rewards
                    FROM
                        quests
                    WHERE
                        id = $1
                """,
                    quest_id,
                )

                if not quest_data:
                    raise ValueError(f"Quest {quest_id} not found")

                # Convert to dict for the generator
                quest_dict = dict(quest_data)

                # Generate embedding
                embedding = await self.embedding_generator.generate_quest_embedding(
                    quest_dict
                )

                # Update the quest with the new embedding
                await conn.execute(
                    """
                    UPDATE quests
                    SET quest_embedding = $1
                    WHERE id = $2
                """,
                    embedding,
                    quest_id,
                )

                logger.info(f"Updated embedding for quest {quest_id}")
                return True

        except EmbeddingError as e:
            logger.error(f"Failed to generate embedding for quest {quest_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error updating quest embedding {quest_id}: {e}")
            return False

    async def create_or_update_knowledge_embedding(self, knowledge_id: UUID) -> bool:
        """
        Create or update the embedding for a player knowledge item.

        This method fetches the knowledge data from the database,
        generates a new embedding, and updates the database with
        the new embedding vector.

        Args:
            knowledge_id: UUID of the knowledge item to update

        Returns:
            True if successful, False otherwise

        Raises:
            ValueError: If knowledge item not found
        """
        try:
            async with get_connection() as conn:
                # Fetch the knowledge data
                knowledge_data = await conn.fetchrow(
                    """
                    SELECT
                        id,
                        knowledge_key as key,
                        knowledge_value as value
                    FROM
                        player_knowledge
                    WHERE
                        id = $1
                """,
                    knowledge_id,
                )

                if not knowledge_data:
                    raise ValueError(f"Knowledge item {knowledge_id} not found")

                # Convert to dict for the generator
                knowledge_dict = dict(knowledge_data)

                # Generate embedding
                embedding = await self.embedding_generator.generate_knowledge_embedding(
                    knowledge_dict
                )

                # Update the knowledge item with the new embedding
                await conn.execute(
                    """
                    UPDATE player_knowledge
                    SET knowledge_embedding = $1
                    WHERE id = $2
                """,
                    embedding,
                    knowledge_id,
                )

                logger.info(f"Updated embedding for knowledge item {knowledge_id}")
                return True

        except EmbeddingError as e:
            logger.error(
                f"Failed to generate embedding for knowledge {knowledge_id}: {e}"
            )
            return False
        except Exception as e:
            logger.error(f"Error updating knowledge embedding {knowledge_id}: {e}")
            return False

    async def create_or_update_rule_embedding(self, rule_id: UUID) -> bool:
        """
        Create or update the embedding for a world rule.

        This method fetches the rule data from the database,
        generates a new embedding, and updates the database with
        the new embedding vector.

        Args:
            rule_id: UUID of the rule to update

        Returns:
            True if successful, False otherwise

        Raises:
            ValueError: If rule not found
        """
        try:
            async with get_connection() as conn:
                # Fetch the rule data
                rule_data = await conn.fetchrow(
                    """
                    SELECT
                        id,
                        rule_name as name,
                        rule_description as description,
                        rule_type,
                        rule_logic as logic_summary
                    FROM
                        world_rules
                    WHERE
                        id = $1
                """,
                    rule_id,
                )

                if not rule_data:
                    raise ValueError(f"Rule {rule_id} not found")

                # Convert to dict for the generator
                rule_dict = dict(rule_data)

                # Generate embedding
                embedding = await self.embedding_generator.generate_rule_embedding(
                    rule_dict
                )

                # Update the rule with the new embedding
                await conn.execute(
                    """
                    UPDATE world_rules
                    SET rule_embedding = $1
                    WHERE id = $2
                """,
                    embedding,
                    rule_id,
                )

                logger.info(f"Updated embedding for world rule {rule_id}")
                return True

        except EmbeddingError as e:
            logger.error(f"Failed to generate embedding for rule {rule_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error updating rule embedding {rule_id}: {e}")
            return False

    async def batch_process_embeddings(
        self, entity_type: str, batch_size: int = 50
    ) -> tuple[int, int]:
        """
        Batch process embeddings for an entity type.

        This method fetches entities that have null embeddings
        and generates embeddings for them in batches.

        Args:
            entity_type: Type of entity to process (locations, objects, npcs, etc.)
            batch_size: Number of entities to process in each batch

        Returns:
            Tuple of (processed_count, success_count)

        Raises:
            ValueError: If entity_type is not valid
        """
        valid_entity_types = {
            "locations",
            "objects",
            "npcs",
            "quests",
            "knowledge",
            "rules",
            "player_history",
        }

        if entity_type not in valid_entity_types:
            raise ValueError(
                f"Invalid entity type: {entity_type}. Valid types: {valid_entity_types}"
            )

        processed_count = 0
        success_count = 0

        try:
            async with get_connection() as conn:
                # Map entity types to their respective tables and columns
                entity_mapping: dict[
                    str, dict[str, str | Callable[[UUID], Coroutine[Any, Any, bool]]]
                ] = {
                    "locations": {
                        "table": "locations",
                        "id_col": "id",
                        "embed_col": "location_embedding",
                        "func": self.create_or_update_location_embedding,
                    },
                    "objects": {
                        "table": "objects",
                        "id_col": "id",
                        "embed_col": "object_embedding",
                        "func": self.create_or_update_object_embedding,
                    },
                    "npcs": {
                        "table": "npcs",
                        "id_col": "id",
                        "embed_col": "npc_embedding",
                        "func": self.create_or_update_npc_embedding,
                    },
                    "quests": {
                        "table": "quests",
                        "id_col": "id",
                        "embed_col": "quest_embedding",
                        "func": self.create_or_update_quest_embedding,
                    },
                    "knowledge": {
                        "table": "player_knowledge",
                        "id_col": "id",
                        "embed_col": "knowledge_embedding",
                        "func": self.create_or_update_knowledge_embedding,
                    },
                    "rules": {
                        "table": "world_rules",
                        "id_col": "id",
                        "embed_col": "rule_embedding",
                        "func": self.create_or_update_rule_embedding,
                    },
                }

                mapping = entity_mapping[entity_type]

                # Get entities with null embeddings
                query = f"""
                    SELECT {mapping['id_col']}
                    FROM {mapping['table']}
                    WHERE {mapping['embed_col']} IS NULL
                    LIMIT $1
                """

                entity_ids = await conn.fetch(query, batch_size)

                for entity_record in entity_ids:
                    entity_id = entity_record[mapping["id_col"]]
                    processed_count += 1

                    # Get the function from mapping and call it properly
                    update_func = mapping["func"]
                    # Tell mypy that update_func is callable
                    if callable(update_func):
                        success = await update_func(entity_id)
                        if success:
                            success_count += 1

                    # Small delay to prevent overloading the embedding service
                    await asyncio.sleep(0.1)

                logger.info(
                    f"Batch processed {processed_count} {entity_type}, "
                    f"{success_count} successful"
                )
                return (processed_count, success_count)

        except Exception as e:
            logger.error(f"Error during batch processing of {entity_type}: {e}")
            return (processed_count, success_count)

    async def find_similar_entities(
        self,
        entity_type: str,
        query_text: str,
        limit: int = 5,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Find entities similar to the query text using vector similarity.

        This method generates an embedding for the query text and then
        searches for similar entities using cosine similarity.

        Args:
            entity_type: Type of entity to search (locations, objects, npcs, etc.)
            query_text: Text to search for
            limit: Maximum number of results to return
            threshold: Optional similarity threshold (0.0 to 1.0)

        Returns:
            List of similar entities with their similarity scores

        Raises:
            ValueError: If entity_type is not valid
            EmbeddingError: If embedding generation fails
        """
        valid_entity_types = {
            "locations",
            "objects",
            "npcs",
            "quests",
            "knowledge",
            "rules",
            "player_history",
        }

        if entity_type not in valid_entity_types:
            raise ValueError(
                f"Invalid entity type: {entity_type}. Valid types: {valid_entity_types}"
            )

        # Map entity types to their respective tables and columns
        entity_mapping: dict[str, dict[str, str | list[str]]] = {
            "locations": {
                "table": "locations",
                "embed_col": "location_embedding",
                "id_col": "id",
                "name_col": "name",
                "desc_col": "short_desc",
                "other_cols": ["full_desc", "location_type"],
            },
            "objects": {
                "table": "objects",
                "embed_col": "object_embedding",
                "id_col": "id",
                "name_col": "name",
                "desc_col": "short_desc",
                "other_cols": ["full_desc", "object_type"],
            },
            "npcs": {
                "table": "npcs",
                "embed_col": "npc_embedding",
                "id_col": "id",
                "name_col": "name",
                "desc_col": "short_desc",
                "other_cols": ["full_desc", "npc_type"],
            },
            "quests": {
                "table": "quests",
                "embed_col": "quest_embedding",
                "id_col": "id",
                "name_col": "title",
                "desc_col": "description",
                "other_cols": ["quest_type"],
            },
            "knowledge": {
                "table": "player_knowledge",
                "embed_col": "knowledge_embedding",
                "id_col": "id",
                "name_col": "knowledge_key",
                "desc_col": "knowledge_value",
                "other_cols": ["player_id"],
            },
            "rules": {
                "table": "world_rules",
                "embed_col": "rule_embedding",
                "id_col": "id",
                "name_col": "rule_name",
                "desc_col": "rule_description",
                "other_cols": ["rule_type"],
            },
        }

        mapping = entity_mapping[entity_type]

        try:
            # Generate embedding for the query text
            embedding = (
                await self.embedding_generator.embedding_service.generate_embedding(
                    query_text
                )
            )

            # Build the query columns - ensure we're working with a list of strings only
            select_cols: list[str] = [
                str(mapping["id_col"]),
                str(mapping["name_col"]),
                str(mapping["desc_col"]),
            ]

            # Add each item from other_cols individually to ensure proper typing
            other_cols = mapping["other_cols"]
            if isinstance(other_cols, list):
                for col in other_cols:
                    if isinstance(col, str):  # Ensure we only add strings
                        select_cols.append(col)

            # Now join the list of strings
            select_cols_str = ", ".join(select_cols)

            # Add similarity calculation
            similarity_col = f"1 - ({mapping['embed_col']} <=> $1) AS similarity"

            threshold_clause = ""
            if threshold is not None:
                threshold_clause = (
                    f"HAVING 1 - ({mapping['embed_col']} <=> $1) >= "
                    f"${len(select_cols) + 2}"
                )

            # Build the final query
            query = f"""
                SELECT {select_cols_str}, {similarity_col}
                FROM {mapping['table']}
                WHERE {mapping['embed_col']} IS NOT NULL
                {threshold_clause}
                ORDER BY similarity DESC
                LIMIT ${len(select_cols) + 1}
            """

            params = [embedding, limit]
            if threshold is not None:
                params.append(threshold)

            # Execute the query
            async with get_connection() as conn:
                results = await conn.fetch(query, *params)

            # Convert to list of dicts
            entities = []
            for row in results:
                entity = dict(row)
                entities.append(entity)

            return entities

        except EmbeddingError as e:
            logger.error(f"Error generating embedding for similarity search: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            raise
