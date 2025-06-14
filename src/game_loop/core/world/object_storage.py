"""
Object Storage System for persistence, embedding generation, and semantic search.

This module handles object database operations, embedding generation, and
semantic search capabilities for the object generation system.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from game_loop.core.models.object_models import (
    GeneratedObject,
    ObjectSearchCriteria,
    ObjectStorageResult,
)
from game_loop.database.session_factory import DatabaseSessionFactory
from game_loop.embeddings.embedding_manager import EmbeddingManager
from game_loop.state.models import WorldState

logger = logging.getLogger(__name__)


class ObjectStorage:
    """Handles object persistence, embedding generation, and semantic search."""

    def __init__(
        self,
        world_state: WorldState,
        session_factory: DatabaseSessionFactory,
        embedding_manager: EmbeddingManager,
    ):
        self.world_state = world_state
        self.session_factory = session_factory
        self.embedding_manager = embedding_manager

    async def store_object(
        self, generated_object: GeneratedObject
    ) -> ObjectStorageResult:
        """Store object with embedding generation and database persistence."""
        start_time = datetime.now()

        try:
            async with self.session_factory.get_session() as session:
                # Generate embedding for the object
                embedding_text = self._create_embedding_text(generated_object)
                embedding_vector = await self.embedding_manager.generate_embedding(
                    embedding_text
                )
                generated_object.embedding_vector = embedding_vector

                # Store in world_objects table
                object_id = await self._store_base_object(session, generated_object)

                # Store properties and interactions
                await self._store_object_properties(
                    session, object_id, generated_object
                )

                # Store generation history
                await self._store_generation_history(
                    session, object_id, generated_object
                )

                # Store placement if available
                placement_stored = False
                if generated_object.placement_info:
                    await self._store_object_placement(
                        session, generated_object.placement_info
                    )
                    placement_stored = True

                await session.commit()

                # Update world state
                self.world_state.locations[
                    generated_object.generation_metadata.get(
                        "location_id", UUID("00000000-0000-0000-0000-000000000000")
                    )
                ].objects[object_id] = generated_object.base_object

                duration = (datetime.now() - start_time).total_seconds() * 1000

                logger.info(
                    f"Stored object '{generated_object.properties.name}' with ID {object_id}"
                )

                return ObjectStorageResult(
                    success=True,
                    object_id=object_id,
                    storage_errors=[],
                    embedding_generated=True,
                    placement_stored=placement_stored,
                    storage_duration_ms=duration,
                )

        except Exception as e:
            logger.error(f"Error storing object: {e}")
            duration = (datetime.now() - start_time).total_seconds() * 1000

            return ObjectStorageResult(
                success=False,
                object_id=None,
                storage_errors=[str(e)],
                embedding_generated=False,
                placement_stored=False,
                storage_duration_ms=duration,
            )

    async def search_objects(
        self, criteria: ObjectSearchCriteria
    ) -> list[GeneratedObject]:
        """Search objects using semantic embeddings and filters."""
        try:
            async with self.session_factory.get_session() as session:
                # Build query based on criteria
                query_conditions = []

                # Text-based semantic search
                if criteria.query_text:
                    query_embedding = await self.embedding_manager.generate_embedding(
                        criteria.query_text
                    )

                    # Vector similarity search using pgvector
                    similarity_condition = text(
                        "embedding <-> :query_embedding < :threshold"
                    ).params(
                        query_embedding=query_embedding,
                        threshold=1.0 - criteria.similarity_threshold,
                    )
                    query_conditions.append(similarity_condition)

                # Build full query
                base_query = text(
                    """
                    SELECT wo.object_id, wo.name, wo.description, wo.embedding,
                           op.properties, op.interactions,
                           ogh.generation_context, ogh.generation_metadata
                    FROM world_objects wo
                    LEFT JOIN object_properties op ON wo.object_id = op.object_id
                    LEFT JOIN object_generation_history ogh ON wo.object_id = ogh.object_id
                    WHERE 1=1
                """
                )

                # Add filters
                filter_params = {}
                filter_conditions = []

                if criteria.object_types:
                    filter_conditions.append(
                        "(op.properties->>'object_type') = ANY(:object_types)"
                    )
                    filter_params["object_types"] = criteria.object_types

                # Execute query
                if query_conditions or filter_conditions:
                    full_query = base_query
                    if filter_conditions:
                        full_query = text(
                            str(base_query) + " AND " + " AND ".join(filter_conditions)
                        )

                    if criteria.query_text:
                        # Add ordering by similarity
                        full_query = text(
                            str(full_query)
                            + " ORDER BY embedding <-> :query_embedding LIMIT :max_results"
                        )
                        filter_params["query_embedding"] = query_embedding
                        filter_params["max_results"] = criteria.max_results
                    else:
                        full_query = text(str(full_query) + " LIMIT :max_results")
                        filter_params["max_results"] = criteria.max_results

                    result = await session.execute(full_query, filter_params)
                else:
                    # No filters, get recent objects
                    full_query = text(
                        str(base_query)
                        + " ORDER BY wo.object_id DESC LIMIT :max_results"
                    )
                    result = await session.execute(
                        full_query, {"max_results": criteria.max_results}
                    )

                # Convert results to GeneratedObject instances
                objects = []
                for row in result.fetchall():
                    try:
                        generated_object = self._row_to_generated_object(row)
                        objects.append(generated_object)
                    except Exception as e:
                        logger.warning(f"Error converting row to object: {e}")
                        continue

                logger.info(f"Found {len(objects)} objects matching search criteria")
                return objects

        except Exception as e:
            logger.error(f"Error searching objects: {e}")
            return []

    async def update_object_state(
        self, object_id: UUID, state_changes: dict[str, Any]
    ) -> bool:
        """Update object state and properties."""
        try:
            async with self.session_factory.get_session() as session:
                # Update object properties
                update_query = text(
                    """
                    UPDATE object_properties 
                    SET properties = properties || :state_changes,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE object_id = :object_id
                """
                )

                result = await session.execute(
                    update_query,
                    {
                        "object_id": object_id,
                        "state_changes": json.dumps(state_changes),
                    },
                )

                await session.commit()

                if result.rowcount > 0:
                    logger.info(f"Updated object state for {object_id}")
                    return True
                else:
                    logger.warning(f"No object found with ID {object_id}")
                    return False

        except Exception as e:
            logger.error(f"Error updating object state: {e}")
            return False

    async def get_object_by_id(self, object_id: UUID) -> GeneratedObject | None:
        """Retrieve complete object data by ID."""
        try:
            async with self.session_factory.get_session() as session:
                query = text(
                    """
                    SELECT wo.object_id, wo.name, wo.description, wo.embedding,
                           op.properties, op.interactions,
                           ogh.generation_context, ogh.generation_metadata,
                           opl.placement_data
                    FROM world_objects wo
                    LEFT JOIN object_properties op ON wo.object_id = op.object_id
                    LEFT JOIN object_generation_history ogh ON wo.object_id = ogh.object_id
                    LEFT JOIN object_placements opl ON wo.object_id = opl.object_id
                    WHERE wo.object_id = :object_id
                """
                )

                result = await session.execute(query, {"object_id": object_id})
                row = result.fetchone()

                if row:
                    return self._row_to_generated_object(row)
                else:
                    logger.warning(f"Object {object_id} not found")
                    return None

        except Exception as e:
            logger.error(f"Error retrieving object {object_id}: {e}")
            return None

    async def generate_object_embeddings(self, objects: list[GeneratedObject]) -> None:
        """Generate embeddings for batch of objects."""
        try:
            async with self.session_factory.get_session() as session:
                for obj in objects:
                    try:
                        # Generate embedding
                        embedding_text = self._create_embedding_text(obj)
                        embedding_vector = (
                            await self.embedding_manager.generate_embedding(
                                embedding_text
                            )
                        )

                        # Update in database
                        update_query = text(
                            """
                            UPDATE world_objects 
                            SET embedding = :embedding 
                            WHERE object_id = :object_id
                        """
                        )

                        await session.execute(
                            update_query,
                            {
                                "object_id": obj.base_object.object_id,
                                "embedding": embedding_vector,
                            },
                        )

                        obj.embedding_vector = embedding_vector

                    except Exception as e:
                        logger.warning(
                            f"Failed to generate embedding for object {obj.base_object.object_id}: {e}"
                        )
                        continue

                await session.commit()
                logger.info(f"Generated embeddings for {len(objects)} objects")

        except Exception as e:
            logger.error(f"Error generating object embeddings: {e}")

    async def _store_base_object(
        self, session: AsyncSession, generated_object: GeneratedObject
    ) -> UUID:
        """Store base object in world_objects table."""
        object_id = generated_object.base_object.object_id

        insert_query = text(
            """
            INSERT INTO world_objects (object_id, name, description, embedding)
            VALUES (:object_id, :name, :description, :embedding)
            ON CONFLICT (object_id) DO UPDATE SET
                name = EXCLUDED.name,
                description = EXCLUDED.description,
                embedding = EXCLUDED.embedding
        """
        )

        await session.execute(
            insert_query,
            {
                "object_id": object_id,
                "name": generated_object.properties.name,
                "description": generated_object.properties.description,
                "embedding": generated_object.embedding_vector,
            },
        )

        return object_id

    async def _store_object_properties(
        self, session: AsyncSession, object_id: UUID, generated_object: GeneratedObject
    ):
        """Store object properties and interactions."""
        properties_data = {
            "name": generated_object.properties.name,
            "object_type": generated_object.properties.object_type,
            "material": generated_object.properties.material,
            "size": generated_object.properties.size,
            "weight": generated_object.properties.weight,
            "durability": generated_object.properties.durability,
            "value": generated_object.properties.value,
            "special_properties": generated_object.properties.special_properties,
            "cultural_significance": generated_object.properties.cultural_significance,
            "description": generated_object.properties.description,
        }

        interactions_data = {
            "available_actions": generated_object.interactions.available_actions,
            "use_requirements": generated_object.interactions.use_requirements,
            "interaction_results": generated_object.interactions.interaction_results,
            "state_changes": generated_object.interactions.state_changes,
            "consumable": generated_object.interactions.consumable,
            "portable": generated_object.interactions.portable,
            "examination_text": generated_object.interactions.examination_text,
            "hidden_properties": generated_object.interactions.hidden_properties,
        }

        insert_query = text(
            """
            INSERT INTO object_properties (object_id, properties, interactions)
            VALUES (:object_id, :properties, :interactions)
            ON CONFLICT (object_id) DO UPDATE SET
                properties = EXCLUDED.properties,
                interactions = EXCLUDED.interactions,
                updated_at = CURRENT_TIMESTAMP
        """
        )

        await session.execute(
            insert_query,
            {
                "object_id": object_id,
                "properties": json.dumps(properties_data),
                "interactions": json.dumps(interactions_data),
            },
        )

    async def _store_generation_history(
        self, session: AsyncSession, object_id: UUID, generated_object: GeneratedObject
    ):
        """Store object generation history."""
        insert_query = text(
            """
            INSERT INTO object_generation_history (object_id, generation_context, generation_metadata)
            VALUES (:object_id, :generation_context, :generation_metadata)
        """
        )

        await session.execute(
            insert_query,
            {
                "object_id": object_id,
                "generation_context": json.dumps(
                    {}
                ),  # Context would be complex to serialize
                "generation_metadata": json.dumps(generated_object.generation_metadata),
            },
        )

    async def _store_object_placement(self, session: AsyncSession, placement_info):
        """Store object placement information."""
        if not placement_info:
            return

        insert_query = text(
            """
            INSERT INTO object_placements (object_id, location_id, placement_data, visibility_rules)
            VALUES (:object_id, :location_id, :placement_data, :visibility_rules)
            ON CONFLICT (object_id) DO UPDATE SET
                location_id = EXCLUDED.location_id,
                placement_data = EXCLUDED.placement_data,
                visibility_rules = EXCLUDED.visibility_rules
        """
        )

        placement_data = {
            "placement_type": placement_info.placement_type,
            "visibility": placement_info.visibility,
            "accessibility": placement_info.accessibility,
            "spatial_description": placement_info.spatial_description,
            "discovery_difficulty": placement_info.discovery_difficulty,
            "placement_metadata": placement_info.placement_metadata,
        }

        await session.execute(
            insert_query,
            {
                "object_id": placement_info.object_id,
                "location_id": placement_info.location_id,
                "placement_data": json.dumps(placement_data),
                "visibility_rules": json.dumps(
                    {"visibility": placement_info.visibility}
                ),
            },
        )

    def _create_embedding_text(self, generated_object: GeneratedObject) -> str:
        """Create text representation for embedding generation."""
        parts = [
            generated_object.properties.name,
            generated_object.properties.object_type,
            generated_object.properties.description,
            generated_object.properties.material,
            " ".join(generated_object.properties.special_properties),
            generated_object.properties.cultural_significance,
        ]

        # Add interaction context
        if generated_object.interactions.available_actions:
            parts.append(" ".join(generated_object.interactions.available_actions))

        if generated_object.interactions.examination_text:
            parts.append(generated_object.interactions.examination_text)

        return " ".join(filter(None, parts))

    def _row_to_generated_object(self, row) -> GeneratedObject:
        """Convert database row to GeneratedObject instance."""
        from game_loop.core.models.object_models import (
            ObjectInteractions,
            ObjectProperties,
        )
        from game_loop.state.models import WorldObject

        # Parse properties
        properties_data = json.loads(row.properties) if row.properties else {}
        properties = ObjectProperties(
            name=properties_data.get("name", row.name),
            object_type=properties_data.get("object_type", "unknown"),
            material=properties_data.get("material", "unknown"),
            size=properties_data.get("size", "medium"),
            weight=properties_data.get("weight", "normal"),
            durability=properties_data.get("durability", "sturdy"),
            value=properties_data.get("value", 0),
            special_properties=properties_data.get("special_properties", []),
            cultural_significance=properties_data.get(
                "cultural_significance", "common"
            ),
            description=properties_data.get("description", row.description or ""),
        )

        # Parse interactions
        interactions_data = json.loads(row.interactions) if row.interactions else {}
        interactions = ObjectInteractions(
            available_actions=interactions_data.get("available_actions", ["examine"]),
            use_requirements=interactions_data.get("use_requirements", {}),
            interaction_results=interactions_data.get("interaction_results", {}),
            state_changes=interactions_data.get("state_changes", {}),
            consumable=interactions_data.get("consumable", False),
            portable=interactions_data.get("portable", True),
            examination_text=interactions_data.get("examination_text", ""),
            hidden_properties=interactions_data.get("hidden_properties", {}),
        )

        # Create base object
        base_object = WorldObject(
            object_id=row.object_id, name=row.name, description=row.description or ""
        )

        # Parse generation metadata
        generation_metadata = (
            json.loads(row.generation_metadata) if row.generation_metadata else {}
        )

        return GeneratedObject(
            base_object=base_object,
            properties=properties,
            interactions=interactions,
            generation_metadata=generation_metadata,
            embedding_vector=row.embedding if row.embedding else [],
        )
