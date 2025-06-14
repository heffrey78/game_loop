"""
Location storage for handling storage, retrieval, and caching of generated locations.
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from ...database.session_factory import DatabaseSessionFactory
from ...embeddings.manager import EmbeddingManager
from ...state.models import Location
from ..models.location_models import (
    CachedGeneration,
    EmbeddingUpdateResult,
    GeneratedLocation,
    LocationConnection,
    LocationTheme,
    StorageResult,
)

logger = logging.getLogger(__name__)


class LocationStorage:
    """Handles storage, retrieval, and caching of generated locations with embedding integration."""

    def __init__(
        self,
        session_factory: DatabaseSessionFactory,
        embedding_manager: EmbeddingManager,
    ):
        self.session_factory = session_factory
        self.embedding_manager = embedding_manager
        self._memory_cache: dict[UUID, Location] = {}
        self._generation_cache: dict[str, CachedGeneration] = {}

    async def store_generated_location(
        self, location: GeneratedLocation
    ) -> StorageResult:
        """Store location with embeddings and relationships."""
        start_time = datetime.now()

        try:
            logger.debug(f"Storing generated location: {location.name}")

            # Generate new location ID
            location_id = uuid4()

            # Store in database
            async with self.session_factory.get_session() as session:
                # Insert location record
                location_query = """
                    INSERT INTO locations (
                        location_id, name, description, connections, objects, npcs,
                        theme_id, generation_metadata, last_generated_at
                    ) VALUES (
                        :location_id, :name, :description, :connections, :objects, :npcs,
                        :theme_id, :generation_metadata, :last_generated_at
                    )
                """

                # Prepare data
                connections_json = json.dumps(location.connections)
                objects_json = json.dumps({obj: {} for obj in location.objects})
                npcs_json = json.dumps({npc: {} for npc in location.npcs})

                generation_metadata = {
                    "location_type": location.location_type,
                    "atmosphere": location.atmosphere,
                    "special_features": location.special_features,
                    "generation_source": "llm_generated",
                    **location.metadata,
                }

                await session.execute(
                    location_query,
                    {
                        "location_id": location_id,
                        "name": location.name,
                        "description": location.description,
                        "connections": connections_json,
                        "objects": objects_json,
                        "npcs": npcs_json,
                        "theme_id": (
                            location.theme.theme_id if location.theme.theme_id else None
                        ),
                        "generation_metadata": json.dumps(generation_metadata),
                        "last_generated_at": datetime.now(),
                    },
                )

                # Store generation history
                if location.generation_context:
                    await self._store_generation_history(session, location_id, location)

                await session.commit()

            # Generate and store embeddings
            embedding_generated = False
            try:
                embedding_success = (
                    await self.embedding_manager.create_or_update_location_embedding(
                        location_id
                    )
                )
                embedding_generated = embedding_success
            except Exception as e:
                logger.warning(
                    f"Failed to generate embeddings for location {location_id}: {e}"
                )

            # Create proper WorldObject and NonPlayerCharacter instances
            from ...state.models import NonPlayerCharacter, WorldObject

            objects_dict = {}
            for obj_name in location.objects:
                obj_id = uuid4()
                objects_dict[obj_id] = WorldObject(
                    object_id=obj_id,
                    name=obj_name,
                    description=f"A {obj_name} in {location.name}",
                    is_takeable=(
                        True
                        if "mushroom" in obj_name.lower() or "coin" in obj_name.lower()
                        else False
                    ),
                )

            npcs_dict = {}
            for npc_name in location.npcs:
                npc_id = uuid4()
                npcs_dict[npc_id] = NonPlayerCharacter(
                    npc_id=npc_id,
                    name=npc_name,
                    description=f"A {npc_name} in {location.name}",
                    dialogue_state="neutral",
                )

            # Cache in memory
            cached_location = Location(
                location_id=location_id,
                name=location.name,
                description=location.description,
                connections={},  # Will be populated when connections are created
                objects=objects_dict,
                npcs=npcs_dict,
                state_flags=generation_metadata,
            )
            self._memory_cache[location_id] = cached_location

            storage_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            return StorageResult(
                success=True,
                location_id=location_id,
                storage_time_ms=storage_time_ms,
                embedding_generated=embedding_generated,
            )

        except Exception as e:
            logger.error(f"Error storing generated location: {e}")
            storage_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            return StorageResult(
                success=False, error_message=str(e), storage_time_ms=storage_time_ms
            )

    async def _store_generation_history(
        self, session, location_id: UUID, location: GeneratedLocation
    ) -> None:
        """Store location generation history."""
        try:
            history_query = """
                INSERT INTO location_generation_history (
                    location_id, generation_context, generated_content,
                    validation_result, generation_time_ms
                ) VALUES (
                    :location_id, :generation_context, :generated_content,
                    :validation_result, :generation_time_ms
                )
            """

            # Serialize context (simplified)
            context_data = {
                "expansion_point": {
                    "location_id": str(
                        location.generation_context.expansion_point.location_id
                    ),
                    "direction": location.generation_context.expansion_point.direction,
                    "priority": location.generation_context.expansion_point.priority,
                },
                "adjacent_locations": len(
                    location.generation_context.adjacent_locations
                ),
                "player_preferences": {
                    "environments": location.generation_context.player_preferences.environments,
                    "complexity_level": location.generation_context.player_preferences.complexity_level,
                },
                "world_themes": [
                    theme.name for theme in location.generation_context.world_themes
                ],
            }

            # Serialize generated content
            content_data = {
                "name": location.name,
                "description": location.description,
                "theme": location.theme.name,
                "location_type": location.location_type,
                "objects": location.objects,
                "npcs": location.npcs,
                "connections": location.connections,
                "special_features": location.special_features,
            }

            await session.execute(
                history_query,
                {
                    "location_id": location_id,
                    "generation_context": json.dumps(context_data),
                    "generated_content": json.dumps(content_data),
                    "validation_result": None,  # Would be populated if validation was performed
                    "generation_time_ms": location.metadata.get(
                        "generation_time_ms", 0
                    ),
                },
            )

        except Exception as e:
            logger.error(f"Error storing generation history: {e}")

    async def retrieve_location(
        self, location_id: UUID, include_embeddings: bool = False
    ) -> Location | None:
        """Retrieve location with optional embedding data."""

        # Check memory cache first
        if location_id in self._memory_cache:
            logger.debug(f"Retrieved location {location_id} from memory cache")
            return self._memory_cache[location_id]

        try:
            async with self.session_factory.get_session() as session:
                query = """
                    SELECT l.location_id, l.name, l.description, l.connections,
                           l.objects, l.npcs, l.state_flags, l.generation_metadata,
                           lt.name as theme_name
                    FROM locations l
                    LEFT JOIN location_themes lt ON l.theme_id = lt.theme_id
                    WHERE l.location_id = :location_id
                """

                result = await session.execute(query, {"location_id": location_id})
                row = result.fetchone()

                if not row:
                    return None

                # Parse JSON fields
                connections = json.loads(row.connections) if row.connections else {}
                objects = json.loads(row.objects) if row.objects else {}
                npcs = json.loads(row.npcs) if row.npcs else {}
                state_flags = json.loads(row.state_flags) if row.state_flags else {}
                generation_metadata = (
                    json.loads(row.generation_metadata)
                    if row.generation_metadata
                    else {}
                )

                # Merge state flags and generation metadata
                combined_flags = {**state_flags, **generation_metadata}
                if row.theme_name:
                    combined_flags["theme"] = row.theme_name

                location = Location(
                    location_id=location_id,
                    name=row.name,
                    description=row.description,
                    connections=connections,
                    objects=objects,
                    npcs=npcs,
                    state_flags=combined_flags,
                )

                # Cache for future retrieval
                self._memory_cache[location_id] = location

                return location

        except Exception as e:
            logger.error(f"Error retrieving location {location_id}: {e}")
            return None

    async def cache_location(
        self, location: Location, cache_duration: timedelta
    ) -> None:
        """Cache location for improved retrieval performance."""
        self._memory_cache[location.location_id] = location
        logger.debug(f"Cached location {location.location_id} for {cache_duration}")

    async def update_location_embeddings(
        self, location_id: UUID
    ) -> EmbeddingUpdateResult:
        """Update location embeddings after content changes."""
        start_time = datetime.now()

        try:
            logger.debug(f"Updating embeddings for location {location_id}")

            # Use embedding manager to update embeddings
            success = await self.embedding_manager.create_or_update_location_embedding(
                location_id
            )

            update_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            if success:
                return EmbeddingUpdateResult(
                    success=True, location_id=location_id, update_time_ms=update_time_ms
                )
            else:
                return EmbeddingUpdateResult(
                    success=False,
                    location_id=location_id,
                    update_time_ms=update_time_ms,
                    error_message="Embedding generation failed",
                )

        except Exception as e:
            update_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.error(f"Error updating embeddings for location {location_id}: {e}")
            return EmbeddingUpdateResult(
                success=False,
                location_id=location_id,
                update_time_ms=update_time_ms,
                error_message=str(e),
            )

    async def store_location_connections(
        self, connections: list[LocationConnection]
    ) -> bool:
        """Store location connections in the database."""
        try:
            async with self.session_factory.get_session() as session:
                for connection in connections:
                    # Update the from_location's connections
                    update_query = """
                        UPDATE locations 
                        SET connections = COALESCE(connections, '{}')::jsonb || :new_connection::jsonb
                        WHERE location_id = :location_id
                    """

                    new_connection = {
                        connection.direction: str(connection.to_location_id)
                    }

                    await session.execute(
                        update_query,
                        {
                            "location_id": connection.from_location_id,
                            "new_connection": json.dumps(new_connection),
                        },
                    )

                    # If bidirectional, update the reverse connection
                    if connection.is_bidirectional:
                        reverse_direction = self._get_reverse_direction(
                            connection.direction
                        )
                        if reverse_direction:
                            reverse_connection = {
                                reverse_direction: str(connection.from_location_id)
                            }

                            await session.execute(
                                update_query,
                                {
                                    "location_id": connection.to_location_id,
                                    "new_connection": json.dumps(reverse_connection),
                                },
                            )

                await session.commit()

                # Clear memory cache to force refresh
                for connection in connections:
                    self._memory_cache.pop(connection.from_location_id, None)
                    if connection.is_bidirectional:
                        self._memory_cache.pop(connection.to_location_id, None)

                return True

        except Exception as e:
            logger.error(f"Error storing location connections: {e}")
            return False

    def _get_reverse_direction(self, direction: str) -> str | None:
        """Get the reverse of a direction."""
        reverse_map = {
            "north": "south",
            "south": "north",
            "east": "west",
            "west": "east",
            "up": "down",
            "down": "up",
            "in": "out",
            "out": "in",
        }
        return reverse_map.get(direction)

    async def cache_generation_result(
        self,
        context_hash: str,
        generated_location: GeneratedLocation,
        cache_duration: timedelta,
    ) -> None:
        """Cache a generation result for reuse."""
        cache_expires_at = datetime.now() + cache_duration

        cached_generation = CachedGeneration(
            context_hash=context_hash,
            generated_location=generated_location,
            cache_expires_at=cache_expires_at,
            created_at=datetime.now(),
        )

        self._generation_cache[context_hash] = cached_generation

        # Also store in database cache table
        try:
            async with self.session_factory.get_session() as session:
                cache_query = """
                    INSERT INTO location_generation_cache (
                        context_hash, generated_location, cache_expires_at
                    ) VALUES (:context_hash, :generated_location, :cache_expires_at)
                    ON CONFLICT (context_hash) DO UPDATE SET
                        generated_location = EXCLUDED.generated_location,
                        cache_expires_at = EXCLUDED.cache_expires_at,
                        usage_count = location_generation_cache.usage_count + 1
                """

                # Serialize generated location
                location_data = {
                    "name": generated_location.name,
                    "description": generated_location.description,
                    "theme_name": generated_location.theme.name,
                    "location_type": generated_location.location_type,
                    "objects": generated_location.objects,
                    "npcs": generated_location.npcs,
                    "connections": generated_location.connections,
                    "metadata": generated_location.metadata,
                }

                await session.execute(
                    cache_query,
                    {
                        "context_hash": context_hash,
                        "generated_location": json.dumps(location_data),
                        "cache_expires_at": cache_expires_at,
                    },
                )

                await session.commit()

        except Exception as e:
            logger.error(f"Error caching generation result: {e}")

    async def get_cached_generation(self, context_hash: str) -> CachedGeneration | None:
        """Get cached generation result if available and not expired."""

        # Check memory cache first
        if context_hash in self._generation_cache:
            cached = self._generation_cache[context_hash]
            if not cached.is_expired:
                cached.usage_count += 1
                return cached
            else:
                # Remove expired cache
                del self._generation_cache[context_hash]

        # Check database cache
        try:
            async with self.session_factory.get_session() as session:
                cache_query = """
                    SELECT generated_location, cache_expires_at, usage_count, created_at
                    FROM location_generation_cache
                    WHERE context_hash = :context_hash
                      AND cache_expires_at > :current_time
                """

                result = await session.execute(
                    cache_query,
                    {"context_hash": context_hash, "current_time": datetime.now()},
                )

                row = result.fetchone()
                if row:
                    # Deserialize location data (simplified for this example)
                    location_data = json.loads(row.generated_location)

                    # Create a basic GeneratedLocation object
                    # Note: This is simplified - in practice you'd need to reconstruct
                    # the full object with theme and context
                    generated_location = GeneratedLocation(
                        name=location_data["name"],
                        description=location_data["description"],
                        theme=LocationTheme(
                            name=location_data["theme_name"],
                            description="",
                            visual_elements=[],
                            atmosphere="",
                            typical_objects=[],
                            typical_npcs=[],
                            generation_parameters={},
                        ),
                        location_type=location_data["location_type"],
                        objects=location_data["objects"],
                        npcs=location_data["npcs"],
                        connections=location_data["connections"],
                        metadata=location_data["metadata"],
                    )

                    cached_generation = CachedGeneration(
                        context_hash=context_hash,
                        generated_location=generated_location,
                        cache_expires_at=row.cache_expires_at,
                        usage_count=row.usage_count,
                        created_at=row.created_at,
                    )

                    # Update usage count
                    await session.execute(
                        "UPDATE location_generation_cache SET usage_count = usage_count + 1 WHERE context_hash = :context_hash",
                        {"context_hash": context_hash},
                    )
                    await session.commit()

                    # Cache in memory
                    self._generation_cache[context_hash] = cached_generation

                    return cached_generation

        except Exception as e:
            logger.error(f"Error retrieving cached generation: {e}")

        return None

    def generate_context_hash(self, context_data: dict[str, Any]) -> str:
        """Generate a hash for context data to use as cache key."""
        # Create a stable representation of the context
        stable_context = {
            "expansion_point": {
                "location_id": str(
                    context_data.get("expansion_point", {}).get("location_id", "")
                ),
                "direction": context_data.get("expansion_point", {}).get(
                    "direction", ""
                ),
            },
            "adjacent_locations": sorted(
                [
                    loc.get("name", "")
                    for loc in context_data.get("adjacent_locations", [])
                ]
            ),
            "themes": sorted(context_data.get("world_themes", [])),
            "preferences": context_data.get("player_preferences", {}),
        }

        # Create hash
        context_json = json.dumps(stable_context, sort_keys=True)
        return hashlib.sha256(context_json.encode()).hexdigest()

    async def cleanup_expired_cache(self) -> int:
        """Clean up expired cache entries and return count of removed entries."""
        removed_count = 0

        # Clean memory cache
        expired_keys = [
            key for key, cached in self._generation_cache.items() if cached.is_expired
        ]

        for key in expired_keys:
            del self._generation_cache[key]
            removed_count += 1

        # Clean database cache
        try:
            async with self.session_factory.get_session() as session:
                delete_query = """
                    DELETE FROM location_generation_cache
                    WHERE cache_expires_at < :current_time
                """

                result = await session.execute(
                    delete_query, {"current_time": datetime.now()}
                )

                db_removed = result.rowcount
                await session.commit()

                removed_count += db_removed

        except Exception as e:
            logger.error(f"Error cleaning up database cache: {e}")

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} expired cache entries")

        return removed_count

    def clear_cache(self) -> None:
        """Clear all caches."""
        self._memory_cache.clear()
        self._generation_cache.clear()
        logger.debug("Location storage caches cleared")
