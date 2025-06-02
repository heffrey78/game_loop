"""
Embedding database manager for storing and retrieving entity embeddings.

This module provides database operations for managing entity embeddings,
ensuring they are correctly stored and retrieved from the database.
"""

import logging
from typing import Any

from sqlalchemy import delete, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from ...embeddings.entity_registry import EntityEmbeddingRegistry
from ..models.embedding import EntityEmbedding
from ..session_factory import DatabaseSessionFactory

logger = logging.getLogger(__name__)


class EmbeddingDatabaseManager:
    """Manage entity embeddings in the database."""

    def __init__(
        self,
        session_factory: DatabaseSessionFactory,
        registry: EntityEmbeddingRegistry | None = None,
    ):
        """
        Initialize the embedding database manager.

        Args:
            session_factory: Factory for creating database sessions
            registry: Optional embedding registry for in-memory operations
        """
        self.session_factory = session_factory
        self.registry = registry

    async def store_embedding(
        self,
        entity_id: str,
        entity_type: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Store an entity embedding in the database.

        Args:
            entity_id: Unique identifier for the entity
            entity_type: Type of the entity
            embedding: Vector representation of the entity
            metadata: Optional metadata for the entity
        """
        async with self.session_factory.get_session() as session:
            # Check if entity already exists
            existing = await self._get_entity_by_id(session, entity_id)

            if existing:
                # Update existing entity
                await self._update_entity_embedding(
                    session, entity_id, embedding, metadata
                )
            else:
                # Create new entity embedding
                entity_embedding = EntityEmbedding(
                    entity_id=entity_id,
                    entity_type=entity_type,
                    embedding=embedding,
                    metadata=metadata or {},
                )
                session.add(entity_embedding)

            await session.commit()
            logger.debug(f"Stored embedding for entity: {entity_id}")

    async def store_embeddings_batch(self, entities: list[dict[str, Any]]) -> None:
        """
        Store multiple entity embeddings in a single transaction.

        Args:
            entities: List of entity dictionaries with id, type,
            embedding, and optional metadata
        """
        async with self.session_factory.get_session() as session:
            for entity_data in entities:
                entity_id = entity_data.get("entity_id")
                entity_type = entity_data.get("entity_type")
                embedding = entity_data.get("embedding")
                metadata = entity_data.get("metadata", {})

                if not entity_id or not entity_type or not embedding:
                    logger.warning(f"Skipping invalid entity data: {entity_data}")
                    continue

                # Check if entity already exists
                existing = await self._get_entity_by_id(session, entity_id)

                if existing:
                    # Update existing entity
                    await self._update_entity_embedding(
                        session, entity_id, embedding, metadata
                    )
                else:
                    # Create new entity embedding
                    entity_embedding = EntityEmbedding(
                        entity_id=entity_id,
                        entity_type=entity_type,
                        embedding=embedding,
                        metadata=metadata,
                    )
                    session.add(entity_embedding)

            await session.commit()
            logger.debug(f"Stored batch of {len(entities)} embeddings")

    async def get_embedding(self, entity_id: str) -> list[float]:
        """
        Retrieve an entity embedding from the database.

        Args:
            entity_id: Entity ID to retrieve

        Returns:
            Vector embedding for the entity

        Raises:
            ValueError: If entity not found
        """
        async with self.session_factory.get_session() as session:
            entity = await self._get_entity_by_id(session, entity_id)

            if not entity:
                raise ValueError(f"Entity not found: {entity_id}")

            return list(entity.embedding)  # Convert to list if needed

    async def get_embeddings_batch(
        self, entity_ids: list[str]
    ) -> dict[str, list[float]]:
        """
        Retrieve multiple entity embeddings in a single query.

        Args:
            entity_ids: List of entity IDs to retrieve

        Returns:
            Dictionary mapping entity IDs to embeddings
        """
        result = {}
        async with self.session_factory.get_session() as session:
            stmt = select(EntityEmbedding).where(
                EntityEmbedding.entity_id.in_(entity_ids)
            )
            query_result = await session.execute(stmt)
            entities = query_result.scalars().all()

            for entity in entities:
                result[entity.entity_id] = list(entity.embedding)

            # Log missing entities
            found_ids = set(result.keys())
            missing_ids = set(entity_ids) - found_ids  # type: ignore[operator]
            if missing_ids:
                logger.warning(f"Missing embeddings for entities: {missing_ids}")

            return result  # type: ignore[return-value]

    async def update_embedding(
        self,
        entity_id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Update an existing entity embedding.

        Args:
            entity_id: Entity ID to update
            embedding: New vector embedding
            metadata: Optional new metadata

        Raises:
            ValueError: If entity not found
        """
        async with self.session_factory.get_session() as session:
            await self._update_entity_embedding(session, entity_id, embedding, metadata)
            await session.commit()

    async def delete_embedding(self, entity_id: str) -> None:
        """
        Delete an entity embedding from the database.

        Args:
            entity_id: Entity ID to delete
        """
        async with self.session_factory.get_session() as session:
            stmt = delete(EntityEmbedding).where(EntityEmbedding.entity_id == entity_id)
            result = await session.execute(stmt)
            await session.commit()

            if result.rowcount == 0:
                logger.warning(f"No embedding found to delete for entity: {entity_id}")
            else:
                logger.debug(f"Deleted embedding for entity: {entity_id}")

    async def get_embeddings_by_entity_type(
        self, entity_type: str
    ) -> dict[str, list[float]]:
        """
        Get all embeddings for a specific entity type.

        Args:
            entity_type: Entity type to filter by

        Returns:
            Dictionary mapping entity IDs to embeddings
        """
        result = {}
        async with self.session_factory.get_session() as session:
            stmt = select(EntityEmbedding).where(
                EntityEmbedding.entity_type == entity_type
            )
            query_result = await session.execute(stmt)
            entities = query_result.scalars().all()

            for entity in entities:
                result[entity.entity_id] = list(entity.embedding)

            return result  # type: ignore[return-value]

    async def get_all_embeddings(self) -> dict[str, list[float]]:
        """
        Get all embeddings from the database.

        Returns:
            Dictionary mapping entity IDs to embeddings
        """
        result = {}
        async with self.session_factory.get_session() as session:
            stmt = select(EntityEmbedding)
            query_result = await session.execute(stmt)
            entities = query_result.scalars().all()

            for entity in entities:
                result[entity.entity_id] = list(entity.embedding)

            return result  # type: ignore[return-value]

    async def get_embedding_metadata(self, entity_id: str) -> dict[str, Any]:
        """
        Get metadata for an entity embedding.

        Args:
            entity_id: Entity ID

        Returns:
            Metadata dictionary

        Raises:
            ValueError: If entity not found
        """
        async with self.session_factory.get_session() as session:
            entity = await self._get_entity_by_id(session, entity_id)

            if not entity:
                raise ValueError(f"Entity not found: {entity_id}")

            return entity.metadata if isinstance(entity.metadata, dict) else {}  # type: ignore[call-overload]

    async def get_entity_type(self, entity_id: str) -> str:
        """
        Get entity type for an entity ID.

        Args:
            entity_id: Entity ID

        Returns:
            Entity type

        Raises:
            ValueError: If entity not found
        """
        async with self.session_factory.get_session() as session:
            entity = await self._get_entity_by_id(session, entity_id)

            if not entity:
                raise ValueError(f"Entity not found: {entity_id}")

            return str(entity.entity_type)

    async def sync_with_registry(self) -> None:
        """
        Synchronize database embeddings with the in-memory registry.

        This ensures that the database has all the embeddings from the registry.
        """
        if not self.registry:
            logger.warning("No registry provided, cannot synchronize")
            return

        # Get all embeddings from registry
        if hasattr(self.registry, "get_all_embeddings"):
            registry_embeddings = self.registry.get_all_embeddings()

            # Prepare batch of entity data
            entities = []
            for entity_id, embedding in registry_embeddings.items():
                entity_type = (
                    self.registry.entity_types.get(entity_id)
                    if hasattr(self.registry, "entity_types")
                    else "unknown"
                )

                entities.append(
                    {
                        "entity_id": entity_id,
                        "entity_type": entity_type or "unknown",
                        "embedding": embedding,
                    }
                )

            # Store in database
            await self.store_embeddings_batch(entities)
            logger.info(
                f"Synchronized {len(entities)} embeddings from registry to database"
            )

    async def _get_entity_by_id(
        self, session: AsyncSession, entity_id: str
    ) -> EntityEmbedding | None:
        """
        Helper method to get entity by ID.

        Args:
            session: Database session
            entity_id: Entity ID

        Returns:
            EntityEmbedding object or None
        """
        stmt = select(EntityEmbedding).where(EntityEmbedding.entity_id == entity_id)
        result = await session.execute(stmt)
        return result.scalars().first()  # type: ignore[no-any-return]

    async def _update_entity_embedding(
        self,
        session: AsyncSession,
        entity_id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Helper method to update entity embedding.

        Args:
            session: Database session
            entity_id: Entity ID
            embedding: New vector embedding
            metadata: Optional new metadata

        Raises:
            ValueError: If entity not found
        """
        entity = await self._get_entity_by_id(session, entity_id)

        if not entity:
            raise ValueError(f"Entity not found for update: {entity_id}")

        # Prepare update values
        update_values: dict[str, Any] = {"embedding": embedding}

        if metadata is not None:
            # Merge with existing metadata if available
            if entity.metadata:
                existing_metadata = (
                    entity.metadata if isinstance(entity.metadata, dict) else {}
                )  # type: ignore[call-overload]
                merged_metadata = {**existing_metadata, **metadata}
                update_values["metadata"] = merged_metadata
            else:
                update_values["metadata"] = metadata

        # Execute update
        stmt = (
            update(EntityEmbedding)
            .where(EntityEmbedding.entity_id == entity_id)
            .values(**update_values)
        )
        await session.execute(stmt)
