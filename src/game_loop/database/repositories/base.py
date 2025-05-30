"""Base repository with common CRUD operations."""

import uuid
from typing import Any, Generic, TypeVar, cast

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from game_loop.database.models.base import Base, EntityNotFoundError

T = TypeVar("T", bound=Base)


class BaseRepository(Generic[T]):
    """Base repository providing common database operations."""

    def __init__(self, model_class: type[T], session: AsyncSession):
        self.model_class = model_class
        self.session = session

    async def create(self, entity_data: T) -> T:
        """Create a new entity from existing entity data."""
        self.session.add(entity_data)
        await self.session.flush()
        await self.session.refresh(entity_data)
        return entity_data

    async def get_by_id(self, entity_id: uuid.UUID) -> T | None:
        """Get an entity by its ID."""
        result = await self.session.get(self.model_class, entity_id)
        return cast(T | None, result)

    async def get_by_id_or_raise(self, entity_id: uuid.UUID) -> T:
        """Get an entity by its ID or raise EntityNotFoundError."""
        entity = await self.get_by_id(entity_id)
        if entity is None:
            raise EntityNotFoundError(
                f"{self.model_class.__name__} with id {entity_id} not found"
            )
        return entity

    async def list_all(self) -> list[T]:
        """List all entities."""
        result = await self.session.execute(select(self.model_class))
        return list(result.scalars().all())

    async def get_all(self) -> list[T]:
        """Get all entities (alias for list_all)."""
        return await self.list_all()

    async def update(self, entity: T, **kwargs: Any) -> T:
        """Update an entity."""
        for key, value in kwargs.items():
            if hasattr(entity, key):
                setattr(entity, key, value)
        await self.session.flush()
        await self.session.refresh(entity)
        return entity

    async def delete_by_id(self, entity_id: uuid.UUID) -> bool:
        """Delete an entity by its ID."""
        entity = await self.get_by_id(entity_id)
        if entity:
            await self.session.delete(entity)
            await self.session.flush()
            return True
        return False

    async def delete(self, entity_id: uuid.UUID) -> bool:
        """Delete an entity by ID.

        Returns True if deleted, False if not found.
        """
        entity = await self.get_by_id(entity_id)
        if entity is None:
            return False
        await self.session.delete(entity)
        await self.session.flush()
        return True

    async def delete_entity(self, entity: T) -> None:
        """Delete an entity object."""
        await self.session.delete(entity)
        await self.session.flush()

    async def remove(self, entity: T) -> None:
        """Remove an entity."""
        await self.session.delete(entity)
        await self.session.flush()
