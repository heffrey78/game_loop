"""Repository for world-related entities."""

import uuid
from typing import cast

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from game_loop.database.models.world import (
    NPC,
    Location,
    LocationConnection,
    Object,
    Quest,
    Region,
)

from .base import BaseRepository


class RegionRepository(BaseRepository[Region]):
    """Repository for Region entities."""

    def __init__(self, session: AsyncSession):
        super().__init__(Region, session)

    async def get_by_name(self, name: str) -> Region | None:
        """Get a region by name."""
        result = await self.session.execute(select(Region).where(Region.name == name))
        region = result.scalar_one_or_none()
        return cast(Region | None, region)


class LocationRepository(BaseRepository[Location]):
    """Repository for Location entities."""

    def __init__(self, session: AsyncSession):
        super().__init__(Location, session)

    async def get_by_region(self, region_id: uuid.UUID) -> list[Location]:
        """Get all locations in a region."""
        result = await self.session.execute(
            select(Location).where(Location.region_id == region_id)
        )
        return list(result.scalars().all())

    async def get_by_name(self, name: str) -> Location | None:
        """Get a location by name."""
        result = await self.session.execute(
            select(Location).where(Location.name == name)
        )
        location = result.scalar_one_or_none()
        return cast(Location | None, location)

    async def find_by_region(self, region_id: uuid.UUID) -> list[Location]:
        """Find all locations in a region (alias for get_by_region)."""
        return await self.get_by_region(region_id)

    async def find_similar_locations(
        self, embedding: list[float], limit: int = 10
    ) -> list[Location]:
        """Find locations similar to the given embedding."""
        # Perform vector similarity search using pgvector <-> operator
        distance_expr = Location.location_embedding.op("<->")(embedding)
        stmt = select(Location).order_by(distance_expr).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


class ObjectRepository(BaseRepository[Object]):
    """Repository for Object entities."""

    def __init__(self, session: AsyncSession):
        super().__init__(Object, session)

    async def get_by_location(self, location_id: uuid.UUID) -> list[Object]:
        """Get all objects in a location."""
        result = await self.session.execute(
            select(Object).where(Object.location_id == location_id)
        )
        return list(result.scalars().all())

    async def get_takeable_objects(self) -> list[Object]:
        """Get all takeable objects."""
        result = await self.session.execute(select(Object).where(Object.is_takeable))
        return list(result.scalars().all())


class NPCRepository(BaseRepository[NPC]):
    """Repository for NPC entities."""

    def __init__(self, session: AsyncSession):
        super().__init__(NPC, session)

    async def get_by_location(self, location_id: uuid.UUID) -> list[NPC]:
        """Get all NPCs in a location."""
        result = await self.session.execute(
            select(NPC).where(NPC.location_id == location_id)
        )
        return list(result.scalars().all())


class QuestRepository(BaseRepository[Quest]):
    """Repository for Quest entities."""

    def __init__(self, session: AsyncSession):
        super().__init__(Quest, session)

    async def get_by_status(self, status: str) -> list[Quest]:
        """Get quests by status."""
        result = await self.session.execute(select(Quest).where(Quest.status == status))
        return list(result.scalars().all())


class LocationConnectionRepository(BaseRepository[LocationConnection]):
    """Repository for LocationConnection entities."""

    def __init__(self, session: AsyncSession):
        super().__init__(LocationConnection, session)

    async def get_connections_from(
        self, location_id: uuid.UUID
    ) -> list[LocationConnection]:
        """Get all connections from a location."""
        result = await self.session.execute(
            select(LocationConnection).where(
                LocationConnection.from_location_id == location_id
            )
        )
        return list(result.scalars().all())

    async def get_connections_to(
        self, location_id: uuid.UUID
    ) -> list[LocationConnection]:
        """Get all connections to a location."""
        result = await self.session.execute(
            select(LocationConnection).where(
                LocationConnection.to_location_id == location_id
            )
        )
        return list(result.scalars().all())
