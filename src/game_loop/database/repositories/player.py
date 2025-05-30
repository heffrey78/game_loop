"""Repository for player-related entities."""

import uuid
from typing import cast

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from game_loop.database.models.player import (
    Player,
    PlayerHistory,
    PlayerInventory,
    PlayerKnowledge,
    PlayerSkill,
)

from .base import BaseRepository


class PlayerRepository(BaseRepository[Player]):
    """Repository for Player entities."""

    def __init__(self, session: AsyncSession):
        super().__init__(Player, session)

    async def get_by_username(self, username: str) -> Player | None:
        """Get a player by username."""
        result = await self.session.execute(
            select(Player).where(Player.username == username)
        )
        player = result.scalar_one_or_none()
        return cast(Player | None, player)

    async def get_with_full_data(self, player_id: uuid.UUID) -> Player | None:
        """Get a player with all related data loaded."""
        result = await self.session.execute(
            select(Player)
            .options(
                selectinload(Player.inventory_items),
                selectinload(Player.knowledge_items),
                selectinload(Player.skills),
                selectinload(Player.history_events),
            )
            .where(Player.id == player_id)
        )
        player = result.scalar_one_or_none()
        return cast(Player | None, player)


class PlayerInventoryRepository(BaseRepository[PlayerInventory]):
    """Repository for PlayerInventory entities."""

    def __init__(self, session: AsyncSession):
        super().__init__(PlayerInventory, session)

    async def get_by_player_id(self, player_id: uuid.UUID) -> list[PlayerInventory]:
        """Get all inventory items for a player."""
        result = await self.session.execute(
            select(PlayerInventory).where(PlayerInventory.player_id == player_id)
        )
        return list(result.scalars().all())

    async def get_by_object_id(
        self, player_id: uuid.UUID, object_id: uuid.UUID
    ) -> PlayerInventory | None:
        """Get a specific inventory item."""
        result = await self.session.execute(
            select(PlayerInventory).where(
                PlayerInventory.player_id == player_id,
                PlayerInventory.object_id == object_id,
            )
        )
        item = result.scalar_one_or_none()
        return cast(PlayerInventory | None, item)


class PlayerKnowledgeRepository(BaseRepository[PlayerKnowledge]):
    """Repository for PlayerKnowledge entities."""

    def __init__(self, session: AsyncSession):
        super().__init__(PlayerKnowledge, session)

    async def get_by_player_id(self, player_id: uuid.UUID) -> list[PlayerKnowledge]:
        """Get all knowledge for a player."""
        result = await self.session.execute(
            select(PlayerKnowledge).where(PlayerKnowledge.player_id == player_id)
        )
        return list(result.scalars().all())

    async def get_by_key(
        self, player_id: uuid.UUID, knowledge_key: str
    ) -> PlayerKnowledge | None:
        """Get specific knowledge by key."""
        result = await self.session.execute(
            select(PlayerKnowledge).where(
                PlayerKnowledge.player_id == player_id,
                PlayerKnowledge.knowledge_key == knowledge_key,
            )
        )
        knowledge = result.scalar_one_or_none()
        return cast(PlayerKnowledge | None, knowledge)


class PlayerSkillRepository(BaseRepository[PlayerSkill]):
    """Repository for PlayerSkill entities."""

    def __init__(self, session: AsyncSession):
        super().__init__(PlayerSkill, session)

    async def get_by_player_id(self, player_id: uuid.UUID) -> list[PlayerSkill]:
        """Get all skills for a player."""
        result = await self.session.execute(
            select(PlayerSkill).where(PlayerSkill.player_id == player_id)
        )
        return list(result.scalars().all())

    async def get_by_skill_name(
        self, player_id: uuid.UUID, skill_name: str
    ) -> PlayerSkill | None:
        """Get a specific skill by name."""
        result = await self.session.execute(
            select(PlayerSkill).where(
                PlayerSkill.player_id == player_id,
                PlayerSkill.skill_name == skill_name,
            )
        )
        skill = result.scalar_one_or_none()
        return cast(PlayerSkill | None, skill)


class PlayerHistoryRepository(BaseRepository[PlayerHistory]):
    """Repository for PlayerHistory entities."""

    def __init__(self, session: AsyncSession):
        super().__init__(PlayerHistory, session)

    async def get_by_player_id(
        self, player_id: uuid.UUID, limit: int = 100
    ) -> list[PlayerHistory]:
        """Get recent history for a player."""
        result = await self.session.execute(
            select(PlayerHistory)
            .where(PlayerHistory.player_id == player_id)
            .order_by(PlayerHistory.event_timestamp.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_event_type(
        self, player_id: uuid.UUID, event_type: str
    ) -> list[PlayerHistory]:
        """Get history events by type."""
        result = await self.session.execute(
            select(PlayerHistory).where(
                PlayerHistory.player_id == player_id,
                PlayerHistory.event_type == event_type,
            )
        )
        return list(result.scalars().all())
