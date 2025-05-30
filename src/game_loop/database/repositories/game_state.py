"""Repository for game state entities."""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from game_loop.database.models.game_state import (
    EvolutionEvent,
    GameSession,
    WorldRule,
)

from .base import BaseRepository


class GameSessionRepository(BaseRepository[GameSession]):
    """Repository for GameSession entities."""

    def __init__(self, session: AsyncSession):
        super().__init__(GameSession, session)

    async def get_by_id(self, session_id: uuid.UUID) -> GameSession | None:
        """Get a game session by session_id."""
        session: GameSession | None = await self.session.get(GameSession, session_id)
        return session

    async def get_by_player_id(self, player_id: uuid.UUID) -> list[GameSession]:
        """Get all sessions for a player."""
        result = await self.session.execute(
            select(GameSession).where(GameSession.player_id == player_id)
        )
        return list(result.scalars().all())

    async def get_active_sessions(self) -> list[GameSession]:
        """Get all active sessions (ended_at is None)."""
        result = await self.session.execute(
            select(GameSession).where(GameSession.ended_at.is_(None))
        )
        return list(result.scalars().all())


class WorldRuleRepository(BaseRepository[WorldRule]):
    """Repository for WorldRule entities."""

    def __init__(self, session: AsyncSession):
        super().__init__(WorldRule, session)

    async def get_by_type(self, rule_type: str) -> list[WorldRule]:
        """Get rules by type."""
        result = await self.session.execute(
            select(WorldRule).where(WorldRule.rule_type == rule_type)
        )
        return list(result.scalars().all())


class EvolutionEventRepository(BaseRepository[EvolutionEvent]):
    """Repository for EvolutionEvent entities."""

    def __init__(self, session: AsyncSession):
        super().__init__(EvolutionEvent, session)

    async def get_unprocessed_events(self) -> list[EvolutionEvent]:
        """Get all unprocessed events."""
        result = await self.session.execute(
            select(EvolutionEvent).where(~EvolutionEvent.is_processed)
        )
        return list(result.scalars().all())

    async def get_by_target(self, target_id: uuid.UUID) -> list[EvolutionEvent]:
        """Get events by target ID."""
        result = await self.session.execute(
            select(EvolutionEvent).where(EvolutionEvent.target_id == target_id)
        )
        return list(result.scalars().all())
