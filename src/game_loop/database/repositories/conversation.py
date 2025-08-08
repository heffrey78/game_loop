"""Repository layer for conversation system database operations."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import and_, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.sql import select

from ..models.conversation import (
    ConversationContext,
    ConversationExchange,
    ConversationKnowledge,
    NPCPersonality,
)
from .base import BaseRepository
from .semantic_memory import SemanticMemoryRepository


class NPCPersonalityRepository(BaseRepository[NPCPersonality]):
    """Repository for NPC personality data."""

    def __init__(self, session: AsyncSession):
        super().__init__(NPCPersonality, session)

    async def get_by_npc_id(self, npc_id: uuid.UUID) -> NPCPersonality | None:
        """Get NPC personality by NPC ID."""
        stmt = select(NPCPersonality).where(NPCPersonality.npc_id == npc_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def create_personality(
        self,
        npc_id: uuid.UUID,
        traits: dict[str, float],
        knowledge_areas: list[str],
        speech_patterns: dict[str, Any],
        background_story: str,
        default_mood: str = "neutral",
    ) -> NPCPersonality:
        """Create a new NPC personality."""
        personality = NPCPersonality(
            npc_id=npc_id,
            traits=traits,
            knowledge_areas=knowledge_areas,
            speech_patterns=speech_patterns,
            background_story=background_story,
            default_mood=default_mood,
            relationships={},
        )
        return await self.create(personality)

    async def update_relationship(
        self, npc_id: uuid.UUID, entity_id: uuid.UUID, change: float
    ) -> NPCPersonality | None:
        """Update relationship level for an NPC."""
        personality = await self.get_by_npc_id(npc_id)
        if personality:
            personality.update_relationship(str(entity_id), change)
            await self.session.commit()
        return personality

    async def get_npcs_by_knowledge_area(
        self, knowledge_area: str
    ) -> list[NPCPersonality]:
        """Get NPCs that have knowledge in a specific area."""
        stmt = select(NPCPersonality).where(
            NPCPersonality.knowledge_areas.any(knowledge_area)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


class ConversationContextRepository(BaseRepository[ConversationContext]):
    """Repository for conversation contexts."""

    def __init__(self, session: AsyncSession):
        super().__init__(ConversationContext, session)

    async def get_with_exchanges(
        self, conversation_id: uuid.UUID
    ) -> ConversationContext | None:
        """Get conversation with all exchanges loaded."""
        stmt = (
            select(ConversationContext)
            .options(selectinload(ConversationContext.exchanges))
            .where(ConversationContext.conversation_id == conversation_id)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_active_conversation(
        self, player_id: uuid.UUID, npc_id: uuid.UUID
    ) -> ConversationContext | None:
        """Get active conversation between player and NPC."""
        stmt = select(ConversationContext).where(
            and_(
                ConversationContext.player_id == player_id,
                ConversationContext.npc_id == npc_id,
                ConversationContext.status == "active",
            )
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_player_conversations(
        self, player_id: uuid.UUID, status: str | None = None, limit: int = 50
    ) -> list[ConversationContext]:
        """Get conversations for a player, optionally filtered by status."""
        stmt = select(ConversationContext).where(
            ConversationContext.player_id == player_id
        )

        if status:
            stmt = stmt.where(ConversationContext.status == status)

        stmt = stmt.order_by(desc(ConversationContext.last_updated)).limit(limit)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def create_conversation(
        self,
        player_id: uuid.UUID,
        npc_id: uuid.UUID,
        topic: str | None = None,
        initial_mood: str = "neutral",
        relationship_level: float = 0.0,
    ) -> ConversationContext:
        """Create a new conversation."""
        conversation = ConversationContext(
            player_id=player_id,
            npc_id=npc_id,
            topic=topic,
            mood=initial_mood,
            relationship_level=relationship_level,
        )
        return await self.create(conversation)

    async def end_conversation(
        self, conversation_id: uuid.UUID, reason: str = "natural_end"
    ) -> ConversationContext | None:
        """End a conversation."""
        conversation = await self.get_by_id(conversation_id)
        if conversation:
            conversation.end_conversation(reason)
            await self.session.commit()
        return conversation

    async def get_recent_conversations(
        self, hours: int = 24, limit: int = 100
    ) -> list[ConversationContext]:
        """Get conversations from the last N hours."""
        cutoff = datetime.utcnow() - datetime.timedelta(hours=hours)
        stmt = (
            select(ConversationContext)
            .where(ConversationContext.started_at >= cutoff)
            .order_by(desc(ConversationContext.started_at))
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


class ConversationExchangeRepository(BaseRepository[ConversationExchange]):
    """Repository for conversation exchanges."""

    def __init__(self, session: AsyncSession):
        super().__init__(ConversationExchange, session)

    async def create_exchange(
        self,
        conversation_id: uuid.UUID,
        speaker_id: uuid.UUID,
        message_text: str,
        message_type: str,
        emotion: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationExchange:
        """Create a new conversation exchange."""
        exchange = ConversationExchange(
            conversation_id=conversation_id,
            speaker_id=speaker_id,
            message_text=message_text,
            message_type=message_type,
            emotion=emotion,
            exchange_metadata=metadata or {},
        )
        return await self.create(exchange)

    async def get_conversation_exchanges(
        self, conversation_id: uuid.UUID, limit: int | None = None
    ) -> list[ConversationExchange]:
        """Get exchanges for a conversation."""
        stmt = (
            select(ConversationExchange)
            .where(ConversationExchange.conversation_id == conversation_id)
            .order_by(ConversationExchange.timestamp)
        )

        if limit:
            stmt = stmt.limit(limit)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_recent_exchanges(
        self, conversation_id: uuid.UUID, limit: int = 10
    ) -> list[ConversationExchange]:
        """Get the most recent exchanges for a conversation."""
        stmt = (
            select(ConversationExchange)
            .where(ConversationExchange.conversation_id == conversation_id)
            .order_by(desc(ConversationExchange.timestamp))
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        # Reverse to get chronological order
        return list(reversed(result.scalars().all()))

    async def search_exchanges_by_content(
        self,
        search_term: str,
        conversation_id: uuid.UUID | None = None,
        limit: int = 50,
    ) -> list[ConversationExchange]:
        """Search exchanges by message content."""
        stmt = select(ConversationExchange).where(
            ConversationExchange.message_text.ilike(f"%{search_term}%")
        )

        if conversation_id:
            stmt = stmt.where(ConversationExchange.conversation_id == conversation_id)

        stmt = stmt.order_by(desc(ConversationExchange.timestamp)).limit(limit)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())


class ConversationKnowledgeRepository(BaseRepository[ConversationKnowledge]):
    """Repository for conversation knowledge extraction."""

    def __init__(self, session: AsyncSession):
        super().__init__(ConversationKnowledge, session)

    async def create_knowledge_entry(
        self,
        conversation_id: uuid.UUID,
        information_type: str,
        extracted_info: dict[str, Any],
        confidence_score: float | None = None,
        source_exchange_id: uuid.UUID | None = None,
    ) -> ConversationKnowledge:
        """Create a new knowledge entry."""
        knowledge = ConversationKnowledge(
            conversation_id=conversation_id,
            information_type=information_type,
            extracted_info=extracted_info,
            confidence_score=confidence_score,
            source_exchange_id=source_exchange_id,
        )
        return await self.create(knowledge)

    async def get_knowledge_by_type(
        self, information_type: str, limit: int = 100
    ) -> list[ConversationKnowledge]:
        """Get knowledge entries by information type."""
        stmt = (
            select(ConversationKnowledge)
            .where(ConversationKnowledge.information_type == information_type)
            .order_by(desc(ConversationKnowledge.created_at))
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_conversation_knowledge(
        self, conversation_id: uuid.UUID
    ) -> list[ConversationKnowledge]:
        """Get all knowledge extracted from a conversation."""
        stmt = (
            select(ConversationKnowledge)
            .where(ConversationKnowledge.conversation_id == conversation_id)
            .order_by(ConversationKnowledge.created_at)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def search_knowledge_content(
        self, search_term: str, information_type: str | None = None, limit: int = 50
    ) -> list[ConversationKnowledge]:
        """Search knowledge by content."""
        # Use PostgreSQL's JSONB operators for searching
        stmt = select(ConversationKnowledge).where(
            ConversationKnowledge.extracted_info.op("@>")({"content": search_term})
        )

        if information_type:
            stmt = stmt.where(
                ConversationKnowledge.information_type == information_type
            )

        stmt = stmt.order_by(desc(ConversationKnowledge.created_at)).limit(limit)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_high_confidence_knowledge(
        self, min_confidence: float = 0.8, limit: int = 100
    ) -> list[ConversationKnowledge]:
        """Get knowledge entries with high confidence scores."""
        stmt = (
            select(ConversationKnowledge)
            .where(ConversationKnowledge.confidence_score >= min_confidence)
            .order_by(desc(ConversationKnowledge.confidence_score))
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


class ConversationRepositoryManager:
    """Manager class that provides access to all conversation repositories."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.npc_personalities = NPCPersonalityRepository(session)
        self.contexts = ConversationContextRepository(session)
        self.exchanges = ConversationExchangeRepository(session)
        self.knowledge = ConversationKnowledgeRepository(session)
        self.semantic_memory = SemanticMemoryRepository(session)

    async def get_full_conversation(self, conversation_id: uuid.UUID) -> tuple[
        ConversationContext | None,
        list[ConversationExchange],
        list[ConversationKnowledge],
    ]:
        """Get complete conversation data including exchanges and knowledge."""
        context = await self.contexts.get_with_exchanges(conversation_id)
        if not context:
            return None, [], []

        exchanges = await self.exchanges.get_conversation_exchanges(conversation_id)
        knowledge = await self.knowledge.get_conversation_knowledge(conversation_id)

        return context, exchanges, knowledge

    async def create_complete_conversation(
        self,
        player_id: uuid.UUID,
        npc_id: uuid.UUID,
        initial_message: str,
        message_type: str = "greeting",
        topic: str | None = None,
        initial_mood: str = "neutral",
    ) -> tuple[ConversationContext, ConversationExchange]:
        """Create a new conversation with initial exchange."""
        # Create conversation
        conversation = await self.contexts.create_conversation(
            player_id=player_id,
            npc_id=npc_id,
            topic=topic,
            initial_mood=initial_mood,
        )

        # Create initial exchange
        exchange = await self.exchanges.create_exchange(
            conversation_id=conversation.conversation_id,
            speaker_id=npc_id,  # NPC greeting
            message_text=initial_message,
            message_type=message_type,
        )

        return conversation, exchange
