"""Models for conversation and NPC interaction system."""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ConversationStatus(Enum):
    """Status of a conversation."""

    ACTIVE = "active"
    ENDED = "ended"
    PAUSED = "paused"
    ABANDONED = "abandoned"


class MessageType(Enum):
    """Types of conversation messages."""

    GREETING = "greeting"
    QUESTION = "question"
    STATEMENT = "statement"
    FAREWELL = "farewell"
    SYSTEM = "system"


@dataclass
class ConversationExchange:
    """Single exchange in a conversation."""

    exchange_id: str
    speaker_id: str  # player_id or npc_id
    message_text: str
    message_type: MessageType
    emotion: str | None = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create_player_message(
        cls,
        player_id: str,
        message_text: str,
        message_type: MessageType = MessageType.STATEMENT,
        metadata: dict[str, Any] | None = None,
    ) -> "ConversationExchange":
        """Create a player message exchange."""
        return cls(
            exchange_id=str(uuid.uuid4()),
            speaker_id=player_id,
            message_text=message_text,
            message_type=message_type,
            metadata=metadata or {},
        )

    @classmethod
    def create_npc_message(
        cls,
        npc_id: str,
        message_text: str,
        message_type: MessageType = MessageType.STATEMENT,
        emotion: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ConversationExchange":
        """Create an NPC message exchange."""
        return cls(
            exchange_id=str(uuid.uuid4()),
            speaker_id=npc_id,
            message_text=message_text,
            message_type=message_type,
            emotion=emotion,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "exchange_id": self.exchange_id,
            "speaker_id": self.speaker_id,
            "message_text": self.message_text,
            "message_type": self.message_type.value,
            "emotion": self.emotion,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class NPCPersonality:
    """Defines NPC personality traits and characteristics."""

    npc_id: str
    traits: dict[str, float]  # personality traits with strength values
    knowledge_areas: list[str]  # areas of expertise
    speech_patterns: dict[str, Any]  # speech characteristics
    relationships: dict[str, float]  # relationships with other entities
    background_story: str
    default_mood: str = "neutral"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def get_trait_strength(self, trait: str) -> float:
        """Get the strength of a personality trait (0.0 to 1.0)."""
        return self.traits.get(trait, 0.0)

    def get_relationship_level(self, entity_id: str) -> float:
        """Get relationship level with another entity (-1.0 to 1.0)."""
        return self.relationships.get(entity_id, 0.0)

    def update_relationship(self, entity_id: str, change: float) -> None:
        """Update relationship level with another entity."""
        current = self.relationships.get(entity_id, 0.0)
        new_level = max(-1.0, min(1.0, current + change))
        self.relationships[entity_id] = new_level
        self.updated_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "npc_id": self.npc_id,
            "traits": self.traits,
            "knowledge_areas": self.knowledge_areas,
            "speech_patterns": self.speech_patterns,
            "relationships": self.relationships,
            "background_story": self.background_story,
            "default_mood": self.default_mood,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class ConversationContext:
    """Tracks conversation state and history."""

    conversation_id: str
    player_id: str
    npc_id: str
    topic: str | None = None
    mood: str = "neutral"
    relationship_level: float = 0.0
    conversation_history: list[ConversationExchange] = field(default_factory=list)
    context_data: dict[str, Any] = field(default_factory=dict)
    status: ConversationStatus = ConversationStatus.ACTIVE
    started_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    ended_at: float | None = None

    @classmethod
    def create(
        cls,
        player_id: str,
        npc_id: str,
        topic: str | None = None,
        initial_mood: str = "neutral",
        relationship_level: float = 0.0,
    ) -> "ConversationContext":
        """Create a new conversation context."""
        return cls(
            conversation_id=str(uuid.uuid4()),
            player_id=player_id,
            npc_id=npc_id,
            topic=topic,
            mood=initial_mood,
            relationship_level=relationship_level,
        )

    def add_exchange(self, exchange: ConversationExchange) -> None:
        """Add a new exchange to the conversation."""
        self.conversation_history.append(exchange)
        self.last_updated = time.time()

    def get_recent_exchanges(self, limit: int = 10) -> list[ConversationExchange]:
        """Get the most recent exchanges."""
        return self.conversation_history[-limit:]

    def get_exchange_count(self) -> int:
        """Get total number of exchanges in conversation."""
        return len(self.conversation_history)

    def end_conversation(self, reason: str = "natural_end") -> None:
        """Mark conversation as ended."""
        self.status = ConversationStatus.ENDED
        self.ended_at = time.time()
        self.context_data["end_reason"] = reason

    def update_mood(self, new_mood: str) -> None:
        """Update the current mood of the conversation."""
        self.mood = new_mood
        self.last_updated = time.time()

    def update_relationship(self, change: float) -> None:
        """Update the relationship level."""
        self.relationship_level = max(-1.0, min(1.0, self.relationship_level + change))
        self.last_updated = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "conversation_id": self.conversation_id,
            "player_id": self.player_id,
            "npc_id": self.npc_id,
            "topic": self.topic,
            "mood": self.mood,
            "relationship_level": self.relationship_level,
            "conversation_history": [exchange.to_dict() for exchange in self.conversation_history],
            "context_data": self.context_data,
            "status": self.status.value,
            "started_at": self.started_at,
            "last_updated": self.last_updated,
            "ended_at": self.ended_at,
        }


@dataclass
class ConversationResult:
    """Result of processing a conversation interaction."""

    success: bool
    npc_response: ConversationExchange | None = None
    relationship_change: float = 0.0
    mood_change: str | None = None
    knowledge_extracted: list[dict[str, Any]] = field(default_factory=list)
    context_updates: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success_result(
        cls,
        npc_response: ConversationExchange,
        relationship_change: float = 0.0,
        mood_change: str | None = None,
        knowledge_extracted: list[dict[str, Any]] | None = None,
        context_updates: dict[str, Any] | None = None,
    ) -> "ConversationResult":
        """Create a successful conversation result."""
        return cls(
            success=True,
            npc_response=npc_response,
            relationship_change=relationship_change,
            mood_change=mood_change,
            knowledge_extracted=knowledge_extracted or [],
            context_updates=context_updates or {},
        )

    @classmethod
    def error_result(
        cls, error_message: str, errors: list[str] | None = None
    ) -> "ConversationResult":
        """Create an error conversation result."""
        return cls(
            success=False,
            errors=errors or [error_message],
        )