"""SQLAlchemy ORM models for the Game Loop database."""

from .base import (
    Base,
    DatabaseError,
    EntityNotFoundError,
    TimestampMixin,
    UUIDMixin,
    ValidationError,
)
from .conversation import (
    ConversationContext,
    ConversationExchange,
    ConversationKnowledge,
    EmotionalContext,
    MemoryAccessLog,
    MemoryCluster,
    MemoryEmbedding,
    MemoryPersonalityConfig,
    NPCPersonality,
)
from .game_state import EvolutionEvent, GameSession, WorldRule
from .player import Player, PlayerHistory, PlayerInventory, PlayerKnowledge, PlayerSkill
from .world import NPC, Location, LocationConnection, Object, Quest, Region

__all__ = [
    # Base classes
    "Base",
    "TimestampMixin",
    "UUIDMixin",
    "DatabaseError",
    "EntityNotFoundError",
    "ValidationError",
    # Conversation models
    "NPCPersonality",
    "ConversationContext",
    "ConversationExchange",
    "ConversationKnowledge",
    "MemoryEmbedding",
    "MemoryAccessLog",
    "MemoryPersonalityConfig",
    "EmotionalContext",
    "MemoryCluster",
    # Player models
    "Player",
    "PlayerInventory",
    "PlayerKnowledge",
    "PlayerSkill",
    "PlayerHistory",
    # World models
    "Region",
    "Location",
    "Object",
    "NPC",
    "Quest",
    "LocationConnection",
    # Game state models
    "GameSession",
    "WorldRule",
    "EvolutionEvent",
]
