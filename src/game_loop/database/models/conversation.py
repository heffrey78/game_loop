"""SQLAlchemy models for conversation system."""

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    DECIMAL,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .base import Base


class NPCPersonality(Base):
    """SQLAlchemy model for NPC personality data."""

    __tablename__ = "npc_personalities"

    npc_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    traits = Column(JSONB, nullable=False, default=dict)
    knowledge_areas = Column(ARRAY(Text), nullable=False, default=list)
    speech_patterns = Column(JSONB, nullable=False, default=dict)
    relationships = Column(JSONB, nullable=False, default=dict)
    background_story = Column(Text, nullable=False, default="")
    default_mood = Column(String(50), nullable=False, default="neutral")
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()
    )

    # Relationships
    conversations = relationship(
        "ConversationContext", back_populates="npc_personality", cascade="all, delete"
    )

    def get_trait_strength(self, trait: str) -> float:
        """Get the strength of a personality trait (0.0 to 1.0)."""
        return self.traits.get(trait, 0.0)

    def get_relationship_level(self, entity_id: str) -> float:
        """Get relationship level with another entity (-1.0 to 1.0)."""
        return self.relationships.get(str(entity_id), 0.0)

    def update_relationship(self, entity_id: str, change: float) -> None:
        """Update relationship level with another entity."""
        current = self.relationships.get(str(entity_id), 0.0)
        new_level = max(-1.0, min(1.0, current + change))
        if not self.relationships:
            self.relationships = {}
        self.relationships[str(entity_id)] = new_level
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "npc_id": str(self.npc_id),
            "traits": self.traits or {},
            "knowledge_areas": self.knowledge_areas or [],
            "speech_patterns": self.speech_patterns or {},
            "relationships": self.relationships or {},
            "background_story": self.background_story,
            "default_mood": self.default_mood,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class ConversationContext(Base):
    """SQLAlchemy model for conversation state tracking."""

    __tablename__ = "conversation_contexts"

    conversation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    player_id = Column(UUID(as_uuid=True), nullable=False)
    npc_id = Column(
        UUID(as_uuid=True), ForeignKey("npc_personalities.npc_id"), nullable=False
    )
    topic = Column(String(255), nullable=True)
    mood = Column(String(50), nullable=False, default="neutral")
    relationship_level = Column(DECIMAL(3, 2), nullable=False, default=0.0)
    context_data = Column(JSONB, nullable=False, default=dict)
    status = Column(String(20), nullable=False, default="active")
    started_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()
    )
    last_updated = Column(
        DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()
    )
    ended_at = Column(DateTime(timezone=True), nullable=True)

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "relationship_level >= -1.0 AND relationship_level <= 1.0",
            name="chk_relationship_level",
        ),
        CheckConstraint(
            "status IN ('active', 'ended', 'paused', 'abandoned')", name="chk_status"
        ),
    )

    # Relationships
    npc_personality = relationship("NPCPersonality", back_populates="conversations")
    exchanges = relationship(
        "ConversationExchange",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ConversationExchange.timestamp",
    )
    knowledge_entries = relationship(
        "ConversationKnowledge",
        back_populates="conversation",
        cascade="all, delete-orphan",
    )

    def get_exchange_count(self) -> int:
        """Get total number of exchanges in conversation."""
        return len(self.exchanges) if self.exchanges else 0

    def update_mood(self, new_mood: str) -> None:
        """Update the current mood of the conversation."""
        self.mood = new_mood
        self.last_updated = datetime.now(timezone.utc)

    def update_relationship(self, change: float) -> None:
        """Update the relationship level."""
        current = float(self.relationship_level)
        self.relationship_level = max(-1.0, min(1.0, current + change))
        self.last_updated = datetime.now(timezone.utc)

    def end_conversation(self, reason: str = "natural_end") -> None:
        """Mark conversation as ended."""
        self.status = "ended"
        self.ended_at = datetime.now(timezone.utc)
        if not self.context_data:
            self.context_data = {}
        self.context_data["end_reason"] = reason

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "conversation_id": str(self.conversation_id),
            "player_id": str(self.player_id),
            "npc_id": str(self.npc_id),
            "topic": self.topic,
            "mood": self.mood,
            "relationship_level": float(self.relationship_level),
            "context_data": self.context_data or {},
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_updated": (
                self.last_updated.isoformat() if self.last_updated else None
            ),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "exchange_count": self.get_exchange_count(),
        }


class ConversationExchange(Base):
    """SQLAlchemy model for individual conversation messages."""

    __tablename__ = "conversation_exchanges"

    exchange_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversation_contexts.conversation_id", ondelete="CASCADE"),
        nullable=False,
    )
    speaker_id = Column(UUID(as_uuid=True), nullable=False)
    message_text = Column(Text, nullable=False)
    message_type = Column(String(20), nullable=False)
    emotion = Column(String(50), nullable=True)
    timestamp = Column(
        DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()
    )
    exchange_metadata = Column(JSONB, nullable=False, default=dict)

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "message_type IN ('greeting', 'question', 'statement', 'farewell', 'system')",
            name="chk_message_type",
        ),
    )

    # Relationships
    conversation = relationship("ConversationContext", back_populates="exchanges")
    knowledge_source = relationship(
        "ConversationKnowledge", back_populates="source_exchange"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "exchange_id": str(self.exchange_id),
            "conversation_id": str(self.conversation_id),
            "speaker_id": str(self.speaker_id),
            "message_text": self.message_text,
            "message_type": self.message_type,
            "emotion": self.emotion,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.exchange_metadata or {},
        }


class ConversationKnowledge(Base):
    """SQLAlchemy model for knowledge extracted from conversations."""

    __tablename__ = "conversation_knowledge"

    knowledge_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversation_contexts.conversation_id", ondelete="CASCADE"),
        nullable=False,
    )
    information_type = Column(String(50), nullable=False)
    extracted_info = Column(JSONB, nullable=False)
    confidence_score = Column(DECIMAL(3, 2), nullable=True)
    source_exchange_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversation_exchanges.exchange_id"),
        nullable=True,
    )
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "confidence_score >= 0.0 AND confidence_score <= 1.0",
            name="chk_confidence_score",
        ),
    )

    # Relationships
    conversation = relationship(
        "ConversationContext", back_populates="knowledge_entries"
    )
    source_exchange = relationship(
        "ConversationExchange", back_populates="knowledge_source"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "knowledge_id": str(self.knowledge_id),
            "conversation_id": str(self.conversation_id),
            "information_type": self.information_type,
            "extracted_info": self.extracted_info or {},
            "confidence_score": (
                float(self.confidence_score) if self.confidence_score else None
            ),
            "source_exchange_id": (
                str(self.source_exchange_id) if self.source_exchange_id else None
            ),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
