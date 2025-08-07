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
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

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
    memory_config = relationship(
        "MemoryPersonalityConfig", back_populates="npc", uselist=False, cascade="all, delete-orphan"
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
    
    # Semantic memory fields
    confidence_score = Column(DECIMAL(3, 2), nullable=False, default=1.0)
    emotional_weight = Column(DECIMAL(3, 2), nullable=False, default=0.5)
    trust_level_required = Column(DECIMAL(3, 2), nullable=False, default=0.0)
    last_accessed = Column(
        DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()
    )
    access_count = Column(Integer, nullable=False, default=0)
    memory_embedding = Column(Vector(384), nullable=True)  # 384-dimensional embeddings

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "message_type IN ('greeting', 'question', 'statement', 'farewell', 'system')",
            name="chk_message_type",
        ),
        CheckConstraint(
            "confidence_score >= 0.0 AND confidence_score <= 1.0",
            name="chk_confidence_score",
        ),
        CheckConstraint(
            "emotional_weight >= 0.0 AND emotional_weight <= 1.0",
            name="chk_emotional_weight",
        ),
        CheckConstraint(
            "trust_level_required >= 0.0 AND trust_level_required <= 1.0",
            name="chk_trust_level_required",
        ),
    )

    # Relationships
    conversation = relationship("ConversationContext", back_populates="exchanges")
    knowledge_source = relationship(
        "ConversationKnowledge", back_populates="source_exchange"
    )
    memory_embedding_entry = relationship(
        "MemoryEmbedding", back_populates="exchange", uselist=False, cascade="all, delete-orphan"
    )
    emotional_context_entry = relationship(
        "EmotionalContext", back_populates="exchange", uselist=False, cascade="all, delete-orphan"
    )
    access_log = relationship(
        "MemoryAccessLog", back_populates="exchange", cascade="all, delete-orphan"
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
            "confidence_score": float(self.confidence_score) if self.confidence_score else None,
            "emotional_weight": float(self.emotional_weight) if self.emotional_weight else None,
            "trust_level_required": float(self.trust_level_required) if self.trust_level_required else None,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "access_count": self.access_count,
            "memory_embedding": self.memory_embedding if self.memory_embedding else None,
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


class MemoryEmbedding(Base):
    """SQLAlchemy model for conversation memory embeddings."""

    __tablename__ = "memory_embeddings"

    embedding_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    exchange_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversation_exchanges.exchange_id", ondelete="CASCADE"),
        nullable=False,
    )
    embedding = Column(Vector(384), nullable=False)
    embedding_model = Column(String(100), nullable=False, default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_metadata = Column(JSONB, nullable=False, default=dict)
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()
    )

    # Relationships
    exchange = relationship("ConversationExchange", back_populates="memory_embedding_entry")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "embedding_id": str(self.embedding_id),
            "exchange_id": str(self.exchange_id),
            "embedding": self.embedding if self.embedding else None,
            "embedding_model": self.embedding_model,
            "embedding_metadata": self.embedding_metadata or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class MemoryAccessLog(Base):
    """SQLAlchemy model for memory access tracking."""

    __tablename__ = "memory_access_log"

    access_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    exchange_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversation_exchanges.exchange_id", ondelete="CASCADE"),
        nullable=False,
    )
    accessed_by = Column(UUID(as_uuid=True), nullable=False)  # NPC ID who accessed the memory
    access_context = Column(JSONB, nullable=False, default=dict)
    confidence_at_access = Column(DECIMAL(3, 2), nullable=False)
    access_type = Column(String(50), nullable=False, default="retrieval")
    accessed_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "confidence_at_access >= 0.0 AND confidence_at_access <= 1.0",
            name="chk_access_confidence",
        ),
        CheckConstraint(
            "access_type IN ('retrieval', 'reference', 'update', 'decay_calculation')",
            name="chk_access_type",
        ),
    )

    # Relationships
    exchange = relationship("ConversationExchange", back_populates="access_log")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "access_id": str(self.access_id),
            "exchange_id": str(self.exchange_id),
            "accessed_by": str(self.accessed_by),
            "access_context": self.access_context or {},
            "confidence_at_access": float(self.confidence_at_access) if self.confidence_at_access else None,
            "access_type": self.access_type,
            "accessed_at": self.accessed_at.isoformat() if self.accessed_at else None,
        }


class MemoryPersonalityConfig(Base):
    """SQLAlchemy model for NPC memory personality configuration."""

    __tablename__ = "memory_personality_config"

    config_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    npc_id = Column(
        UUID(as_uuid=True),
        ForeignKey("npc_personalities.npc_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    decay_rate_modifier = Column(DECIMAL(3, 2), nullable=False, default=1.0)
    emotional_sensitivity = Column(DECIMAL(3, 2), nullable=False, default=1.0)
    detail_retention_strength = Column(DECIMAL(3, 2), nullable=False, default=0.5)
    name_retention_strength = Column(DECIMAL(3, 2), nullable=False, default=0.5)
    uncertainty_threshold = Column(DECIMAL(3, 2), nullable=False, default=0.3)
    max_memory_capacity = Column(Integer, nullable=False, default=10000)
    memory_clustering_preference = Column(DECIMAL(3, 2), nullable=False, default=0.5)
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "decay_rate_modifier > 0.0 AND decay_rate_modifier <= 5.0",
            name="chk_decay_rate_modifier",
        ),
        CheckConstraint(
            "emotional_sensitivity >= 0.0 AND emotional_sensitivity <= 2.0",
            name="chk_emotional_sensitivity",
        ),
        CheckConstraint(
            "detail_retention_strength >= 0.0 AND detail_retention_strength <= 1.0",
            name="chk_detail_retention",
        ),
        CheckConstraint(
            "name_retention_strength >= 0.0 AND name_retention_strength <= 1.0",
            name="chk_name_retention",
        ),
        CheckConstraint(
            "uncertainty_threshold >= 0.0 AND uncertainty_threshold <= 1.0",
            name="chk_uncertainty_threshold",
        ),
        CheckConstraint(
            "memory_clustering_preference >= 0.0 AND memory_clustering_preference <= 1.0",
            name="chk_memory_clustering",
        ),
    )

    # Relationships
    npc = relationship("NPCPersonality", back_populates="memory_config")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config_id": str(self.config_id),
            "npc_id": str(self.npc_id),
            "decay_rate_modifier": float(self.decay_rate_modifier) if self.decay_rate_modifier else None,
            "emotional_sensitivity": float(self.emotional_sensitivity) if self.emotional_sensitivity else None,
            "detail_retention_strength": float(self.detail_retention_strength) if self.detail_retention_strength else None,
            "name_retention_strength": float(self.name_retention_strength) if self.name_retention_strength else None,
            "uncertainty_threshold": float(self.uncertainty_threshold) if self.uncertainty_threshold else None,
            "max_memory_capacity": self.max_memory_capacity,
            "memory_clustering_preference": float(self.memory_clustering_preference) if self.memory_clustering_preference else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class EmotionalContext(Base):
    """SQLAlchemy model for emotional context of conversation exchanges."""

    __tablename__ = "emotional_context"

    context_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    exchange_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversation_exchanges.exchange_id", ondelete="CASCADE"),
        nullable=False,
    )
    sentiment_score = Column(DECIMAL(3, 2), nullable=False, default=0.0)
    emotional_keywords = Column(ARRAY(Text), nullable=False, default=list)
    participant_emotions = Column(JSONB, nullable=False, default=dict)
    emotional_intensity = Column(DECIMAL(3, 2), nullable=False, default=0.0)
    relationship_impact_score = Column(DECIMAL(3, 2), nullable=False, default=0.0)
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "sentiment_score >= -1.0 AND sentiment_score <= 1.0",
            name="chk_sentiment_score",
        ),
        CheckConstraint(
            "emotional_intensity >= 0.0 AND emotional_intensity <= 1.0",
            name="chk_emotional_intensity",
        ),
        CheckConstraint(
            "relationship_impact_score >= 0.0 AND relationship_impact_score <= 1.0",
            name="chk_relationship_impact",
        ),
    )

    # Relationships
    exchange = relationship("ConversationExchange", back_populates="emotional_context_entry")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "context_id": str(self.context_id),
            "exchange_id": str(self.exchange_id),
            "sentiment_score": float(self.sentiment_score) if self.sentiment_score else None,
            "emotional_keywords": self.emotional_keywords or [],
            "participant_emotions": self.participant_emotions or {},
            "emotional_intensity": float(self.emotional_intensity) if self.emotional_intensity else None,
            "relationship_impact_score": float(self.relationship_impact_score) if self.relationship_impact_score else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
