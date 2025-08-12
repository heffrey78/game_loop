"""SQLAlchemy models for conversation threading and topic continuity."""

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
    Boolean,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .base import Base


class ConversationThread(Base):
    """
    Persistent conversation threads that span multiple game sessions.

    Tracks ongoing conversation topics and their progression across
    multiple individual conversation sessions between a player and NPC.
    """

    __tablename__ = "conversation_threads"

    thread_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    player_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    npc_id = Column(
        UUID(as_uuid=True),
        ForeignKey("npc_personalities.npc_id"),
        nullable=False,
        index=True,
    )

    # Thread metadata
    primary_topic = Column(String(255), nullable=False, index=True)
    thread_title = Column(String(500), nullable=True)  # Human-readable thread title
    thread_status = Column(
        String(20), nullable=False, default="active"
    )  # active, dormant, concluded, abandoned

    # Topic progression
    topic_evolution = Column(
        JSONB, nullable=False, default=list
    )  # List of topic progressions
    subtopics = Column(ARRAY(String(255)), nullable=False, default=list)
    resolved_questions = Column(ARRAY(Text), nullable=False, default=list)
    pending_questions = Column(ARRAY(Text), nullable=False, default=list)

    # Relationship tracking
    trust_progression = Column(
        JSONB, nullable=False, default=list
    )  # Track trust over time
    emotional_arc = Column(
        JSONB, nullable=False, default=list
    )  # Track emotional progression
    relationship_milestones = Column(JSONB, nullable=False, default=list)

    # Session tracking
    session_count = Column(Integer, nullable=False, default=0)
    last_session_id = Column(UUID(as_uuid=True), nullable=True)
    next_conversation_hooks = Column(
        ARRAY(Text), nullable=False, default=list
    )  # Topics to bring up next

    # Importance and priority
    importance_score = Column(DECIMAL(3, 2), nullable=False, default=0.5)  # 0.0-1.0
    priority_level = Column(
        String(10), nullable=False, default="normal"
    )  # urgent, high, normal, low

    # Timing
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()
    )
    last_updated = Column(
        DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()
    )
    last_referenced = Column(DateTime(timezone=True), nullable=True)
    dormant_since = Column(
        DateTime(timezone=True), nullable=True
    )  # When thread went dormant

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "importance_score >= 0.0 AND importance_score <= 1.0",
            name="chk_thread_importance_score",
        ),
        CheckConstraint(
            "thread_status IN ('active', 'dormant', 'concluded', 'abandoned')",
            name="chk_thread_status",
        ),
        CheckConstraint(
            "priority_level IN ('urgent', 'high', 'normal', 'low')",
            name="chk_thread_priority",
        ),
        CheckConstraint(
            "session_count >= 0",
            name="chk_thread_session_count",
        ),
    )

    # Relationships
    npc_personality = relationship(
        "NPCPersonality", back_populates="conversation_threads"
    )
    conversation_sessions = relationship(
        "ConversationContext",
        back_populates="thread",
        cascade="all, delete-orphan",
        order_by="ConversationContext.started_at",
    )
    topic_evolutions = relationship(
        "TopicEvolution",
        back_populates="thread",
        cascade="all, delete-orphan",
        order_by="TopicEvolution.evolved_at",
    )

    def add_topic_progression(
        self, from_topic: str, to_topic: str, reason: str, confidence: float = 0.5
    ) -> None:
        """Add a topic progression to the evolution list."""
        progression = {
            "from_topic": from_topic,
            "to_topic": to_topic,
            "reason": reason,
            "confidence": confidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if self.topic_evolution is None:
            self.topic_evolution = []

        self.topic_evolution = self.topic_evolution + [progression]
        self.last_updated = datetime.now(timezone.utc)

    def add_trust_milestone(
        self, event: str, old_level: float, new_level: float
    ) -> None:
        """Add a trust progression milestone."""
        milestone = {
            "event": event,
            "old_level": old_level,
            "new_level": new_level,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if self.trust_progression is None:
            self.trust_progression = []

        self.trust_progression = self.trust_progression + [milestone]
        self.last_updated = datetime.now(timezone.utc)

    def update_activity(self, session_id: uuid.UUID) -> None:
        """Update thread activity with new session."""
        self.session_count += 1
        self.last_session_id = session_id
        self.last_referenced = datetime.now(timezone.utc)
        self.last_updated = datetime.now(timezone.utc)

        # Reactivate if dormant
        if self.thread_status == "dormant":
            self.thread_status = "active"
            self.dormant_since = None

    def mark_dormant(self) -> None:
        """Mark thread as dormant due to inactivity."""
        self.thread_status = "dormant"
        self.dormant_since = datetime.now(timezone.utc)
        self.last_updated = datetime.now(timezone.utc)

    def get_recent_progressions(self, limit: int = 5) -> list[dict]:
        """Get recent topic progressions."""
        if not self.topic_evolution:
            return []

        sorted_progressions = sorted(
            self.topic_evolution, key=lambda x: x.get("timestamp", ""), reverse=True
        )
        return sorted_progressions[:limit]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "thread_id": str(self.thread_id),
            "player_id": str(self.player_id),
            "npc_id": str(self.npc_id),
            "primary_topic": self.primary_topic,
            "thread_title": self.thread_title,
            "thread_status": self.thread_status,
            "topic_evolution": self.topic_evolution or [],
            "subtopics": self.subtopics or [],
            "session_count": self.session_count,
            "importance_score": (
                float(self.importance_score) if self.importance_score else 0.0
            ),
            "priority_level": self.priority_level,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_updated": (
                self.last_updated.isoformat() if self.last_updated else None
            ),
            "last_referenced": (
                self.last_referenced.isoformat() if self.last_referenced else None
            ),
        }


class PlayerMemoryProfile(Base):
    """
    NPC-specific memory profiles for individual players.

    Maintains persistent memory and relationship state that NPCs
    have about specific players across all interactions.
    """

    __tablename__ = "player_memory_profiles"

    profile_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    player_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    npc_id = Column(
        UUID(as_uuid=True),
        ForeignKey("npc_personalities.npc_id"),
        nullable=False,
        index=True,
    )

    # Player characteristics as remembered by NPC
    remembered_name = Column(String(255), nullable=True)  # What NPC calls the player
    player_traits = Column(
        JSONB, nullable=False, default=dict
    )  # Observed player traits
    player_interests = Column(ARRAY(String(255)), nullable=False, default=list)
    player_dislikes = Column(ARRAY(String(255)), nullable=False, default=list)
    player_goals = Column(ARRAY(Text), nullable=False, default=list)
    player_secrets = Column(
        ARRAY(Text), nullable=False, default=list
    )  # Secrets player confided

    # Relationship state
    relationship_level = Column(
        DECIMAL(3, 2), nullable=False, default=0.0
    )  # Current relationship
    trust_level = Column(
        DECIMAL(3, 2), nullable=False, default=0.0
    )  # How much NPC trusts player
    familiarity_score = Column(
        DECIMAL(3, 2), nullable=False, default=0.0
    )  # How well NPC knows player

    # Interaction patterns
    conversation_style = Column(
        String(50), nullable=False, default="formal"
    )  # formal, casual, intimate
    preferred_topics = Column(ARRAY(String(255)), nullable=False, default=list)
    avoided_topics = Column(ARRAY(String(255)), nullable=False, default=list)
    shared_experiences = Column(JSONB, nullable=False, default=list)

    # Memory quality and importance
    memory_accuracy = Column(
        DECIMAL(3, 2), nullable=False, default=1.0
    )  # How accurate memories are
    memory_importance = Column(
        DECIMAL(3, 2), nullable=False, default=0.5
    )  # How important player is to NPC
    last_memory_quality = Column(
        String(20), nullable=False, default="clear"
    )  # clear, hazy, confused

    # Emotional context
    last_interaction_mood = Column(String(50), nullable=False, default="neutral")
    emotional_associations = Column(
        JSONB, nullable=False, default=dict
    )  # Emotions associated with player
    sentiment_history = Column(
        JSONB, nullable=False, default=list
    )  # Sentiment over time

    # Interaction tracking
    total_interactions = Column(Integer, nullable=False, default=0)
    successful_interactions = Column(Integer, nullable=False, default=0)
    memorable_moments = Column(
        JSONB, nullable=False, default=list
    )  # Highly memorable interactions

    # Timing and updates
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()
    )
    last_updated = Column(
        DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()
    )
    last_interaction = Column(DateTime(timezone=True), nullable=True)

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "relationship_level >= -1.0 AND relationship_level <= 1.0",
            name="chk_profile_relationship_level",
        ),
        CheckConstraint(
            "trust_level >= 0.0 AND trust_level <= 1.0",
            name="chk_profile_trust_level",
        ),
        CheckConstraint(
            "familiarity_score >= 0.0 AND familiarity_score <= 1.0",
            name="chk_profile_familiarity_score",
        ),
        CheckConstraint(
            "memory_accuracy >= 0.0 AND memory_accuracy <= 1.0",
            name="chk_profile_memory_accuracy",
        ),
        CheckConstraint(
            "memory_importance >= 0.0 AND memory_importance <= 1.0",
            name="chk_profile_memory_importance",
        ),
        CheckConstraint(
            "conversation_style IN ('formal', 'casual', 'intimate', 'professional', 'friendly')",
            name="chk_profile_conversation_style",
        ),
        CheckConstraint(
            "last_memory_quality IN ('clear', 'hazy', 'confused', 'fragmented', 'vivid')",
            name="chk_profile_memory_quality",
        ),
    )

    # Relationships
    npc_personality = relationship("NPCPersonality", back_populates="player_profiles")

    def update_relationship(self, change: float, reason: str = "") -> None:
        """Update relationship level with change tracking."""
        old_level = float(self.relationship_level)
        new_level = max(-1.0, min(1.0, old_level + change))
        self.relationship_level = new_level

        # Add to sentiment history
        sentiment_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "old_relationship": old_level,
            "new_relationship": new_level,
            "change": change,
            "reason": reason,
        }

        if self.sentiment_history is None:
            self.sentiment_history = []

        self.sentiment_history = self.sentiment_history + [sentiment_entry]
        self.last_updated = datetime.now(timezone.utc)

    def add_memorable_moment(
        self, description: str, importance: float, emotions: list[str]
    ) -> None:
        """Add a highly memorable interaction."""
        moment = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "description": description,
            "importance": importance,
            "emotions": emotions,
        }

        if self.memorable_moments is None:
            self.memorable_moments = []

        self.memorable_moments = self.memorable_moments + [moment]

        # Keep only the most important moments (max 20)
        self.memorable_moments = sorted(
            self.memorable_moments, key=lambda x: x.get("importance", 0), reverse=True
        )[:20]

        self.last_updated = datetime.now(timezone.utc)

    def update_interaction_stats(self, successful: bool = True) -> None:
        """Update interaction statistics."""
        self.total_interactions += 1
        if successful:
            self.successful_interactions += 1
        self.last_interaction = datetime.now(timezone.utc)
        self.last_updated = datetime.now(timezone.utc)

    def get_success_rate(self) -> float:
        """Get interaction success rate."""
        if self.total_interactions == 0:
            return 0.0
        return self.successful_interactions / self.total_interactions

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "profile_id": str(self.profile_id),
            "player_id": str(self.player_id),
            "npc_id": str(self.npc_id),
            "remembered_name": self.remembered_name,
            "player_traits": self.player_traits or {},
            "relationship_level": (
                float(self.relationship_level) if self.relationship_level else 0.0
            ),
            "trust_level": float(self.trust_level) if self.trust_level else 0.0,
            "conversation_style": self.conversation_style,
            "total_interactions": self.total_interactions,
            "success_rate": self.get_success_rate(),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_updated": (
                self.last_updated.isoformat() if self.last_updated else None
            ),
        }


class TopicEvolution(Base):
    """
    Track how conversation topics evolve and transform over time.

    Records the natural progression of topics within conversation
    threads, helping NPCs maintain contextual awareness.
    """

    __tablename__ = "topic_evolutions"

    evolution_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    thread_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversation_threads.thread_id"),
        nullable=False,
        index=True,
    )
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversation_contexts.conversation_id"),
        nullable=True,
    )

    # Topic transition
    source_topic = Column(String(255), nullable=False)
    target_topic = Column(String(255), nullable=False)
    transition_type = Column(
        String(20), nullable=False
    )  # natural, forced, interrupted, concluded

    # Context and reasoning
    transition_reason = Column(Text, nullable=True)  # Why topic changed
    player_initiated = Column(Boolean, nullable=False, default=False)
    confidence_score = Column(
        DECIMAL(3, 2), nullable=False, default=0.5
    )  # How confident in this evolution

    # Semantic information
    topic_keywords = Column(ARRAY(String(100)), nullable=False, default=list)
    emotional_context = Column(String(50), nullable=False, default="neutral")
    conversation_depth = Column(
        String(20), nullable=False, default="surface"
    )  # surface, moderate, deep

    # Relationship to other topics
    parent_topics = Column(
        ARRAY(String(255)), nullable=False, default=list
    )  # Topics that led here
    child_topics = Column(
        ARRAY(String(255)), nullable=False, default=list
    )  # Topics this may lead to
    related_topics = Column(
        ARRAY(String(255)), nullable=False, default=list
    )  # Semantically related

    # Quality and importance
    evolution_quality = Column(
        String(20), nullable=False, default="smooth"
    )  # smooth, awkward, natural, forced
    importance_to_relationship = Column(DECIMAL(3, 2), nullable=False, default=0.5)
    memory_formation_likelihood = Column(DECIMAL(3, 2), nullable=False, default=0.5)

    # Timing
    evolved_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "confidence_score >= 0.0 AND confidence_score <= 1.0",
            name="chk_evolution_confidence_score",
        ),
        CheckConstraint(
            "importance_to_relationship >= 0.0 AND importance_to_relationship <= 1.0",
            name="chk_evolution_importance",
        ),
        CheckConstraint(
            "memory_formation_likelihood >= 0.0 AND memory_formation_likelihood <= 1.0",
            name="chk_evolution_memory_likelihood",
        ),
        CheckConstraint(
            "transition_type IN ('natural', 'forced', 'interrupted', 'concluded', 'branched')",
            name="chk_evolution_transition_type",
        ),
        CheckConstraint(
            "conversation_depth IN ('surface', 'moderate', 'deep', 'intimate')",
            name="chk_evolution_depth",
        ),
        CheckConstraint(
            "evolution_quality IN ('smooth', 'awkward', 'natural', 'forced', 'seamless')",
            name="chk_evolution_quality",
        ),
    )

    # Relationships
    thread = relationship("ConversationThread", back_populates="topic_evolutions")
    session = relationship("ConversationContext", back_populates="topic_evolutions")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "evolution_id": str(self.evolution_id),
            "thread_id": str(self.thread_id),
            "source_topic": self.source_topic,
            "target_topic": self.target_topic,
            "transition_type": self.transition_type,
            "player_initiated": self.player_initiated,
            "confidence_score": (
                float(self.confidence_score) if self.confidence_score else 0.0
            ),
            "emotional_context": self.emotional_context,
            "conversation_depth": self.conversation_depth,
            "evolution_quality": self.evolution_quality,
            "evolved_at": self.evolved_at.isoformat() if self.evolved_at else None,
        }
