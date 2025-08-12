"""Repository layer for conversation threading and topic continuity operations."""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import and_, desc, func, or_, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.sql import select

from ..models.conversation import ConversationContext
from ..models.conversation_threading import (
    ConversationThread,
    PlayerMemoryProfile,
    TopicEvolution,
)
from .base import BaseRepository


class ConversationThreadRepository(BaseRepository[ConversationThread]):
    """Repository for managing conversation threads and topic continuity."""

    def __init__(self, session: AsyncSession):
        super().__init__(ConversationThread, session)

    async def get_active_thread_for_player_npc(
        self, player_id: uuid.UUID, npc_id: uuid.UUID, topic: str = None
    ) -> ConversationThread | None:
        """Get the most relevant active thread for a player-NPC pair."""
        stmt = (
            select(ConversationThread)
            .options(selectinload(ConversationThread.topic_evolutions))
            .where(
                and_(
                    ConversationThread.player_id == player_id,
                    ConversationThread.npc_id == npc_id,
                    ConversationThread.thread_status.in_(["active", "dormant"]),
                )
            )
        )

        if topic:
            # Prioritize threads matching the current topic
            stmt = stmt.order_by(
                desc(
                    func.similarity(ConversationThread.primary_topic, topic)
                ).nulls_last(),
                desc(ConversationThread.last_referenced),
                desc(ConversationThread.importance_score),
            )
        else:
            # Order by most recent and important
            stmt = stmt.order_by(
                desc(ConversationThread.last_referenced),
                desc(ConversationThread.importance_score),
            )

        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def create_thread(
        self,
        player_id: uuid.UUID,
        npc_id: uuid.UUID,
        primary_topic: str,
        thread_title: str = None,
        importance_score: float = 0.5,
        priority_level: str = "normal",
    ) -> ConversationThread:
        """Create a new conversation thread."""
        thread = ConversationThread(
            player_id=player_id,
            npc_id=npc_id,
            primary_topic=primary_topic,
            thread_title=thread_title or f"Conversation about {primary_topic}",
            importance_score=importance_score,
            priority_level=priority_level,
            thread_status="active",
        )

        return await self.create(thread)

    async def get_player_threads(
        self,
        player_id: uuid.UUID,
        npc_id: uuid.UUID = None,
        status: str = None,
        limit: int = 50,
    ) -> list[ConversationThread]:
        """Get threads for a player, optionally filtered by NPC and status."""
        stmt = (
            select(ConversationThread)
            .options(selectinload(ConversationThread.topic_evolutions))
            .where(ConversationThread.player_id == player_id)
        )

        if npc_id:
            stmt = stmt.where(ConversationThread.npc_id == npc_id)

        if status:
            stmt = stmt.where(ConversationThread.thread_status == status)

        stmt = stmt.order_by(
            desc(ConversationThread.last_referenced),
            desc(ConversationThread.importance_score),
        ).limit(limit)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def find_related_threads(
        self,
        player_id: uuid.UUID,
        npc_id: uuid.UUID,
        topic: str,
        exclude_thread_id: uuid.UUID = None,
        limit: int = 10,
    ) -> list[ConversationThread]:
        """Find threads related to a specific topic."""
        stmt = select(ConversationThread).where(
            and_(
                ConversationThread.player_id == player_id,
                ConversationThread.npc_id == npc_id,
                or_(
                    ConversationThread.primary_topic.ilike(f"%{topic}%"),
                    ConversationThread.subtopics.op("&&")([topic]),
                    func.array_to_string(
                        ConversationThread.next_conversation_hooks, " "
                    ).ilike(f"%{topic}%"),
                ),
            )
        )

        if exclude_thread_id:
            stmt = stmt.where(ConversationThread.thread_id != exclude_thread_id)

        stmt = stmt.order_by(
            desc(ConversationThread.importance_score),
            desc(ConversationThread.last_referenced),
        ).limit(limit)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def update_thread_activity(
        self,
        thread_id: uuid.UUID,
        session_id: uuid.UUID,
        new_topic: str = None,
        importance_change: float = 0.0,
    ) -> ConversationThread | None:
        """Update thread activity after a conversation session."""
        thread = await self.get_by_id(thread_id)
        if not thread:
            return None

        thread.update_activity(session_id)

        # Update importance if specified
        if importance_change != 0.0:
            new_importance = max(
                0.0, min(1.0, thread.importance_score + importance_change)
            )
            thread.importance_score = new_importance

        # Update primary topic if it has evolved significantly
        if new_topic and new_topic != thread.primary_topic:
            # Add topic progression
            thread.add_topic_progression(
                from_topic=thread.primary_topic,
                to_topic=new_topic,
                reason="Natural conversation evolution",
                confidence=0.7,
            )
            thread.primary_topic = new_topic

        await self.session.commit()
        return thread

    async def mark_dormant_threads(
        self, days_inactive: int = 30, batch_size: int = 100
    ) -> int:
        """Mark old threads as dormant if not referenced recently."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_inactive)

        stmt = (
            update(ConversationThread)
            .where(
                and_(
                    ConversationThread.thread_status == "active",
                    ConversationThread.last_referenced < cutoff_date,
                )
            )
            .values(
                thread_status="dormant",
                dormant_since=func.current_timestamp(),
                last_updated=func.current_timestamp(),
            )
        )

        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount

    async def get_conversation_hooks(
        self,
        player_id: uuid.UUID,
        npc_id: uuid.UUID,
        priority_level: str = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Get conversation hooks for upcoming interactions."""
        stmt = select(ConversationThread).where(
            and_(
                ConversationThread.player_id == player_id,
                ConversationThread.npc_id == npc_id,
                ConversationThread.thread_status.in_(["active", "dormant"]),
                func.array_length(ConversationThread.next_conversation_hooks, 1) > 0,
            )
        )

        if priority_level:
            stmt = stmt.where(ConversationThread.priority_level == priority_level)

        stmt = stmt.order_by(
            desc(ConversationThread.importance_score),
            desc(ConversationThread.last_referenced),
        ).limit(limit)

        result = await self.session.execute(stmt)
        threads = result.scalars().all()

        hooks = []
        for thread in threads:
            if thread.next_conversation_hooks:
                for hook in thread.next_conversation_hooks:
                    hooks.append(
                        {
                            "thread_id": str(thread.thread_id),
                            "hook": hook,
                            "topic": thread.primary_topic,
                            "importance": float(thread.importance_score),
                            "priority": thread.priority_level,
                            "last_referenced": thread.last_referenced,
                        }
                    )

        return sorted(hooks, key=lambda x: x["importance"], reverse=True)[:limit]

    async def analyze_thread_patterns(
        self, player_id: uuid.UUID, npc_id: uuid.UUID
    ) -> dict[str, Any]:
        """Analyze conversation patterns for a player-NPC pair."""
        stmt = (
            select(ConversationThread)
            .options(selectinload(ConversationThread.topic_evolutions))
            .where(
                and_(
                    ConversationThread.player_id == player_id,
                    ConversationThread.npc_id == npc_id,
                )
            )
        )

        result = await self.session.execute(stmt)
        threads = list(result.scalars().all())

        if not threads:
            return {
                "total_threads": 0,
                "active_threads": 0,
                "common_topics": [],
                "relationship_progression": [],
                "engagement_level": "unknown",
            }

        # Analyze patterns
        total_threads = len(threads)
        active_threads = len([t for t in threads if t.thread_status == "active"])
        all_topics = []
        relationship_points = []

        for thread in threads:
            all_topics.extend(thread.subtopics or [])
            all_topics.append(thread.primary_topic)

            # Extract relationship progression
            if thread.trust_progression:
                for milestone in thread.trust_progression:
                    relationship_points.append(
                        {
                            "timestamp": milestone.get("timestamp"),
                            "level": milestone.get("new_level", 0.0),
                            "event": milestone.get("event", ""),
                        }
                    )

        # Find most common topics
        topic_counts = {}
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

        common_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]

        # Determine engagement level
        avg_sessions = sum(t.session_count for t in threads) / total_threads
        engagement_level = (
            "high" if avg_sessions > 5 else "medium" if avg_sessions > 2 else "low"
        )

        return {
            "total_threads": total_threads,
            "active_threads": active_threads,
            "dormant_threads": len(
                [t for t in threads if t.thread_status == "dormant"]
            ),
            "common_topics": [
                {"topic": topic, "frequency": count} for topic, count in common_topics
            ],
            "relationship_progression": sorted(
                relationship_points, key=lambda x: x.get("timestamp", "")
            ),
            "engagement_level": engagement_level,
            "avg_sessions_per_thread": avg_sessions,
            "total_sessions": sum(t.session_count for t in threads),
        }


class PlayerMemoryProfileRepository(BaseRepository[PlayerMemoryProfile]):
    """Repository for managing player memory profiles from NPC perspective."""

    def __init__(self, session: AsyncSession):
        super().__init__(PlayerMemoryProfile, session)

    async def get_profile(
        self, player_id: uuid.UUID, npc_id: uuid.UUID
    ) -> PlayerMemoryProfile | None:
        """Get player memory profile for a specific NPC."""
        stmt = select(PlayerMemoryProfile).where(
            and_(
                PlayerMemoryProfile.player_id == player_id,
                PlayerMemoryProfile.npc_id == npc_id,
            )
        )

        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def create_or_update_profile(
        self,
        player_id: uuid.UUID,
        npc_id: uuid.UUID,
        **profile_data: Any,
    ) -> PlayerMemoryProfile:
        """Create new profile or update existing one."""
        profile = await self.get_profile(player_id, npc_id)

        if profile is None:
            # Create new profile
            profile = PlayerMemoryProfile(
                player_id=player_id, npc_id=npc_id, **profile_data
            )
            profile = await self.create(profile)
        else:
            # Update existing profile
            for key, value in profile_data.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)
            profile.last_updated = datetime.now(timezone.utc)
            await self.session.commit()

        return profile

    async def update_relationship(
        self,
        player_id: uuid.UUID,
        npc_id: uuid.UUID,
        relationship_change: float,
        trust_change: float = 0.0,
        reason: str = "",
    ) -> PlayerMemoryProfile | None:
        """Update relationship and trust levels."""
        profile = await self.get_profile(player_id, npc_id)
        if not profile:
            return None

        profile.update_relationship(relationship_change, reason)

        if trust_change != 0.0:
            new_trust = max(0.0, min(1.0, float(profile.trust_level) + trust_change))
            profile.trust_level = new_trust

        profile.last_interaction = datetime.now(timezone.utc)
        await self.session.commit()
        return profile

    async def add_player_trait(
        self,
        player_id: uuid.UUID,
        npc_id: uuid.UUID,
        trait: str,
        strength: float,
        confidence: float = 0.8,
    ) -> PlayerMemoryProfile | None:
        """Add or update a player trait as observed by NPC."""
        profile = await self.get_profile(player_id, npc_id)
        if not profile:
            return None

        if profile.player_traits is None:
            profile.player_traits = {}

        profile.player_traits[trait] = {
            "strength": strength,
            "confidence": confidence,
            "observed_at": datetime.now(timezone.utc).isoformat(),
        }

        profile.last_updated = datetime.now(timezone.utc)
        await self.session.commit()
        return profile

    async def record_memorable_moment(
        self,
        player_id: uuid.UUID,
        npc_id: uuid.UUID,
        description: str,
        importance: float,
        emotions: list[str],
    ) -> PlayerMemoryProfile | None:
        """Record a highly memorable interaction."""
        profile = await self.get_profile(player_id, npc_id)
        if not profile:
            return None

        profile.add_memorable_moment(description, importance, emotions)
        await self.session.commit()
        return profile

    async def get_profiles_for_npc(
        self, npc_id: uuid.UUID, min_interactions: int = 1
    ) -> list[PlayerMemoryProfile]:
        """Get all player profiles for an NPC."""
        stmt = (
            select(PlayerMemoryProfile)
            .where(
                and_(
                    PlayerMemoryProfile.npc_id == npc_id,
                    PlayerMemoryProfile.total_interactions >= min_interactions,
                )
            )
            .order_by(
                desc(PlayerMemoryProfile.relationship_level),
                desc(PlayerMemoryProfile.total_interactions),
            )
        )

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def analyze_player_from_npc_perspective(
        self, player_id: uuid.UUID, npc_id: uuid.UUID
    ) -> dict[str, Any]:
        """Get comprehensive analysis of player from NPC's perspective."""
        profile = await self.get_profile(player_id, npc_id)
        if not profile:
            return {"error": "No profile found"}

        analysis = {
            "relationship_summary": {
                "level": float(profile.relationship_level),
                "trust": float(profile.trust_level),
                "familiarity": float(profile.familiarity_score),
                "conversation_style": profile.conversation_style,
                "success_rate": profile.get_success_rate(),
            },
            "perceived_traits": profile.player_traits or {},
            "interests": profile.player_interests or [],
            "dislikes": profile.player_dislikes or [],
            "memorable_moments": profile.memorable_moments or [],
            "interaction_stats": {
                "total": profile.total_interactions,
                "successful": profile.successful_interactions,
                "last_interaction": (
                    profile.last_interaction.isoformat()
                    if profile.last_interaction
                    else None
                ),
            },
        }

        return analysis


class TopicEvolutionRepository(BaseRepository[TopicEvolution]):
    """Repository for managing topic evolution tracking."""

    def __init__(self, session: AsyncSession):
        super().__init__(TopicEvolution, session)

    async def record_evolution(
        self,
        thread_id: uuid.UUID,
        session_id: uuid.UUID,
        source_topic: str,
        target_topic: str,
        transition_type: str,
        player_initiated: bool = False,
        confidence_score: float = 0.5,
        **evolution_data: Any,
    ) -> TopicEvolution:
        """Record a topic evolution."""
        evolution = TopicEvolution(
            thread_id=thread_id,
            session_id=session_id,
            source_topic=source_topic,
            target_topic=target_topic,
            transition_type=transition_type,
            player_initiated=player_initiated,
            confidence_score=confidence_score,
            **evolution_data,
        )

        return await self.create(evolution)

    async def get_thread_evolution_history(
        self, thread_id: uuid.UUID, limit: int = 50
    ) -> list[TopicEvolution]:
        """Get topic evolution history for a thread."""
        stmt = (
            select(TopicEvolution)
            .where(TopicEvolution.thread_id == thread_id)
            .order_by(desc(TopicEvolution.evolved_at))
            .limit(limit)
        )

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def analyze_topic_patterns(
        self, player_id: uuid.UUID, npc_id: uuid.UUID, days: int = 90
    ) -> dict[str, Any]:
        """Analyze topic evolution patterns for a player-NPC pair."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        # Get evolutions through thread relationships
        stmt = (
            select(TopicEvolution)
            .join(
                ConversationThread,
                TopicEvolution.thread_id == ConversationThread.thread_id,
            )
            .where(
                and_(
                    ConversationThread.player_id == player_id,
                    ConversationThread.npc_id == npc_id,
                    TopicEvolution.evolved_at >= cutoff_date,
                )
            )
            .order_by(TopicEvolution.evolved_at)
        )

        result = await self.session.execute(stmt)
        evolutions = list(result.scalars().all())

        if not evolutions:
            return {
                "total_evolutions": 0,
                "common_transitions": [],
                "evolution_quality": "unknown",
                "player_initiative_rate": 0.0,
            }

        # Analyze patterns
        transitions = {}
        player_initiated_count = 0
        quality_scores = []

        for evolution in evolutions:
            transition = f"{evolution.source_topic} -> {evolution.target_topic}"
            transitions[transition] = transitions.get(transition, 0) + 1

            if evolution.player_initiated:
                player_initiated_count += 1

            if evolution.evolution_quality == "smooth":
                quality_scores.append(1.0)
            elif evolution.evolution_quality == "natural":
                quality_scores.append(0.8)
            elif evolution.evolution_quality == "awkward":
                quality_scores.append(0.4)
            else:
                quality_scores.append(0.6)

        common_transitions = sorted(
            transitions.items(), key=lambda x: x[1], reverse=True
        )[:10]
        avg_quality = (
            sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        )
        player_initiative_rate = player_initiated_count / len(evolutions)

        quality_rating = (
            "excellent"
            if avg_quality > 0.8
            else (
                "good" if avg_quality > 0.6 else "fair" if avg_quality > 0.4 else "poor"
            )
        )

        return {
            "total_evolutions": len(evolutions),
            "common_transitions": [
                {"transition": trans, "frequency": freq}
                for trans, freq in common_transitions
            ],
            "evolution_quality": quality_rating,
            "avg_quality_score": avg_quality,
            "player_initiative_rate": player_initiative_rate,
            "analysis_period_days": days,
        }


class ConversationThreadingManager:
    """High-level manager for all conversation threading operations."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.threads = ConversationThreadRepository(session)
        self.profiles = PlayerMemoryProfileRepository(session)
        self.evolutions = TopicEvolutionRepository(session)

    async def initiate_conversation_session(
        self,
        player_id: uuid.UUID,
        npc_id: uuid.UUID,
        initial_topic: str,
        conversation_id: uuid.UUID,
    ) -> tuple[ConversationThread, PlayerMemoryProfile]:
        """Initialize or continue a conversation session with threading."""
        # Get or create player profile
        profile = await self.profiles.get_profile(player_id, npc_id)
        if not profile:
            profile = await self.profiles.create_or_update_profile(
                player_id=player_id,
                npc_id=npc_id,
                conversation_style="formal",
                memory_accuracy=1.0,
                memory_importance=0.5,
            )

        # Get or create conversation thread
        thread = await self.threads.get_active_thread_for_player_npc(
            player_id, npc_id, initial_topic
        )

        if not thread:
            # Create new thread
            thread = await self.threads.create_thread(
                player_id=player_id,
                npc_id=npc_id,
                primary_topic=initial_topic,
                thread_title=f"Discussion about {initial_topic}",
                importance_score=0.5,
            )

        # Update thread activity
        await self.threads.update_thread_activity(thread.thread_id, conversation_id)

        # Update profile interaction stats
        profile.update_interaction_stats(successful=True)
        await self.session.commit()

        return thread, profile

    async def get_conversation_context(
        self, player_id: uuid.UUID, npc_id: uuid.UUID, limit_memories: int = 3
    ) -> dict[str, Any]:
        """Get comprehensive conversation context for memory reference."""
        # Get active threads
        threads = await self.threads.get_player_threads(
            player_id=player_id, npc_id=npc_id, status="active", limit=5
        )

        # Get player profile
        profile = await self.profiles.get_profile(player_id, npc_id)

        # Get conversation hooks
        hooks = await self.threads.get_conversation_hooks(
            player_id=player_id, npc_id=npc_id, limit=3
        )

        # Compile context
        context = {
            "active_threads": [
                {
                    "thread_id": str(t.thread_id),
                    "primary_topic": t.primary_topic,
                    "importance": float(t.importance_score),
                    "session_count": t.session_count,
                    "recent_progressions": t.get_recent_progressions(3),
                }
                for t in threads[:limit_memories]
            ],
            "player_profile": profile.to_dict() if profile else None,
            "conversation_hooks": hooks,
            "relationship_summary": {
                "level": float(profile.relationship_level) if profile else 0.0,
                "trust": float(profile.trust_level) if profile else 0.0,
                "style": profile.conversation_style if profile else "formal",
            },
        }

        return context
