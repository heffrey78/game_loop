"""
Conversation threading service for topic continuity and memory persistence.

This module provides the main service for managing conversation threads that
span multiple game sessions, ensuring NPCs maintain contextual awareness and
natural topic continuity.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from game_loop.database.repositories.conversation_threading import (
    ConversationThreadingManager,
    ConversationThread,
    PlayerMemoryProfile,
    TopicEvolution,
)
from game_loop.database.session_factory import DatabaseSessionFactory

from .conversation_models import ConversationContext, NPCPersonality
from .memory_integration import MemoryContext, MemoryIntegrationInterface

logger = logging.getLogger(__name__)


@dataclass
class ThreadingContext:
    """Context for conversation threading operations."""

    active_thread: ConversationThread | None = None
    player_profile: PlayerMemoryProfile | None = None
    previous_sessions: list[dict[str, Any]] = field(default_factory=list)
    conversation_hooks: list[dict[str, Any]] = field(default_factory=list)
    topic_continuity_score: float = 0.5
    relationship_progression: list[dict[str, Any]] = field(default_factory=list)
    memory_references: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ThreadingAnalysis:
    """Analysis result for conversation threading decision-making."""

    should_reference_past: bool = False
    reference_confidence: float = 0.0
    suggested_references: list[str] = field(default_factory=list)
    topic_evolution_quality: str = "unknown"
    relationship_context: dict[str, Any] = field(default_factory=dict)
    threading_recommendations: list[str] = field(default_factory=list)


class ConversationThreadingService:
    """
    Service for managing conversation threads and topic continuity.

    This service ensures NPCs maintain memory of past conversations
    and can naturally reference previous topics and build relationships
    across multiple game sessions.
    """

    def __init__(
        self,
        session_factory: DatabaseSessionFactory,
        memory_integration: MemoryIntegrationInterface,
        enable_threading: bool = True,
        reference_probability_threshold: float = 0.7,
        max_references_per_response: int = 2,
    ):
        self.session_factory = session_factory
        self.memory_integration = memory_integration
        self.enable_threading = enable_threading
        self.reference_probability_threshold = reference_probability_threshold
        self.max_references_per_response = max_references_per_response

        # Cache for threading contexts to improve performance
        self._threading_contexts: dict[str, ThreadingContext] = {}

    async def initiate_conversation_session(
        self,
        player_id: uuid.UUID,
        npc_id: uuid.UUID,
        conversation: ConversationContext,
        initial_topic: str,
    ) -> ThreadingContext:
        """
        Initiate a conversation session with threading context.

        This is called at the start of each conversation to establish
        the threading context and prepare memory references.
        """
        if not self.enable_threading:
            return ThreadingContext()

        try:
            async with self.session_factory.get_session() as session:
                threading_manager = ConversationThreadingManager(session)

                # Initialize or continue conversation thread
                thread, profile = await threading_manager.initiate_conversation_session(
                    player_id=player_id,
                    npc_id=npc_id,
                    initial_topic=initial_topic,
                    conversation_id=uuid.UUID(conversation.conversation_id),
                )

                # Get comprehensive conversation context
                context_data = await threading_manager.get_conversation_context(
                    player_id=player_id,
                    npc_id=npc_id,
                    limit_memories=3,
                )

                # Create threading context
                threading_context = ThreadingContext(
                    active_thread=thread,
                    player_profile=profile,
                    conversation_hooks=context_data.get("conversation_hooks", []),
                    relationship_progression=context_data.get(
                        "relationship_summary", {}
                    ),
                )

                # Calculate topic continuity score
                threading_context.topic_continuity_score = (
                    await self._calculate_topic_continuity(
                        current_topic=initial_topic,
                        thread=thread,
                        profile=profile,
                    )
                )

                # Cache the context
                cache_key = f"{player_id}:{npc_id}"
                self._threading_contexts[cache_key] = threading_context

                logger.info(
                    f"Initiated threading session for player {player_id} with NPC {npc_id}, "
                    f"thread: {thread.thread_id}, continuity: {threading_context.topic_continuity_score:.2f}"
                )

                return threading_context

        except Exception as e:
            logger.error(f"Error initiating conversation session: {e}")
            return ThreadingContext()  # Return empty context on error

    async def analyze_threading_opportunity(
        self,
        conversation: ConversationContext,
        personality: NPCPersonality,
        player_message: str,
        current_topic: str,
    ) -> ThreadingAnalysis:
        """
        Analyze whether and how to reference past conversations.

        This determines if the NPC should make references to previous
        conversations and what type of references would be appropriate.
        """
        player_id = uuid.UUID(conversation.player_id)
        npc_id = uuid.UUID(conversation.npc_id)
        cache_key = f"{player_id}:{npc_id}"

        # Get threading context
        threading_context = self._threading_contexts.get(cache_key)
        if not threading_context:
            # Initialize if not already done
            threading_context = await self.initiate_conversation_session(
                player_id, npc_id, conversation, current_topic
            )

        analysis = ThreadingAnalysis()

        if not threading_context.active_thread:
            return analysis

        try:
            # Analyze relationship context
            analysis.relationship_context = {
                "level": (
                    float(threading_context.player_profile.relationship_level)
                    if threading_context.player_profile
                    else 0.0
                ),
                "trust": (
                    float(threading_context.player_profile.trust_level)
                    if threading_context.player_profile
                    else 0.0
                ),
                "interaction_history": threading_context.active_thread.session_count,
                "conversation_style": (
                    threading_context.player_profile.conversation_style
                    if threading_context.player_profile
                    else "formal"
                ),
            }

            # Determine if we should reference past conversations
            reference_probability = await self._calculate_reference_probability(
                threading_context=threading_context,
                current_topic=current_topic,
                player_message=player_message,
                personality=personality,
            )

            analysis.should_reference_past = (
                reference_probability >= self.reference_probability_threshold
            )
            analysis.reference_confidence = reference_probability

            # Generate suggested references if appropriate
            if analysis.should_reference_past:
                analysis.suggested_references = await self._generate_memory_references(
                    threading_context=threading_context,
                    current_topic=current_topic,
                    max_references=self.max_references_per_response,
                )

            # Analyze topic evolution quality
            if threading_context.active_thread.topic_evolution:
                recent_evolutions = (
                    threading_context.active_thread.get_recent_progressions(3)
                )
                if recent_evolutions:
                    # Analyze quality of recent topic progressions
                    quality_scores = []
                    for evolution in recent_evolutions:
                        confidence = evolution.get("confidence", 0.5)
                        quality_scores.append(confidence)

                    avg_quality = sum(quality_scores) / len(quality_scores)
                    analysis.topic_evolution_quality = (
                        "excellent"
                        if avg_quality > 0.8
                        else (
                            "good"
                            if avg_quality > 0.6
                            else "fair" if avg_quality > 0.4 else "poor"
                        )
                    )

            # Generate threading recommendations
            analysis.threading_recommendations = (
                self._generate_threading_recommendations(threading_context, analysis)
            )

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing threading opportunity: {e}")
            return analysis

    async def enhance_response_with_threading(
        self,
        base_response: str,
        threading_analysis: ThreadingAnalysis,
        conversation: ConversationContext,
        personality: NPCPersonality,
        current_topic: str,
    ) -> tuple[str, dict[str, Any]]:
        """
        Enhance NPC response with conversation threading references.

        This adds natural references to past conversations based on the
        threading analysis, improving continuity and relationship building.
        """
        if not threading_analysis.should_reference_past:
            return base_response, {"threading_enhanced": False}

        enhanced_response = base_response
        threading_data = {
            "threading_enhanced": True,
            "references_added": 0,
            "reference_confidence": threading_analysis.reference_confidence,
            "references": [],
        }

        try:
            # Add memory references to the response
            if threading_analysis.suggested_references:
                references_to_add = threading_analysis.suggested_references[
                    : self.max_references_per_response
                ]

                for reference in references_to_add:
                    # Choose appropriate integration style based on relationship
                    relationship_level = threading_analysis.relationship_context.get(
                        "level", 0.0
                    )

                    if relationship_level >= 0.7:
                        # Close relationship - direct references
                        reference_phrase = f"Remember when {reference}? "
                    elif relationship_level >= 0.4:
                        # Moderate relationship - indirect references
                        reference_phrase = f"That reminds me of something we discussed before... {reference}. "
                    else:
                        # Distant relationship - subtle hints
                        reference_phrase = f"Something about that sounds familiar... "

                    # Integrate reference naturally into response
                    enhanced_response = self._integrate_reference_naturally(
                        enhanced_response, reference_phrase, personality
                    )

                    threading_data["references_added"] += 1
                    threading_data["references"].append(
                        {
                            "reference": reference,
                            "integration_style": (
                                "direct"
                                if relationship_level >= 0.7
                                else (
                                    "indirect"
                                    if relationship_level >= 0.4
                                    else "subtle"
                                )
                            ),
                        }
                    )

            logger.debug(
                f"Enhanced response with {threading_data['references_added']} threading references"
            )

            return enhanced_response, threading_data

        except Exception as e:
            logger.error(f"Error enhancing response with threading: {e}")
            return base_response, {"threading_enhanced": False, "error": str(e)}

    async def record_conversation_evolution(
        self,
        conversation: ConversationContext,
        previous_topic: str,
        new_topic: str,
        player_initiated: bool = False,
        evolution_quality: str = "natural",
    ) -> None:
        """
        Record topic evolution within a conversation thread.

        This tracks how topics change during conversations to improve
        future topic continuity and threading decisions.
        """
        if not self.enable_threading:
            return

        player_id = uuid.UUID(conversation.player_id)
        npc_id = uuid.UUID(conversation.npc_id)
        cache_key = f"{player_id}:{npc_id}"

        threading_context = self._threading_contexts.get(cache_key)
        if not threading_context or not threading_context.active_thread:
            return

        try:
            async with self.session_factory.get_session() as session:
                threading_manager = ConversationThreadingManager(session)

                # Record the topic evolution
                await threading_manager.evolutions.record_evolution(
                    thread_id=threading_context.active_thread.thread_id,
                    session_id=uuid.UUID(conversation.conversation_id),
                    source_topic=previous_topic,
                    target_topic=new_topic,
                    transition_type=(
                        "natural" if not player_initiated else "player_driven"
                    ),
                    player_initiated=player_initiated,
                    confidence_score=0.8 if evolution_quality == "natural" else 0.6,
                    evolution_quality=evolution_quality,
                    emotional_context=conversation.mood,
                )

                # Update thread with topic progression
                threading_context.active_thread.add_topic_progression(
                    from_topic=previous_topic,
                    to_topic=new_topic,
                    reason=(
                        "Natural conversation flow"
                        if not player_initiated
                        else "Player interest"
                    ),
                    confidence=0.8,
                )

                logger.debug(
                    f"Recorded topic evolution: {previous_topic} -> {new_topic} "
                    f"(player_initiated: {player_initiated})"
                )

        except Exception as e:
            logger.error(f"Error recording conversation evolution: {e}")

    async def finalize_conversation_session(
        self,
        conversation: ConversationContext,
        session_success: bool = True,
        relationship_change: float = 0.0,
        memorable_moments: list[str] = None,
    ) -> None:
        """
        Finalize a conversation session and update threading state.

        This is called when a conversation ends to update relationship
        progress and prepare for future conversations.
        """
        if not self.enable_threading:
            return

        player_id = uuid.UUID(conversation.player_id)
        npc_id = uuid.UUID(conversation.npc_id)
        cache_key = f"{player_id}:{npc_id}"

        threading_context = self._threading_contexts.get(cache_key)
        if not threading_context:
            return

        try:
            async with self.session_factory.get_session() as session:
                threading_manager = ConversationThreadingManager(session)

                # Update player profile
                if threading_context.player_profile:
                    await threading_manager.profiles.update_relationship(
                        player_id=player_id,
                        npc_id=npc_id,
                        relationship_change=relationship_change,
                        reason=f"Conversation session ({'successful' if session_success else 'unsuccessful'})",
                    )

                    # Record memorable moments
                    if memorable_moments:
                        for moment in memorable_moments:
                            await threading_manager.profiles.record_memorable_moment(
                                player_id=player_id,
                                npc_id=npc_id,
                                description=moment,
                                importance=0.8 if session_success else 0.5,
                                emotions=[conversation.mood],
                            )

                # Update thread importance based on session success
                if threading_context.active_thread:
                    importance_change = 0.1 if session_success else -0.05
                    await threading_manager.threads.update_thread_activity(
                        thread_id=threading_context.active_thread.thread_id,
                        session_id=uuid.UUID(conversation.conversation_id),
                        importance_change=importance_change,
                    )

                logger.info(
                    f"Finalized conversation session for player {player_id} with NPC {npc_id}, "
                    f"success: {session_success}, relationship_change: {relationship_change}"
                )

        except Exception as e:
            logger.error(f"Error finalizing conversation session: {e}")
        finally:
            # Clear cached context
            self._threading_contexts.pop(cache_key, None)

    async def get_conversation_preparation(
        self,
        player_id: uuid.UUID,
        npc_id: uuid.UUID,
        suggested_topics: list[str] = None,
    ) -> dict[str, Any]:
        """
        Get conversation preparation data for NPC interactions.

        This provides NPCs with context about past conversations
        and suggestions for natural conversation starters.
        """
        try:
            async with self.session_factory.get_session() as session:
                threading_manager = ConversationThreadingManager(session)

                # Get conversation hooks and context
                context = await threading_manager.get_conversation_context(
                    player_id=player_id,
                    npc_id=npc_id,
                    limit_memories=5,
                )

                # Get threading patterns analysis
                patterns = await threading_manager.threads.analyze_thread_patterns(
                    player_id=player_id,
                    npc_id=npc_id,
                )

                preparation = {
                    "conversation_context": context,
                    "threading_patterns": patterns,
                    "suggested_openers": await self._generate_conversation_openers(
                        context, patterns, suggested_topics or []
                    ),
                    "relationship_notes": context.get("relationship_summary", {}),
                    "topic_preferences": patterns.get("common_topics", [])[:5],
                }

                return preparation

        except Exception as e:
            logger.error(f"Error getting conversation preparation: {e}")
            return {"error": str(e)}

    # Private helper methods

    async def _calculate_topic_continuity(
        self,
        current_topic: str,
        thread: ConversationThread,
        profile: PlayerMemoryProfile,
    ) -> float:
        """Calculate topic continuity score for current context."""
        if not thread or not thread.topic_evolution:
            return 0.5

        # Check if current topic relates to recent progressions
        recent_progressions = thread.get_recent_progressions(5)
        topic_matches = 0

        for progression in recent_progressions:
            if (
                progression.get("to_topic") == current_topic
                or progression.get("from_topic") == current_topic
            ):
                topic_matches += 1

        # Base score on topic matches and thread activity
        continuity_score = min(1.0, (topic_matches / 3) + 0.3)

        # Adjust based on relationship level
        if profile:
            relationship_bonus = float(profile.relationship_level) * 0.2
            continuity_score = min(1.0, continuity_score + relationship_bonus)

        return continuity_score

    async def _calculate_reference_probability(
        self,
        threading_context: ThreadingContext,
        current_topic: str,
        player_message: str,
        personality: NPCPersonality,
    ) -> float:
        """Calculate probability of making memory references."""
        base_probability = 0.3

        # Increase probability based on relationship
        if threading_context.player_profile:
            relationship_factor = (
                float(threading_context.player_profile.relationship_level) * 0.3
            )
            base_probability += relationship_factor

        # Increase based on topic continuity
        continuity_factor = threading_context.topic_continuity_score * 0.2
        base_probability += continuity_factor

        # Increase based on thread session count (more history = more references)
        if threading_context.active_thread:
            history_factor = min(
                0.2, threading_context.active_thread.session_count * 0.05
            )
            base_probability += history_factor

        # Personality modifiers
        if personality:
            # Talkative NPCs reference more
            if personality.get_trait_strength("talkative") > 0.6:
                base_probability += 0.15

            # Wise NPCs reference past conversations more
            if personality.get_trait_strength("wise") > 0.6:
                base_probability += 0.1

            # Reserved NPCs reference less
            if personality.get_trait_strength("reserved") > 0.6:
                base_probability -= 0.1

        return max(0.0, min(1.0, base_probability))

    async def _generate_memory_references(
        self,
        threading_context: ThreadingContext,
        current_topic: str,
        max_references: int,
    ) -> list[str]:
        """Generate specific memory references for the response."""
        references = []

        if not threading_context.active_thread:
            return references

        # Use conversation hooks first
        for hook in threading_context.conversation_hooks[:max_references]:
            hook_text = hook.get("hook", "")
            if hook_text and len(references) < max_references:
                references.append(f"we talked about {hook_text}")

        # Add references from topic progressions
        if len(references) < max_references:
            recent_progressions = (
                threading_context.active_thread.get_recent_progressions(3)
            )
            for progression in recent_progressions:
                if len(references) >= max_references:
                    break

                from_topic = progression.get("from_topic", "")
                reason = progression.get("reason", "")

                if from_topic and current_topic.lower() != from_topic.lower():
                    ref_text = f"you mentioned {from_topic}"
                    if reason and "evolution" not in reason.lower():
                        ref_text += f" in relation to {reason}"
                    references.append(ref_text)

        return references[:max_references]

    def _integrate_reference_naturally(
        self,
        base_response: str,
        reference_phrase: str,
        personality: NPCPersonality,
    ) -> str:
        """Integrate memory reference naturally into the response."""
        # Simple integration at the beginning for now
        # Could be enhanced with more sophisticated NLP integration

        if personality and personality.get_trait_strength("formal") > 0.6:
            # Formal integration
            return f"{reference_phrase}As I was saying, {base_response.lower()}"
        else:
            # Casual integration
            return f"{reference_phrase}{base_response}"

    def _generate_threading_recommendations(
        self,
        threading_context: ThreadingContext,
        analysis: ThreadingAnalysis,
    ) -> list[str]:
        """Generate recommendations for improving threading."""
        recommendations = []

        if threading_context.topic_continuity_score < 0.4:
            recommendations.append("Consider more explicit topic transitions")

        if analysis.relationship_context.get("level", 0.0) < 0.3:
            recommendations.append(
                "Focus on relationship building before deep references"
            )

        if (
            threading_context.active_thread
            and threading_context.active_thread.session_count < 3
        ):
            recommendations.append(
                "Build more conversation history before complex threading"
            )

        return recommendations

    async def _generate_conversation_openers(
        self,
        context: dict[str, Any],
        patterns: dict[str, Any],
        suggested_topics: list[str],
    ) -> list[str]:
        """Generate natural conversation opening suggestions."""
        openers = []

        # Use conversation hooks
        hooks = context.get("conversation_hooks", [])
        for hook in hooks[:2]:
            hook_text = hook.get("hook", "")
            if hook_text:
                openers.append(
                    f"I've been thinking about what you said regarding {hook_text}"
                )

        # Use common topics from patterns
        common_topics = patterns.get("common_topics", [])
        for topic_data in common_topics[:2]:
            topic = topic_data.get("topic", "")
            if topic:
                openers.append(f"How have things been with {topic}?")

        # Use suggested topics
        for topic in suggested_topics[:2]:
            openers.append(f"I wanted to ask you about {topic}")

        # Fallback openers based on relationship
        relationship = context.get("relationship_summary", {})
        if relationship.get("level", 0.0) > 0.5:
            openers.append("It's good to see you again")
        else:
            openers.append("Hello there")

        return openers[:5]  # Limit to 5 suggestions
