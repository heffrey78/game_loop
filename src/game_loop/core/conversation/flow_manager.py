"""Enhanced conversation flow manager with memory integration patterns."""

import logging
import uuid
from typing import Any

from game_loop.core.utils import (
    UUIDSecurityError,
    extract_player_id_from_conversation_id,
)
from game_loop.database.repositories.conversation import ConversationRepositoryManager
from game_loop.database.session_factory import DatabaseSessionFactory

from .conversation_models import (
    ConversationContext,
    ConversationExchange,
    NPCPersonality,
)
from .flow_templates import (
    ConversationFlowLibrary,
    ConversationStage,
    MemoryDisclosureThreshold,
    MemoryIntegrationPattern,
    TransitionPhrase,
    TrustLevel,
)
from .interfaces import ConversationMemoryInterface
from .conversation_threading import ConversationThreadingService, ThreadingAnalysis
from .context_engine import (
    DialogueMemoryIntegrationEngine,
    DialogueContext,
    DialogueEnhancementResult,
)

logger = logging.getLogger(__name__)


class ConversationFlowManager:
    """Manages conversation flow with memory integration patterns."""

    def __init__(
        self,
        memory_integration: ConversationMemoryInterface,
        session_factory: DatabaseSessionFactory,
        flow_library: ConversationFlowLibrary | None = None,
        enable_conversation_threading: bool = True,
    ):
        self.memory_integration = memory_integration
        self.session_factory = session_factory
        self.enable_conversation_threading = enable_conversation_threading

        if flow_library is None:
            from .flow_factory import create_default_flow_library

            self.flow_library = create_default_flow_library()
        else:
            self.flow_library = flow_library

        # Initialize conversation threading service
        if self.enable_conversation_threading:
            self.threading_service = ConversationThreadingService(
                session_factory=session_factory,
                memory_integration=memory_integration,
                enable_threading=True,
            )
        else:
            self.threading_service = None

        # Cache for conversation states
        self._conversation_stages: dict[str, ConversationStage] = {}
        self._memory_usage_history: dict[str, list[dict[str, Any]]] = {}

    async def enhance_response_with_memory_patterns(
        self,
        conversation: ConversationContext,
        personality: NPCPersonality,
        base_response: str,
        player_message: str,
        npc_id: uuid.UUID,
    ) -> tuple[str, dict[str, Any]]:
        """Enhance base response with memory integration patterns."""
        try:
            # Determine current conversation stage
            stage = await self._determine_conversation_stage(conversation, personality)

            # Get trust level
            trust_level = self.flow_library.get_trust_level_from_relationship(
                conversation.relationship_level
            )

            # Get memory context
            memory_context = await self.memory_integration.extract_memory_context(
                conversation, player_message, personality
            )

            # Retrieve relevant memories
            memory_result = await self.memory_integration.retrieve_relevant_memories(
                memory_context, npc_id
            )

            # Analyze conversation threading opportunities
            threading_analysis = None
            if self.threading_service:
                threading_analysis = (
                    await self.threading_service.analyze_threading_opportunity(
                        conversation=conversation,
                        personality=personality,
                        player_message=player_message,
                        current_topic=memory_context.current_topic or "general",
                    )
                )

            # Get appropriate flow template
            template = self.flow_library.get_template_for_stage(stage)
            if not template:
                logger.warning(f"No template found for stage {stage}")
                return base_response, {"stage": stage.value, "memory_enhanced": False}

            # Select memory pattern based on trust and confidence
            if memory_result.relevant_memories:
                enhanced_response, integration_data = (
                    await self._integrate_memories_with_patterns(
                        base_response,
                        memory_result.relevant_memories,
                        trust_level,
                        template,
                        personality,
                        memory_context.emotional_tone,
                    )
                )
            else:
                enhanced_response = base_response
                integration_data = {"memory_enhanced": False}

            # Apply conversation threading enhancement
            threading_data = {"threading_enhanced": False}
            if self.threading_service and threading_analysis:
                enhanced_response, threading_data = (
                    await self.threading_service.enhance_response_with_threading(
                        base_response=enhanced_response,
                        threading_analysis=threading_analysis,
                        conversation=conversation,
                        personality=personality,
                        current_topic=memory_context.current_topic or "general",
                    )
                )

            # Update conversation stage if progression is warranted
            await self._check_stage_progression(
                conversation.conversation_id,
                stage,
                conversation.relationship_level,
                npc_id,
            )

            # Track memory usage
            await self._track_memory_usage(
                conversation.conversation_id,
                memory_result,
                integration_data,
            )

            return enhanced_response, {
                **integration_data,
                **threading_data,
                "stage": stage.value,
                "trust_level": trust_level.value,
                "memory_count": len(memory_result.relevant_memories),
                "threading_analysis": {
                    "should_reference": (
                        threading_analysis.should_reference_past
                        if threading_analysis
                        else False
                    ),
                    "reference_confidence": (
                        threading_analysis.reference_confidence
                        if threading_analysis
                        else 0.0
                    ),
                    "topic_evolution_quality": (
                        threading_analysis.topic_evolution_quality
                        if threading_analysis
                        else "unknown"
                    ),
                },
            }

        except Exception as e:
            logger.error(f"Memory pattern enhancement failed: {e}", exc_info=True)
            return base_response, {"error": str(e), "memory_enhanced": False}

    async def _determine_conversation_stage(
        self,
        conversation: ConversationContext,
        personality: NPCPersonality,
    ) -> ConversationStage:
        """Determine current conversation stage."""
        conversation_id = conversation.conversation_id

        # Check cache first
        if conversation_id in self._conversation_stages:
            return self._conversation_stages[conversation_id]

        try:
            async with self.session_factory.get_session() as session:
                repo_manager = ConversationRepositoryManager(session)

                # Get conversation history count using optimized query
                conversation_count = await repo_manager.contexts.get_conversation_count_for_npc_player_pair(
                    uuid.UUID(conversation.player_id),
                    uuid.UUID(conversation.npc_id),
                    status="active",
                )
                relationship_score = conversation.relationship_level

                # Determine stage based on relationship and history
                if conversation_count == 1 and relationship_score < 0.3:
                    stage = ConversationStage.INITIAL_ENCOUNTER
                elif conversation_count <= 3 and relationship_score < 0.5:
                    stage = ConversationStage.ACQUAINTANCE
                elif relationship_score < 0.6:
                    stage = ConversationStage.RELATIONSHIP_BUILDING
                elif relationship_score < 0.8:
                    stage = ConversationStage.TRUST_DEVELOPMENT
                elif relationship_score < 0.9:
                    stage = ConversationStage.DEEP_CONNECTION
                else:
                    stage = ConversationStage.CONFIDANT

                # Apply personality modifiers
                stage = self._apply_personality_stage_modifiers(stage, personality)

                # Cache the result
                self._conversation_stages[conversation_id] = stage
                return stage

        except Exception as e:
            logger.error(f"Error determining conversation stage: {e}")
            return ConversationStage.ACQUAINTANCE  # Safe default

    def _apply_personality_stage_modifiers(
        self, stage: ConversationStage, personality: NPCPersonality
    ) -> ConversationStage:
        """Apply personality-based modifiers to conversation stage."""
        # Cautious personalities progress slower
        if personality.get_trait_strength("cautious") > 0.7:
            stage_progression = {
                ConversationStage.TRUST_DEVELOPMENT: ConversationStage.RELATIONSHIP_BUILDING,
                ConversationStage.DEEP_CONNECTION: ConversationStage.TRUST_DEVELOPMENT,
                ConversationStage.CONFIDANT: ConversationStage.DEEP_CONNECTION,
            }
            return stage_progression.get(stage, stage)

        # Open personalities progress faster
        elif personality.get_trait_strength("open") > 0.7:
            stage_progression = {
                ConversationStage.ACQUAINTANCE: ConversationStage.RELATIONSHIP_BUILDING,
                ConversationStage.RELATIONSHIP_BUILDING: ConversationStage.TRUST_DEVELOPMENT,
            }
            return stage_progression.get(stage, stage)

        return stage

    async def _integrate_memories_with_patterns(
        self,
        base_response: str,
        memories: list[tuple[ConversationExchange, float]],
        trust_level: TrustLevel,
        template: Any,  # ConversationFlowTemplate but avoiding import issues
        personality: NPCPersonality,
        emotional_tone: str,
    ) -> tuple[str, dict[str, Any]]:
        """Integrate memories using appropriate patterns."""
        if not memories:
            return base_response, {"memory_enhanced": False}

        # Select the highest confidence memory
        top_memory, confidence = memories[0]

        # Get disclosure threshold from confidence
        disclosure_threshold = (
            self.flow_library.get_disclosure_threshold_from_confidence(confidence)
        )

        # Get appropriate memory pattern
        pattern = None
        for mem_pattern in template.memory_patterns:
            if mem_pattern.disclosure_level == disclosure_threshold:
                pattern = mem_pattern
                break

        if not pattern:
            # Fallback to the most restrictive pattern available
            pattern = template.memory_patterns[0] if template.memory_patterns else None

        if not pattern:
            return base_response, {
                "memory_enhanced": False,
                "reason": "no_pattern_available",
            }

        # Validate memory integration appropriateness
        is_valid, validation_errors = self.flow_library.validate_memory_integration(
            pattern, trust_level, confidence, {"conversation_history": True}
        )

        if not is_valid:
            logger.info(f"Memory integration validation failed: {validation_errors}")
            return base_response, {
                "memory_enhanced": False,
                "validation_errors": validation_errors,
            }

        # Get appropriate transition phrase
        transition_phrase = pattern.get_appropriate_phrase(
            confidence, emotional_tone, "introduction"
        )

        if not transition_phrase:
            # Fallback to generic transition phrase
            transition_phrase = self.flow_library.get_transition_phrase(
                "uncertainty", confidence, emotional_tone
            )

        # Build enhanced response
        enhanced_response = await self._build_enhanced_response(
            base_response,
            top_memory,
            pattern,
            transition_phrase,
            confidence,
            personality,
        )

        integration_data = {
            "memory_enhanced": True,
            "pattern_used": pattern.pattern_name,
            "disclosure_level": pattern.disclosure_level.value,
            "confidence": confidence,
            "transition_phrase": transition_phrase.text if transition_phrase else None,
            "memory_content": top_memory.message_text[:50] + "...",
        }

        return enhanced_response, integration_data

    async def _build_enhanced_response(
        self,
        base_response: str,
        memory: ConversationExchange,
        pattern: MemoryIntegrationPattern,
        transition_phrase: TransitionPhrase | None,
        confidence: float,
        personality: NPCPersonality,
    ) -> str:
        """Build response enhanced with memory integration."""

        if not transition_phrase:
            return base_response

        # Apply disclosure level logic
        if pattern.disclosure_level == MemoryDisclosureThreshold.SUBTLE_HINTS:
            memory_reference = self._create_subtle_hint(memory, confidence)
        elif pattern.disclosure_level == MemoryDisclosureThreshold.CLEAR_REFERENCES:
            memory_reference = self._create_clear_reference(memory, confidence)
        elif pattern.disclosure_level == MemoryDisclosureThreshold.DETAILED_MEMORIES:
            memory_reference = self._create_detailed_memory(memory, confidence, pattern)
        elif pattern.disclosure_level == MemoryDisclosureThreshold.PERSONAL_SECRETS:
            memory_reference = self._create_personal_secret_reference(
                memory, confidence
            )
        else:
            memory_reference = memory.message_text[:30] + "..."

        # Combine transition phrase with memory reference
        if pattern.disclosure_level == MemoryDisclosureThreshold.SUBTLE_HINTS:
            # For subtle hints, weave into the response naturally
            enhanced_response = f"{transition_phrase.text} {base_response}"
        else:
            # For clearer references, be more explicit
            enhanced_response = (
                f"{transition_phrase.text} {memory_reference} {base_response}"
            )

        return enhanced_response

    def _create_subtle_hint(
        self, memory: ConversationExchange, confidence: float
    ) -> str:
        """Create a subtle hint about the memory."""
        return "something you mentioned before comes to mind"

    def _create_clear_reference(
        self, memory: ConversationExchange, confidence: float
    ) -> str:
        """Create a clear reference to the memory."""
        # Extract key topic or phrase from memory
        words = memory.message_text.split()[:10]  # First 10 words
        return f"when you talked about {' '.join(words)}"

    def _create_detailed_memory(
        self,
        memory: ConversationExchange,
        confidence: float,
        pattern: MemoryIntegrationPattern,
    ) -> str:
        """Create a detailed memory reference."""
        # Include more specific details and emotion if available
        emotion_modifier = ""
        if memory.emotion and memory.emotion in pattern.emotional_modifiers:
            emotion_modifier = f" {pattern.emotional_modifiers[memory.emotion][0]}"

        return f'you said "{memory.message_text[:60]}..."{emotion_modifier}'

    def _create_personal_secret_reference(
        self, memory: ConversationExchange, confidence: float
    ) -> str:
        """Create reference for personal/secret memories."""
        return f"what you confided in me about {memory.message_text[:40]}..."

    async def _check_stage_progression(
        self,
        conversation_id: str,
        current_stage: ConversationStage,
        relationship_score: float,
        npc_id: uuid.UUID,
    ) -> None:
        """Check if conversation stage should progress."""
        try:
            async with self.session_factory.get_session() as session:
                repo_manager = ConversationRepositoryManager(session)

                # Get conversation count for this NPC-Player pair using optimized query
                try:
                    player_id = extract_player_id_from_conversation_id(conversation_id)
                    conversation_count = await repo_manager.contexts.get_conversation_count_for_npc_player_pair(
                        player_id, npc_id, status="active"
                    )
                except UUIDSecurityError as e:
                    self.logger.error(
                        f"Security error extracting player ID from conversation {conversation_id}: {e}"
                    )
                    return  # Fail safely - don't progress conversation with invalid ID

                # Suggest progression
                suggested_stage = self.flow_library.suggest_conversation_progression(
                    current_stage, relationship_score, conversation_count
                )

                if suggested_stage and suggested_stage != current_stage:
                    self._conversation_stages[conversation_id] = suggested_stage
                    logger.info(
                        f"Conversation {conversation_id} progressed from {current_stage.value} "
                        f"to {suggested_stage.value}"
                    )

        except Exception as e:
            logger.error(f"Error checking stage progression: {e}")

    async def _track_memory_usage(
        self,
        conversation_id: str,
        memory_result: Any,  # MemoryRetrievalResult but avoiding circular import
        integration_data: dict[str, Any],
    ) -> None:
        """Track memory usage for analytics and improvement."""
        if conversation_id not in self._memory_usage_history:
            self._memory_usage_history[conversation_id] = []

        usage_record = {
            "timestamp": conversation_id,  # Simplified for now
            "memory_count": len(memory_result.relevant_memories),
            "enhanced": integration_data.get("memory_enhanced", False),
            "pattern": integration_data.get("pattern_used"),
            "confidence": integration_data.get("confidence"),
            "disclosure_level": integration_data.get("disclosure_level"),
        }

        self._memory_usage_history[conversation_id].append(usage_record)

        # Keep only last 50 records per conversation
        if len(self._memory_usage_history[conversation_id]) > 50:
            self._memory_usage_history[conversation_id] = self._memory_usage_history[
                conversation_id
            ][-50:]

    def get_conversation_stage(self, conversation_id: str) -> ConversationStage | None:
        """Get current conversation stage."""
        return self._conversation_stages.get(conversation_id)

    def get_memory_usage_stats(self, conversation_id: str) -> dict[str, Any]:
        """Get memory usage statistics for conversation."""
        history = self._memory_usage_history.get(conversation_id, [])
        if not history:
            return {"total_interactions": 0, "memory_enhanced_count": 0}

        enhanced_count = sum(1 for record in history if record["enhanced"])
        total_count = len(history)

        return {
            "total_interactions": total_count,
            "memory_enhanced_count": enhanced_count,
            "enhancement_rate": enhanced_count / total_count if total_count > 0 else 0,
            "avg_confidence": (
                sum(
                    record.get("confidence", 0)
                    for record in history
                    if record.get("confidence")
                )
                / max(1, len([r for r in history if r.get("confidence")]))
                if history
                else 0
            ),
            "patterns_used": list(
                set(
                    record.get("pattern") for record in history if record.get("pattern")
                )
            ),
        }

    async def analyze_conversation_flow_quality(
        self, conversation_id: str
    ) -> dict[str, Any]:
        """Analyze the quality of conversation flow and memory integration."""
        try:
            history = self._memory_usage_history.get(conversation_id, [])
            stage = self._conversation_stages.get(conversation_id)

            if not history:
                return {
                    "quality_score": 0.5,
                    "recommendations": ["More conversation data needed"],
                }

            # Calculate quality metrics
            enhancement_rate = sum(1 for r in history if r["enhanced"]) / len(history)
            confidence_records = [r for r in history if r.get("confidence")]
            avg_confidence = (
                sum(r.get("confidence", 0) for r in confidence_records)
                / max(1, len(confidence_records))
                if confidence_records
                else 0
            )

            pattern_diversity = len(
                set(r.get("pattern") for r in history if r.get("pattern"))
            )

            # Calculate overall quality score
            quality_score = (
                enhancement_rate * 0.4
                + avg_confidence * 0.4
                + min(pattern_diversity / 3, 1.0) * 0.2
            )

            recommendations = []
            if enhancement_rate < 0.3:
                recommendations.append("Consider enabling more memory integration")
            if avg_confidence < 0.5:
                recommendations.append("Memory confidence could be improved")
            if pattern_diversity < 2:
                recommendations.append("Try using more varied memory patterns")

            return {
                "quality_score": quality_score,
                "enhancement_rate": enhancement_rate,
                "avg_confidence": avg_confidence,
                "pattern_diversity": pattern_diversity,
                "current_stage": stage.value if stage else "unknown",
                "recommendations": recommendations,
            }

        except Exception as e:
            logger.error(f"Error analyzing conversation flow quality: {e}")
            return {"error": str(e)}

    async def initiate_conversation_session(
        self,
        player_id: uuid.UUID,
        npc_id: uuid.UUID,
        conversation: ConversationContext,
        initial_topic: str,
    ) -> dict[str, Any]:
        """
        Initiate a conversation session with threading context.

        This should be called at the start of each conversation to establish
        threading context and prepare for memory references.
        """
        session_data = {"threading_initialized": False}

        if self.threading_service:
            try:
                threading_context = (
                    await self.threading_service.initiate_conversation_session(
                        player_id=player_id,
                        npc_id=npc_id,
                        conversation=conversation,
                        initial_topic=initial_topic,
                    )
                )

                session_data = {
                    "threading_initialized": True,
                    "thread_id": (
                        str(threading_context.active_thread.thread_id)
                        if threading_context.active_thread
                        else None
                    ),
                    "topic_continuity_score": threading_context.topic_continuity_score,
                    "relationship_level": (
                        float(threading_context.player_profile.relationship_level)
                        if threading_context.player_profile
                        else 0.0
                    ),
                    "conversation_hooks": len(threading_context.conversation_hooks),
                }

                logger.info(
                    f"Initiated conversation session with threading for player {player_id}"
                )

            except Exception as e:
                logger.error(f"Error initializing conversation threading: {e}")
                session_data["error"] = str(e)

        return session_data

    async def record_topic_evolution(
        self,
        conversation: ConversationContext,
        previous_topic: str,
        new_topic: str,
        player_initiated: bool = False,
    ) -> None:
        """
        Record topic evolution during conversation.

        Call this when topics change during conversation to track
        topic progression for future threading decisions.
        """
        if self.threading_service and previous_topic != new_topic:
            await self.threading_service.record_conversation_evolution(
                conversation=conversation,
                previous_topic=previous_topic,
                new_topic=new_topic,
                player_initiated=player_initiated,
                evolution_quality="natural",
            )

    async def finalize_conversation_session(
        self,
        conversation: ConversationContext,
        session_success: bool = True,
        relationship_change: float = 0.0,
        memorable_moments: list[str] = None,
    ) -> dict[str, Any]:
        """
        Finalize conversation session and update threading state.

        Call this when a conversation ends to update relationship
        progress and prepare for future conversations.
        """
        finalization_data = {"threading_finalized": False}

        if self.threading_service:
            try:
                await self.threading_service.finalize_conversation_session(
                    conversation=conversation,
                    session_success=session_success,
                    relationship_change=relationship_change,
                    memorable_moments=memorable_moments or [],
                )

                finalization_data = {
                    "threading_finalized": True,
                    "session_success": session_success,
                    "relationship_change": relationship_change,
                    "memorable_moments_recorded": len(memorable_moments or []),
                }

                logger.info(f"Finalized conversation session with threading")

            except Exception as e:
                logger.error(f"Error finalizing conversation threading: {e}")
                finalization_data["error"] = str(e)

        return finalization_data

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
        if self.threading_service:
            return await self.threading_service.get_conversation_preparation(
                player_id=player_id,
                npc_id=npc_id,
                suggested_topics=suggested_topics or [],
            )
        else:
            return {"threading_disabled": True, "suggested_openers": ["Hello there"]}


class EnhancedConversationFlowManager(ConversationFlowManager):
    """Enhanced conversation flow manager with advanced dialogue memory integration."""

    def __init__(
        self,
        memory_integration: ConversationMemoryInterface,
        session_factory: DatabaseSessionFactory,
        flow_library: ConversationFlowLibrary | None = None,
        enable_conversation_threading: bool = True,
        enable_dialogue_integration: bool = True,
        dialogue_integration_engine: DialogueMemoryIntegrationEngine | None = None,
        threading_service: ConversationThreadingService | None = None,
    ):
        # Initialize parent class
        super().__init__(
            memory_integration=memory_integration,
            session_factory=session_factory,
            flow_library=flow_library,
            enable_conversation_threading=enable_conversation_threading,
        )

        # Override threading service if provided
        if threading_service is not None:
            self.threading_service = threading_service

        # Initialize dialogue integration
        self.enable_dialogue_integration = enable_dialogue_integration
        self.dialogue_integration_engine = dialogue_integration_engine

        # Cache for dialogue contexts
        self._dialogue_contexts: dict[str, DialogueContext] = {}

    async def enhance_response_with_advanced_memory_integration(
        self,
        conversation: ConversationContext,
        personality: NPCPersonality,
        base_response: str,
        player_message: str,
        npc_id: uuid.UUID,
    ) -> tuple[str, dict[str, Any]]:
        """
        Enhanced response generation with advanced dialogue memory integration.

        This method builds on the standard memory patterns by adding:
        - Dialogue state analysis
        - Context-sensitive memory timing
        - Natural language memory weaving
        - Flow disruption risk assessment
        """
        if not self.enable_dialogue_integration or not self.dialogue_integration_engine:
            # Fall back to standard memory patterns
            return await self.enhance_response_with_memory_patterns(
                conversation, personality, base_response, player_message, npc_id
            )

        try:
            # Get threading context if available
            threading_context = None
            if self.threading_service:
                try:
                    # Attempt to get or create threading context
                    player_id = uuid.UUID(conversation.player_id)
                    threading_context = (
                        await self.threading_service.initiate_conversation_session(
                            player_id=player_id,
                            npc_id=npc_id,
                            conversation=conversation,
                            initial_topic="current_conversation",
                        )
                    )
                except Exception as e:
                    logger.warning(f"Could not establish threading context: {e}")

            # Analyze dialogue context
            dialogue_context = (
                await self.dialogue_integration_engine.analyze_dialogue_context(
                    conversation=conversation,
                    player_message=player_message,
                    npc_personality=personality,
                    threading_context=threading_context,
                )
            )

            # Get memory context using existing integration
            memory_context = await self.memory_integration.extract_memory_context(
                conversation, player_message, personality
            )

            # Retrieve memories
            memory_retrieval = await self.memory_integration.retrieve_relevant_memories(
                memory_context, npc_id
            )

            # Create memory integration plan using dialogue engine
            integration_plan = (
                await self.dialogue_integration_engine.create_memory_integration_plan(
                    dialogue_context=dialogue_context,
                    memory_retrieval=memory_retrieval,
                    conversation=conversation,
                    npc_personality=personality,
                )
            )

            # Apply dialogue-aware enhancement
            enhancement_result = (
                await self.dialogue_integration_engine.enhance_dialogue_response(
                    base_response=base_response,
                    integration_plan=integration_plan,
                    dialogue_context=dialogue_context,
                    conversation=conversation,
                    npc_personality=personality,
                )
            )

            # Apply traditional memory patterns as fallback if dialogue integration didn't work
            final_response = enhancement_result.enhanced_response
            traditional_data = {}

            if (
                not enhancement_result.integration_applied
                and memory_retrieval.relevant_memories
            ):
                # Fall back to traditional patterns
                trust_level = self.flow_library.get_trust_level_from_relationship(
                    conversation.relationship_level
                )
                stage = await self._determine_conversation_stage(
                    conversation, personality
                )
                template = self.flow_library.get_template_for_stage(stage)

                if template:
                    final_response, traditional_data = (
                        await self._integrate_memories_with_patterns(
                            enhancement_result.enhanced_response,
                            memory_retrieval.relevant_memories,
                            trust_level,
                            template,
                            personality,
                            memory_context.emotional_tone,
                        )
                    )

            # Collect comprehensive metadata
            metadata = {
                "dialogue_integration_enabled": True,
                "dialogue_integration_applied": enhancement_result.integration_applied,
                "integration_style": (
                    enhancement_result.integration_style.value
                    if enhancement_result.integration_applied
                    else None
                ),
                "memories_referenced": enhancement_result.memories_referenced,
                "confidence_score": enhancement_result.confidence_score,
                "naturalness_score": enhancement_result.naturalness_score,
                "engagement_impact": enhancement_result.engagement_impact,
                "dialogue_state": dialogue_context.current_state.value,
                "conversation_depth": dialogue_context.conversation_depth_level,
                "engagement_momentum": dialogue_context.engagement_momentum,
                "memory_reference_density": dialogue_context.memory_reference_density,
                "fallback_to_traditional": bool(traditional_data),
                **traditional_data,
                **enhancement_result.metadata,
            }

            # Track advanced memory usage
            await self._track_advanced_memory_usage(
                conversation.conversation_id,
                dialogue_context,
                enhancement_result,
                integration_plan,
            )

            return final_response, metadata

        except Exception as e:
            logger.error(f"Advanced memory integration failed: {e}", exc_info=True)
            # Fall back to standard memory patterns
            return await self.enhance_response_with_memory_patterns(
                conversation, personality, base_response, player_message, npc_id
            )

    async def analyze_dialogue_readiness(
        self,
        conversation: ConversationContext,
        personality: NPCPersonality,
        player_message: str,
    ) -> dict[str, Any]:
        """
        Analyze readiness for dialogue memory integration.

        This provides insights into whether the current conversation
        state is suitable for memory integration.
        """
        if not self.enable_dialogue_integration or not self.dialogue_integration_engine:
            return {"dialogue_integration_available": False}

        try:
            # Analyze dialogue context
            dialogue_context = (
                await self.dialogue_integration_engine.analyze_dialogue_context(
                    conversation=conversation,
                    player_message=player_message,
                    npc_personality=personality,
                    threading_context=None,
                )
            )

            # Get memory context
            memory_context = await self.memory_integration.extract_memory_context(
                conversation, player_message, personality
            )

            # Get memories for analysis
            memory_retrieval = await self.memory_integration.retrieve_relevant_memories(
                memory_context, uuid.UUID(conversation.npc_id)
            )

            # Create integration plan (without applying it)
            integration_plan = (
                await self.dialogue_integration_engine.create_memory_integration_plan(
                    dialogue_context=dialogue_context,
                    memory_retrieval=memory_retrieval,
                    conversation=conversation,
                    npc_personality=personality,
                )
            )

            return {
                "dialogue_integration_available": True,
                "should_integrate": integration_plan.should_integrate,
                "confidence_score": integration_plan.confidence_score,
                "flow_disruption_risk": integration_plan.flow_disruption_risk,
                "timing_strategy": integration_plan.timing_strategy.value,
                "integration_style": integration_plan.integration_style.value,
                "dialogue_state": dialogue_context.current_state.value,
                "engagement_momentum": dialogue_context.engagement_momentum,
                "conversation_depth": dialogue_context.conversation_depth_level,
                "available_memories": len(memory_retrieval.relevant_memories),
                "emotional_alignment": memory_retrieval.emotional_alignment,
                "fallback_plan": integration_plan.fallback_plan,
            }

        except Exception as e:
            logger.error(f"Error analyzing dialogue readiness: {e}")
            return {"error": str(e), "dialogue_integration_available": False}

    async def get_dialogue_context_summary(
        self, conversation_id: str
    ) -> dict[str, Any]:
        """Get summary of dialogue context for conversation."""
        dialogue_context = self._dialogue_contexts.get(conversation_id)

        if not dialogue_context:
            return {"context_available": False}

        return {
            "context_available": True,
            "dialogue_state": dialogue_context.current_state.value,
            "conversation_depth": dialogue_context.conversation_depth_level,
            "engagement_momentum": dialogue_context.engagement_momentum,
            "memory_density": dialogue_context.memory_reference_density,
            "topic_transitions": len(dialogue_context.topic_transitions),
            "emotional_arc_length": len(dialogue_context.emotional_arc),
            "last_memory_integration": dialogue_context.last_memory_integration,
            "curiosity_indicators": len(dialogue_context.player_curiosity_indicators),
        }

    async def _track_advanced_memory_usage(
        self,
        conversation_id: str,
        dialogue_context: DialogueContext,
        enhancement_result: DialogueEnhancementResult,
        integration_plan: Any,  # MemoryIntegrationPlan
    ) -> None:
        """Track advanced memory usage metrics."""
        # Store dialogue context
        self._dialogue_contexts[conversation_id] = dialogue_context

        # Call parent tracking
        await self._track_memory_usage(
            conversation_id,
            type(
                "MockResult", (), {"relevant_memories": []}
            )(),  # Mock for parent method
            {
                "memory_enhanced": enhancement_result.integration_applied,
                "confidence": enhancement_result.confidence_score,
                "pattern_used": (
                    enhancement_result.integration_style.value
                    if enhancement_result.integration_applied
                    else None
                ),
                "dialogue_enhanced": True,
                "naturalness_score": enhancement_result.naturalness_score,
                "engagement_impact": enhancement_result.engagement_impact,
            },
        )

    def is_dialogue_integration_enabled(self) -> bool:
        """Check if dialogue integration is enabled and available."""
        return (
            self.enable_dialogue_integration
            and self.dialogue_integration_engine is not None
        )
