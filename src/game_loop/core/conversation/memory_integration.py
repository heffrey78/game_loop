"""Memory integration interface for seamless conversation enhancement."""

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from game_loop.database.repositories.semantic_memory import SemanticMemoryRepository
from game_loop.database.session_factory import DatabaseSessionFactory
from game_loop.llm.ollama.client import OllamaClient

from .conversation_models import (
    ConversationContext,
    ConversationExchange,
    NPCPersonality,
)
from .interfaces import ConversationMemoryInterface

logger = logging.getLogger(__name__)


class MemoryDisclosureLevel(Enum):
    """Levels of memory disclosure in conversations."""

    NONE = "none"
    SUBTLE_HINTS = "subtle_hints"
    DIRECT_REFERENCES = "direct_references"
    DETAILED_MEMORIES = "detailed_memories"


class ConversationFlowState(Enum):
    """States for conversation flow analysis."""

    NATURAL = "natural"
    AWKWARD_TRANSITION = "awkward_transition"
    TOPIC_SHIFT = "topic_shift"
    MEMORY_RELEVANT = "memory_relevant"
    MEMORY_INAPPROPRIATE = "memory_inappropriate"


@dataclass
class MemoryContext:
    """Context for memory query extraction."""

    current_topic: str | None = None
    emotional_tone: str = "neutral"
    conversation_history: list[ConversationExchange] = field(default_factory=list)
    player_interests: list[str] = field(default_factory=list)
    npc_knowledge_areas: list[str] = field(default_factory=list)
    session_disclosure_level: MemoryDisclosureLevel = MemoryDisclosureLevel.NONE
    last_memory_reference: dict[str, Any] | None = None
    topic_continuity_score: float = 0.5


@dataclass
class MemoryRetrievalResult:
    """Result of memory retrieval for conversation enhancement."""

    relevant_memories: list[tuple[ConversationExchange, float]] = field(
        default_factory=list
    )
    context_score: float = 0.0
    emotional_alignment: float = 0.0
    disclosure_recommendation: MemoryDisclosureLevel = MemoryDisclosureLevel.NONE
    flow_analysis: ConversationFlowState = ConversationFlowState.NATURAL
    fallback_triggered: bool = False
    error_message: str | None = None


@dataclass
class ConversationState:
    """Extended conversation state for memory integration."""

    conversation_id: str
    disclosure_level: MemoryDisclosureLevel = MemoryDisclosureLevel.NONE
    topic_history: list[str] = field(default_factory=list)
    memory_references_count: int = 0
    last_memory_timestamp: float | None = None
    player_engagement_score: float = 0.5
    memory_enabled: bool = True  # For A/B testing


class MemoryIntegrationInterface(ConversationMemoryInterface):
    """Concrete implementation for integrating semantic memory with conversation management."""

    def __init__(
        self,
        session_factory: DatabaseSessionFactory,
        llm_client: OllamaClient,
        enable_memory_enhancement: bool = True,
        similarity_threshold: float = 0.7,
        max_memories_per_query: int = 5,
    ):
        self.session_factory = session_factory
        self.llm_client = llm_client
        self.enable_memory_enhancement = enable_memory_enhancement
        self.similarity_threshold = similarity_threshold
        self.max_memories_per_query = max_memories_per_query

        # Cache for conversation states
        self._conversation_states: dict[str, ConversationState] = {}

    async def extract_memory_context(
        self,
        conversation: ConversationContext,
        player_message: str,
        npc_personality: NPCPersonality,
    ) -> MemoryContext:
        """Extract context for memory queries from current conversation state."""
        try:
            # Get current topic using LLM
            current_topic = await self._extract_topic_from_message(
                player_message, conversation
            )

            # Determine emotional tone
            emotional_tone = await self._analyze_emotional_tone(
                player_message, conversation.get_recent_exchanges(3)
            )

            # Extract player interests from conversation history
            player_interests = await self._extract_player_interests(
                conversation.conversation_history
            )

            # Get conversation state
            conv_state = self._get_conversation_state(conversation.conversation_id)

            return MemoryContext(
                current_topic=current_topic,
                emotional_tone=emotional_tone,
                conversation_history=conversation.get_recent_exchanges(5),
                player_interests=player_interests,
                npc_knowledge_areas=npc_personality.knowledge_areas,
                session_disclosure_level=conv_state.disclosure_level,
                topic_continuity_score=self._calculate_topic_continuity(
                    current_topic, conv_state.topic_history
                ),
            )
        except Exception as e:
            logger.error(f"Error extracting memory context: {e}")
            # Return minimal context on error
            return MemoryContext(
                current_topic=None,
                emotional_tone="neutral",
                conversation_history=conversation.get_recent_exchanges(5),
            )

    async def retrieve_relevant_memories(
        self,
        memory_context: MemoryContext,
        npc_id: uuid.UUID,
        query_embedding: list[float] | None = None,
    ) -> MemoryRetrievalResult:
        """Retrieve relevant memories based on context with fallback handling."""
        if not self.enable_memory_enhancement:
            return MemoryRetrievalResult(
                flow_analysis=ConversationFlowState.NATURAL,
                disclosure_recommendation=MemoryDisclosureLevel.NONE,
            )

        try:
            async with self.session_factory.get_session() as session:
                memory_repo = SemanticMemoryRepository(session)

                # If no embedding provided, generate one from current topic
                if query_embedding is None and memory_context.current_topic:
                    query_embedding = await self._generate_topic_embedding(
                        memory_context.current_topic
                    )

                if query_embedding is None:
                    # Fallback: use recent memory only
                    return await self._fallback_recent_memory(memory_context)

                # Retrieve similar memories
                similar_memories = await memory_repo.find_similar_memories(
                    query_embedding=query_embedding,
                    npc_id=npc_id,
                    similarity_threshold=self.similarity_threshold,
                    limit=self.max_memories_per_query,
                    exclude_recent_hours=1,  # Don't include very recent memories
                )

                # Analyze conversation flow
                flow_state = await self._analyze_conversation_flow(
                    memory_context, similar_memories
                )

                # Calculate disclosure recommendation
                disclosure_level = self._recommend_disclosure_level(
                    memory_context, similar_memories, flow_state
                )

                # Calculate context and emotional alignment scores
                context_score = self._calculate_context_relevance(
                    memory_context, similar_memories
                )
                emotional_alignment = self._calculate_emotional_alignment(
                    memory_context.emotional_tone, similar_memories
                )

                return MemoryRetrievalResult(
                    relevant_memories=similar_memories,
                    context_score=context_score,
                    emotional_alignment=emotional_alignment,
                    disclosure_recommendation=disclosure_level,
                    flow_analysis=flow_state,
                )

        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            # Graceful fallback
            return await self._fallback_recent_memory(
                memory_context, error_message=str(e)
            )

    async def update_conversation_state(
        self,
        conversation_id: str,
        memory_result: MemoryRetrievalResult,
        player_message: str,
        npc_response: str,
    ) -> None:
        """Update conversation state based on memory integration results."""
        conv_state = self._get_conversation_state(conversation_id)

        # Update disclosure level based on player engagement
        engagement_change = await self._calculate_engagement_change(
            player_message, npc_response
        )
        conv_state.player_engagement_score = max(
            0.0, min(1.0, conv_state.player_engagement_score + engagement_change)
        )

        # Progress disclosure level if player is engaged
        if conv_state.player_engagement_score > 0.7:
            conv_state.disclosure_level = self._progress_disclosure_level(
                conv_state.disclosure_level
            )
        elif conv_state.player_engagement_score < 0.3:
            # Reduce disclosure if player seems disengaged
            conv_state.disclosure_level = self._regress_disclosure_level(
                conv_state.disclosure_level
            )

        # Update topic history
        if memory_result.relevant_memories:
            conv_state.memory_references_count += len(memory_result.relevant_memories)
            conv_state.last_memory_timestamp = max(
                memory.timestamp for memory, _ in memory_result.relevant_memories
            )

    def enable_memory_for_conversation(
        self, conversation_id: str, enable: bool = True
    ) -> None:
        """Enable or disable memory enhancement for A/B testing."""
        conv_state = self._get_conversation_state(conversation_id)
        conv_state.memory_enabled = enable

    def is_memory_enabled_for_conversation(self, conversation_id: str) -> bool:
        """Check if memory is enabled for a specific conversation (A/B testing)."""
        conv_state = self._get_conversation_state(conversation_id)
        return conv_state.memory_enabled and self.enable_memory_enhancement

    async def _extract_topic_from_message(
        self, message: str, conversation: ConversationContext
    ) -> str | None:
        """Extract current conversation topic using LLM."""
        try:
            # Get recent context
            recent_exchanges = conversation.get_recent_exchanges(3)
            context_text = "\n".join(
                [
                    f"{'Player' if ex.speaker_id == conversation.player_id else 'NPC'}: {ex.message_text}"
                    for ex in recent_exchanges
                ]
            )

            prompt = f"""
            Based on this conversation context and the latest message, identify the main topic being discussed.
            
            Recent conversation:
            {context_text}
            
            Latest message: "{message}"
            
            Respond with just the topic in 1-3 words (e.g., "crops", "weather", "village politics").
            If no clear topic, respond with "general conversation".
            """

            response = await self.llm_client.generate_response(
                prompt, model="qwen3:1.7b"
            )
            return response.strip().lower()
        except Exception as e:
            logger.warning(f"Topic extraction failed: {e}")
            return None

    async def _analyze_emotional_tone(
        self, message: str, recent_exchanges: list[ConversationExchange]
    ) -> str:
        """Analyze emotional tone of current conversation."""
        try:
            context_text = "\n".join([ex.message_text for ex in recent_exchanges[-2:]])

            prompt = f"""
            Analyze the emotional tone of this conversation context and message.
            
            Recent context: {context_text}
            Current message: "{message}"
            
            Respond with one word: happy, sad, angry, worried, curious, neutral, excited, frustrated
            """

            response = await self.llm_client.generate_response(
                prompt, model="qwen3:1.7b"
            )
            tone = response.strip().lower()

            # Validate against known emotions
            valid_tones = [
                "happy",
                "sad",
                "angry",
                "worried",
                "curious",
                "neutral",
                "excited",
                "frustrated",
            ]
            return tone if tone in valid_tones else "neutral"
        except Exception as e:
            logger.warning(f"Emotional tone analysis failed: {e}")
            return "neutral"

    async def _extract_player_interests(
        self, conversation_history: list[ConversationExchange]
    ) -> list[str]:
        """Extract player interests from conversation history."""
        try:
            # Simple keyword extraction - could be enhanced with NLP
            interests = []
            player_messages = [
                ex.message_text.lower()
                for ex in conversation_history
                if ex.speaker_id != conversation_history[0].speaker_id
            ]

            # Look for patterns that indicate interest
            for message in player_messages:
                if any(
                    word in message
                    for word in ["tell me about", "what about", "interested in"]
                ):
                    # Extract the subject of interest
                    words = message.split()
                    for i, word in enumerate(words):
                        if word in ["about", "in"] and i + 1 < len(words):
                            interests.append(words[i + 1])

            return list(set(interests))  # Remove duplicates
        except Exception as e:
            logger.warning(f"Interest extraction failed: {e}")
            return []

    def _calculate_topic_continuity(
        self, current_topic: str | None, topic_history: list[str]
    ) -> float:
        """Calculate how well current topic aligns with conversation history."""
        if not current_topic or not topic_history:
            return 0.5

        # Simple matching - could be enhanced with semantic similarity
        if current_topic in topic_history[-3:]:  # Recent topics
            return 0.9
        elif current_topic in topic_history:
            return 0.6
        else:
            return 0.2

    async def _generate_topic_embedding(self, topic: str) -> list[float] | None:
        """Generate embedding for topic (placeholder - would use actual embedding service)."""
        try:
            # This would typically use the same embedding service as the semantic memory system
            # For now, return None to trigger fallback
            logger.info(f"Would generate embedding for topic: {topic}")
            return None
        except Exception as e:
            logger.error(f"Topic embedding generation failed: {e}")
            return None

    async def _analyze_conversation_flow(
        self,
        memory_context: MemoryContext,
        similar_memories: list[tuple[ConversationExchange, float]],
    ) -> ConversationFlowState:
        """Analyze if introducing memories would disrupt conversation flow."""
        if not similar_memories:
            return ConversationFlowState.NATURAL

        # Check emotional alignment
        emotional_scores = []
        for memory, similarity in similar_memories:
            if memory.emotion:
                # Simple emotional alignment check
                if memory.emotion == memory_context.emotional_tone:
                    emotional_scores.append(1.0)
                elif memory_context.emotional_tone in [
                    "happy",
                    "excited",
                ] and memory.emotion in ["happy", "excited"]:
                    emotional_scores.append(0.8)
                elif memory_context.emotional_tone in [
                    "sad",
                    "worried",
                ] and memory.emotion in ["sad", "worried"]:
                    emotional_scores.append(0.8)
                else:
                    emotional_scores.append(0.3)

        avg_emotional_alignment = (
            sum(emotional_scores) / len(emotional_scores) if emotional_scores else 0.5
        )

        # Determine flow state based on alignment and topic continuity
        if (
            avg_emotional_alignment > 0.7
            and memory_context.topic_continuity_score > 0.6
        ):
            return ConversationFlowState.MEMORY_RELEVANT
        elif avg_emotional_alignment < 0.4:
            return ConversationFlowState.MEMORY_INAPPROPRIATE
        elif memory_context.topic_continuity_score < 0.3:
            return ConversationFlowState.AWKWARD_TRANSITION
        else:
            return ConversationFlowState.NATURAL

    def _recommend_disclosure_level(
        self,
        memory_context: MemoryContext,
        similar_memories: list[tuple[ConversationExchange, float]],
        flow_state: ConversationFlowState,
    ) -> MemoryDisclosureLevel:
        """Recommend appropriate level of memory disclosure."""
        if flow_state == ConversationFlowState.MEMORY_INAPPROPRIATE:
            return MemoryDisclosureLevel.NONE

        if not similar_memories:
            return MemoryDisclosureLevel.NONE

        # Base recommendation on current disclosure level and memory relevance
        max_similarity = max(similarity for _, similarity in similar_memories)
        current_level = memory_context.session_disclosure_level

        if max_similarity > 0.9 and flow_state == ConversationFlowState.MEMORY_RELEVANT:
            # High relevance - can use detailed memories
            if current_level in [
                MemoryDisclosureLevel.DIRECT_REFERENCES,
                MemoryDisclosureLevel.DETAILED_MEMORIES,
            ]:
                return MemoryDisclosureLevel.DETAILED_MEMORIES
            else:
                return MemoryDisclosureLevel.DIRECT_REFERENCES
        elif max_similarity > 0.7:
            # Medium relevance - direct references
            if current_level == MemoryDisclosureLevel.NONE:
                return MemoryDisclosureLevel.SUBTLE_HINTS
            else:
                return MemoryDisclosureLevel.DIRECT_REFERENCES
        else:
            # Low relevance - subtle hints only
            return MemoryDisclosureLevel.SUBTLE_HINTS

    def _calculate_context_relevance(
        self,
        memory_context: MemoryContext,
        similar_memories: list[tuple[ConversationExchange, float]],
    ) -> float:
        """Calculate how relevant memories are to current context."""
        if not similar_memories:
            return 0.0

        # Weighted average of similarity scores
        total_weight = sum(similarity for _, similarity in similar_memories)
        return total_weight / len(similar_memories)

    def _calculate_emotional_alignment(
        self,
        current_tone: str,
        similar_memories: list[tuple[ConversationExchange, float]],
    ) -> float:
        """Calculate emotional alignment between current tone and memories."""
        if not similar_memories:
            return 0.5

        alignment_scores = []
        for memory, _ in similar_memories:
            if memory.emotion == current_tone:
                alignment_scores.append(1.0)
            elif current_tone == "neutral":
                alignment_scores.append(0.7)
            else:
                alignment_scores.append(0.3)

        return sum(alignment_scores) / len(alignment_scores)

    async def _calculate_engagement_change(
        self, player_message: str, npc_response: str
    ) -> float:
        """Calculate how player engagement changed based on message exchange."""
        engagement_indicators = {
            "positive": [
                "tell me more",
                "interesting",
                "what else",
                "how",
                "why",
                "really?",
            ],
            "negative": ["okay", "sure", "whatever", "fine", "yeah"],
            "questions": ["?"],
        }

        message_lower = player_message.lower()
        change = 0.0

        # Check for positive engagement
        for indicator in engagement_indicators["positive"]:
            if indicator in message_lower:
                change += 0.1

        # Check for questions (indicate interest)
        if "?" in player_message:
            change += 0.15

        # Check for negative engagement
        for indicator in engagement_indicators["negative"]:
            if indicator in message_lower:
                change -= 0.1

        return max(-0.3, min(0.3, change))

    def _progress_disclosure_level(
        self, current_level: MemoryDisclosureLevel
    ) -> MemoryDisclosureLevel:
        """Progress to next disclosure level."""
        progression = {
            MemoryDisclosureLevel.NONE: MemoryDisclosureLevel.SUBTLE_HINTS,
            MemoryDisclosureLevel.SUBTLE_HINTS: MemoryDisclosureLevel.DIRECT_REFERENCES,
            MemoryDisclosureLevel.DIRECT_REFERENCES: MemoryDisclosureLevel.DETAILED_MEMORIES,
            MemoryDisclosureLevel.DETAILED_MEMORIES: MemoryDisclosureLevel.DETAILED_MEMORIES,
        }
        return progression[current_level]

    def _regress_disclosure_level(
        self, current_level: MemoryDisclosureLevel
    ) -> MemoryDisclosureLevel:
        """Regress to previous disclosure level."""
        regression = {
            MemoryDisclosureLevel.DETAILED_MEMORIES: MemoryDisclosureLevel.DIRECT_REFERENCES,
            MemoryDisclosureLevel.DIRECT_REFERENCES: MemoryDisclosureLevel.SUBTLE_HINTS,
            MemoryDisclosureLevel.SUBTLE_HINTS: MemoryDisclosureLevel.NONE,
            MemoryDisclosureLevel.NONE: MemoryDisclosureLevel.NONE,
        }
        return regression[current_level]

    async def _fallback_recent_memory(
        self, memory_context: MemoryContext, error_message: str | None = None
    ) -> MemoryRetrievalResult:
        """Fallback to recent conversation memory when semantic memory fails."""
        # Use only the last 5 exchanges as "memory"
        recent_memories = memory_context.conversation_history[-5:]

        return MemoryRetrievalResult(
            relevant_memories=[],  # No semantic memories available
            context_score=0.3,  # Low but not zero
            emotional_alignment=0.5,
            disclosure_recommendation=MemoryDisclosureLevel.NONE,
            flow_analysis=ConversationFlowState.NATURAL,
            fallback_triggered=True,
            error_message=error_message,
        )

    def _get_conversation_state(self, conversation_id: str) -> ConversationState:
        """Get or create conversation state for memory tracking."""
        if conversation_id not in self._conversation_states:
            self._conversation_states[conversation_id] = ConversationState(
                conversation_id=conversation_id
            )
        return self._conversation_states[conversation_id]
