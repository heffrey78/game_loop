"""Context-sensitive memory integration engine for natural dialogue enhancement."""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from game_loop.database.session_factory import DatabaseSessionFactory
from game_loop.llm.ollama.client import OllamaClient

from .conversation_models import (
    ConversationContext,
    ConversationExchange,
    NPCPersonality,
)
from .conversation_threading import ConversationThreadingService, ThreadingContext
from .memory_integration import (
    MemoryIntegrationInterface,
    MemoryContext,
    MemoryRetrievalResult,
)

logger = logging.getLogger(__name__)


class DialogueState(Enum):
    """Current state of dialogue flow for memory integration."""

    OPENING = "opening"
    BUILDING = "building"
    DEEPENING = "deepening"
    CLIMAX = "climax"
    WINDING_DOWN = "winding_down"
    CLOSING = "closing"


class MemoryIntegrationTiming(Enum):
    """Timing strategies for memory integration."""

    IMMEDIATE = "immediate"  # Reference memory right away
    NATURAL_PAUSE = "natural_pause"  # Wait for natural conversation pause
    TOPIC_BRIDGE = "topic_bridge"  # Use memory to bridge topics
    RESPONSE_ENHANCEMENT = "response_enhancement"  # Enhance existing response
    DELAYED = "delayed"  # Save for later in conversation


class IntegrationStyle(Enum):
    """Styles of memory integration into dialogue."""

    SUBTLE_HINT = "subtle_hint"  # "This reminds me of something..."
    DIRECT_REFERENCE = "direct_reference"  # "Remember when we talked about..."
    EMOTIONAL_CONNECTION = "emotional_connection"  # "I felt the same way when..."
    COMPARATIVE = "comparative"  # "Unlike last time, this seems..."
    BUILDUP = "buildup"  # Progressive revelation across responses
    NARRATIVE_WEAVING = "narrative_weaving"  # Weave into ongoing story


@dataclass
class DialogueContext:
    """Extended context for dialogue-aware memory integration."""

    conversation_id: str
    current_state: DialogueState = DialogueState.OPENING
    topic_transitions: list[tuple[str, str, float]] = field(
        default_factory=list
    )  # (from, to, naturalness)
    emotional_arc: list[tuple[str, float]] = field(
        default_factory=list
    )  # (emotion, intensity)
    engagement_momentum: float = 0.5
    memory_reference_density: float = 0.0  # References per exchange
    last_memory_integration: float | None = None
    conversation_depth_level: int = 1  # 1-5, deeper = more intimate memories allowed
    player_curiosity_indicators: list[str] = field(default_factory=list)
    npc_knowledge_revealed: set[str] = field(default_factory=set)


@dataclass
class MemoryIntegrationPlan:
    """Plan for integrating memory into dialogue response."""

    should_integrate: bool = False
    timing_strategy: MemoryIntegrationTiming = MemoryIntegrationTiming.IMMEDIATE
    integration_style: IntegrationStyle = IntegrationStyle.SUBTLE_HINT
    confidence_score: float = 0.0
    memory_references: list[dict[str, Any]] = field(default_factory=list)
    enhancement_text: str = ""
    flow_disruption_risk: float = 0.0
    emotional_alignment: float = 0.0
    fallback_plan: str = ""


@dataclass
class DialogueEnhancementResult:
    """Result of dialogue enhancement with memory integration."""

    enhanced_response: str
    integration_applied: bool = False
    integration_style: IntegrationStyle = IntegrationStyle.SUBTLE_HINT
    memories_referenced: int = 0
    confidence_score: float = 0.0
    dialogue_state_change: DialogueState | None = None
    engagement_impact: float = 0.0
    naturalness_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class DialogueMemoryIntegrationEngine:
    """Advanced engine for context-sensitive memory integration in dialogue."""

    def __init__(
        self,
        session_factory: DatabaseSessionFactory,
        llm_client: OllamaClient,
        threading_service: ConversationThreadingService,
        memory_integration: MemoryIntegrationInterface,
        max_references_per_response: int = 2,
        naturalness_threshold: float = 0.7,
        engagement_threshold: float = 0.6,
    ):
        self.session_factory = session_factory
        self.llm_client = llm_client
        self.threading_service = threading_service
        self.memory_integration = memory_integration
        self.max_references_per_response = max_references_per_response
        self.naturalness_threshold = naturalness_threshold
        self.engagement_threshold = engagement_threshold

        # Cache for dialogue contexts
        self._dialogue_contexts: dict[str, DialogueContext] = {}

        # Timing control to prevent over-integration
        self._last_integration_times: dict[str, float] = {}
        self._min_integration_interval: float = 30.0  # seconds

    async def analyze_dialogue_context(
        self,
        conversation: ConversationContext,
        player_message: str,
        npc_personality: NPCPersonality,
        threading_context: ThreadingContext | None = None,
    ) -> DialogueContext:
        """Analyze current dialogue context for memory integration opportunities."""
        dialogue_context = self._get_dialogue_context(conversation.conversation_id)

        # Extract current topic first
        current_topic = await self._extract_current_topic(player_message, conversation)

        # Update emotional arc
        current_emotion = await self._detect_emotional_tone(
            player_message, conversation.get_recent_exchanges(3)
        )
        emotion_intensity = await self._calculate_emotion_intensity(
            player_message, current_emotion
        )
        dialogue_context.emotional_arc.append((current_emotion, emotion_intensity))

        # Update dialogue state based on conversation flow
        new_state = self._determine_dialogue_state(
            conversation.conversation_history,
            current_topic,
            conversation.relationship_level,
        )
        dialogue_context.current_state = new_state

        # Update engagement momentum
        dialogue_context.engagement_momentum = (
            await self._calculate_engagement_momentum(
                player_message, dialogue_context, threading_context
            )
        )

        # Track topic transitions
        if dialogue_context.topic_transitions and current_topic:
            last_topic = (
                dialogue_context.topic_transitions[-1][1]
                if dialogue_context.topic_transitions
                else None
            )
            if last_topic and last_topic != current_topic:
                transition_naturalness = (
                    await self._assess_topic_transition_naturalness(
                        last_topic, current_topic, player_message
                    )
                )
                dialogue_context.topic_transitions.append(
                    (last_topic, current_topic, transition_naturalness)
                )
        elif current_topic:
            dialogue_context.topic_transitions.append(("", current_topic, 1.0))

        # Update conversation depth
        dialogue_context.conversation_depth_level = self._calculate_conversation_depth(
            conversation, dialogue_context
        )

        # Track player curiosity indicators
        curiosity_indicators = await self._extract_curiosity_indicators(player_message)
        dialogue_context.player_curiosity_indicators.extend(curiosity_indicators)

        return dialogue_context

    async def create_memory_integration_plan(
        self,
        dialogue_context: DialogueContext,
        memory_retrieval: MemoryRetrievalResult,
        conversation: ConversationContext,
        npc_personality: NPCPersonality,
    ) -> MemoryIntegrationPlan:
        """Create a plan for integrating memories into the dialogue response."""
        plan = MemoryIntegrationPlan()

        # Check if we should integrate at all
        if not await self._should_integrate_memories(
            dialogue_context, memory_retrieval, conversation
        ):
            plan.fallback_plan = "No integration - maintaining natural flow"
            return plan

        # Determine timing strategy
        plan.timing_strategy = await self._select_timing_strategy(
            dialogue_context, memory_retrieval
        )

        # Select integration style based on context
        plan.integration_style = await self._select_integration_style(
            dialogue_context, memory_retrieval, npc_personality
        )

        # Calculate confidence and risk scores
        plan.confidence_score = await self._calculate_integration_confidence(
            dialogue_context, memory_retrieval, plan.integration_style
        )
        plan.flow_disruption_risk = await self._assess_flow_disruption_risk(
            dialogue_context, memory_retrieval, plan.timing_strategy
        )

        # Select and prepare memory references
        plan.memory_references = await self._select_memory_references(
            memory_retrieval, plan.integration_style, self.max_references_per_response
        )

        # Generate enhancement text
        if plan.memory_references:
            plan.enhancement_text = await self._generate_integration_text(
                plan.memory_references, plan.integration_style, dialogue_context
            )
            plan.should_integrate = (
                plan.confidence_score >= self.naturalness_threshold
                and plan.flow_disruption_risk < 0.3
            )

        plan.emotional_alignment = memory_retrieval.emotional_alignment

        if not plan.should_integrate:
            plan.fallback_plan = f"Integration declined - confidence: {plan.confidence_score:.2f}, risk: {plan.flow_disruption_risk:.2f}"

        return plan

    async def enhance_dialogue_response(
        self,
        base_response: str,
        integration_plan: MemoryIntegrationPlan,
        dialogue_context: DialogueContext,
        conversation: ConversationContext,
        npc_personality: NPCPersonality,
    ) -> DialogueEnhancementResult:
        """Enhance dialogue response with contextual memory integration."""
        result = DialogueEnhancementResult(enhanced_response=base_response)

        if not integration_plan.should_integrate:
            result.metadata["reason"] = integration_plan.fallback_plan
            return result

        try:
            # Apply memory integration based on timing strategy
            if integration_plan.timing_strategy == MemoryIntegrationTiming.IMMEDIATE:
                enhanced_response = await self._integrate_immediately(
                    base_response, integration_plan, dialogue_context, npc_personality
                )
            elif (
                integration_plan.timing_strategy
                == MemoryIntegrationTiming.NATURAL_PAUSE
            ):
                enhanced_response = await self._integrate_at_natural_pause(
                    base_response, integration_plan, dialogue_context, npc_personality
                )
            elif (
                integration_plan.timing_strategy == MemoryIntegrationTiming.TOPIC_BRIDGE
            ):
                enhanced_response = await self._integrate_as_topic_bridge(
                    base_response, integration_plan, dialogue_context, npc_personality
                )
            else:  # RESPONSE_ENHANCEMENT
                enhanced_response = await self._enhance_existing_response(
                    base_response, integration_plan, dialogue_context, npc_personality
                )

            # Validate naturalness of enhanced response
            naturalness_score = await self._validate_response_naturalness(
                base_response, enhanced_response, dialogue_context
            )

            if naturalness_score >= self.naturalness_threshold:
                result.enhanced_response = enhanced_response
                result.integration_applied = True
                result.integration_style = integration_plan.integration_style
                result.memories_referenced = len(integration_plan.memory_references)
                result.confidence_score = integration_plan.confidence_score
                result.naturalness_score = naturalness_score

                # Calculate engagement impact
                result.engagement_impact = await self._calculate_engagement_impact(
                    integration_plan, dialogue_context
                )

                # Update dialogue context
                await self._update_dialogue_context_post_integration(
                    dialogue_context, integration_plan, result
                )

            else:
                result.metadata["reason"] = (
                    f"Enhancement rejected - naturalness too low: {naturalness_score:.2f}"
                )
                logger.info(
                    f"Memory integration rejected due to low naturalness: {naturalness_score:.2f}"
                )

        except Exception as e:
            logger.error(f"Error enhancing dialogue response: {e}")
            result.metadata["error"] = str(e)

        return result

    def _get_dialogue_context(self, conversation_id: str) -> DialogueContext:
        """Get or create dialogue context."""
        if conversation_id not in self._dialogue_contexts:
            self._dialogue_contexts[conversation_id] = DialogueContext(
                conversation_id=conversation_id
            )
        return self._dialogue_contexts[conversation_id]

    async def _should_integrate_memories(
        self,
        dialogue_context: DialogueContext,
        memory_retrieval: MemoryRetrievalResult,
        conversation: ConversationContext,
    ) -> bool:
        """Determine if memories should be integrated based on context."""
        # Check timing constraints
        last_integration = self._last_integration_times.get(
            conversation.conversation_id
        )
        if (
            last_integration
            and (time.time() - last_integration) < self._min_integration_interval
        ):
            return False

        # Check memory density
        if (
            dialogue_context.memory_reference_density > 0.4
        ):  # Too many recent references
            return False

        # Check dialogue state appropriateness
        inappropriate_states = [DialogueState.OPENING]
        if dialogue_context.current_state in inappropriate_states:
            return False

        # Check engagement level
        if dialogue_context.engagement_momentum < self.engagement_threshold:
            return False

        # Check memory relevance
        if memory_retrieval.context_score < 0.5:
            return False

        return True

    async def _select_timing_strategy(
        self,
        dialogue_context: DialogueContext,
        memory_retrieval: MemoryRetrievalResult,
    ) -> MemoryIntegrationTiming:
        """Select appropriate timing strategy for memory integration."""
        if dialogue_context.current_state == DialogueState.BUILDING:
            return MemoryIntegrationTiming.NATURAL_PAUSE
        elif dialogue_context.current_state == DialogueState.DEEPENING:
            return MemoryIntegrationTiming.IMMEDIATE
        elif len(dialogue_context.topic_transitions) > 0:
            last_transition_naturalness = dialogue_context.topic_transitions[-1][2]
            if last_transition_naturalness < 0.7:
                return MemoryIntegrationTiming.TOPIC_BRIDGE

        return MemoryIntegrationTiming.RESPONSE_ENHANCEMENT

    async def _select_integration_style(
        self,
        dialogue_context: DialogueContext,
        memory_retrieval: MemoryRetrievalResult,
        npc_personality: NPCPersonality,
    ) -> IntegrationStyle:
        """Select appropriate integration style based on context and personality."""
        # Consider relationship depth
        if dialogue_context.conversation_depth_level >= 4:
            return IntegrationStyle.EMOTIONAL_CONNECTION
        elif dialogue_context.conversation_depth_level >= 3:
            return IntegrationStyle.DIRECT_REFERENCE

        # Consider NPC personality
        if npc_personality.get_trait_strength("talkative") > 0.7:
            return IntegrationStyle.NARRATIVE_WEAVING
        elif npc_personality.get_trait_strength("wise") > 0.7:
            return IntegrationStyle.COMPARATIVE
        elif npc_personality.get_trait_strength("reserved") > 0.6:
            return IntegrationStyle.SUBTLE_HINT

        # Consider emotional alignment
        if memory_retrieval.emotional_alignment > 0.8:
            return IntegrationStyle.EMOTIONAL_CONNECTION

        return IntegrationStyle.DIRECT_REFERENCE

    async def _calculate_integration_confidence(
        self,
        dialogue_context: DialogueContext,
        memory_retrieval: MemoryRetrievalResult,
        integration_style: IntegrationStyle,
    ) -> float:
        """Calculate confidence score for memory integration."""
        base_confidence = memory_retrieval.context_score

        # Adjust for dialogue state
        state_multipliers = {
            DialogueState.OPENING: 0.3,
            DialogueState.BUILDING: 0.8,
            DialogueState.DEEPENING: 1.0,
            DialogueState.CLIMAX: 1.2,
            DialogueState.WINDING_DOWN: 0.7,
            DialogueState.CLOSING: 0.4,
        }
        base_confidence *= state_multipliers[dialogue_context.current_state]

        # Adjust for engagement
        base_confidence *= dialogue_context.engagement_momentum

        # Adjust for integration style complexity
        style_difficulty = {
            IntegrationStyle.SUBTLE_HINT: 0.9,
            IntegrationStyle.DIRECT_REFERENCE: 1.0,
            IntegrationStyle.EMOTIONAL_CONNECTION: 0.8,
            IntegrationStyle.COMPARATIVE: 0.7,
            IntegrationStyle.BUILDUP: 0.6,
            IntegrationStyle.NARRATIVE_WEAVING: 0.5,
        }
        base_confidence *= style_difficulty[integration_style]

        return min(1.0, max(0.0, base_confidence))

    async def _assess_flow_disruption_risk(
        self,
        dialogue_context: DialogueContext,
        memory_retrieval: MemoryRetrievalResult,
        timing_strategy: MemoryIntegrationTiming,
    ) -> float:
        """Assess risk of disrupting conversation flow."""
        risk = 0.0

        # High risk if recent topic transition was unnatural
        if dialogue_context.topic_transitions:
            last_transition_naturalness = dialogue_context.topic_transitions[-1][2]
            if last_transition_naturalness < 0.5:
                risk += 0.3

        # Risk based on emotional misalignment
        if memory_retrieval.emotional_alignment < 0.4:
            risk += 0.4

        # Risk based on timing strategy
        timing_risks = {
            MemoryIntegrationTiming.IMMEDIATE: 0.1,
            MemoryIntegrationTiming.NATURAL_PAUSE: 0.0,
            MemoryIntegrationTiming.TOPIC_BRIDGE: 0.2,
            MemoryIntegrationTiming.RESPONSE_ENHANCEMENT: 0.0,
            MemoryIntegrationTiming.DELAYED: 0.0,
        }
        risk += timing_risks[timing_strategy]

        # Risk from memory density
        risk += dialogue_context.memory_reference_density * 0.3

        return min(1.0, max(0.0, risk))

    async def _select_memory_references(
        self,
        memory_retrieval: MemoryRetrievalResult,
        integration_style: IntegrationStyle,
        max_references: int,
    ) -> list[dict[str, Any]]:
        """Select most appropriate memory references for integration."""
        if not memory_retrieval.relevant_memories:
            return []

        # Sort by relevance
        sorted_memories = sorted(
            memory_retrieval.relevant_memories,
            key=lambda x: x[1],  # Sort by similarity score
            reverse=True,
        )

        # Select based on integration style
        if integration_style == IntegrationStyle.SUBTLE_HINT:
            # Use only the most relevant memory
            selected = sorted_memories[:1]
        elif integration_style in [
            IntegrationStyle.NARRATIVE_WEAVING,
            IntegrationStyle.BUILDUP,
        ]:
            # Can use multiple memories
            selected = sorted_memories[:max_references]
        else:
            # Standard selection
            selected = sorted_memories[: min(max_references, 2)]

        # Convert to reference format
        references = []
        for memory, similarity in selected:
            references.append(
                {
                    "memory": memory,
                    "similarity": similarity,
                    "reference_text": memory.message_text,
                    "emotional_context": memory.emotion or "neutral",
                    "timestamp": memory.timestamp,
                }
            )

        return references

    async def _generate_integration_text(
        self,
        memory_references: list[dict[str, Any]],
        integration_style: IntegrationStyle,
        dialogue_context: DialogueContext,
    ) -> str:
        """Generate natural integration text based on memories and style."""
        if not memory_references:
            return ""

        # Style-specific generation patterns
        if integration_style == IntegrationStyle.SUBTLE_HINT:
            return await self._generate_subtle_hint(memory_references[0])
        elif integration_style == IntegrationStyle.DIRECT_REFERENCE:
            return await self._generate_direct_reference(memory_references[0])
        elif integration_style == IntegrationStyle.EMOTIONAL_CONNECTION:
            return await self._generate_emotional_connection(memory_references[0])
        elif integration_style == IntegrationStyle.COMPARATIVE:
            return await self._generate_comparative_reference(memory_references[0])
        elif integration_style == IntegrationStyle.NARRATIVE_WEAVING:
            return await self._generate_narrative_weaving(memory_references)

        # Fallback to direct reference
        return await self._generate_direct_reference(memory_references[0])

    async def _generate_subtle_hint(self, memory_reference: dict[str, Any]) -> str:
        """Generate subtle hint integration."""
        hints = [
            "This reminds me of something...",
            "Something about this feels familiar.",
            "I've encountered something like this before.",
            "This brings back memories.",
            "There's something familiar about this situation.",
        ]
        return hints[hash(memory_reference["reference_text"]) % len(hints)]

    async def _generate_direct_reference(self, memory_reference: dict[str, Any]) -> str:
        """Generate direct reference integration."""
        memory_text = memory_reference["reference_text"]

        patterns = [
            f"Remember when we discussed {self._extract_key_topic(memory_text)}?",
            f"This reminds me of our conversation about {self._extract_key_topic(memory_text)}.",
            f"Like we talked about before regarding {self._extract_key_topic(memory_text)}...",
            f"As I mentioned when we spoke about {self._extract_key_topic(memory_text)}...",
        ]
        return patterns[hash(memory_text) % len(patterns)]

    async def _generate_emotional_connection(
        self, memory_reference: dict[str, Any]
    ) -> str:
        """Generate emotional connection integration."""
        emotion = memory_reference["emotional_context"]

        emotional_patterns = {
            "happy": "I felt the same joy when we talked about this before.",
            "sad": "This brings back the melancholy we shared earlier.",
            "excited": "I feel that same excitement we had when discussing this!",
            "worried": "This concerns me the same way it did before.",
            "curious": "My curiosity is piqued just like last time.",
        }

        return emotional_patterns.get(
            emotion, "This evokes the same feelings as before."
        )

    async def _generate_comparative_reference(
        self, memory_reference: dict[str, Any]
    ) -> str:
        """Generate comparative reference integration."""
        patterns = [
            "Unlike our previous discussion, this seems different.",
            "This contrasts with what we talked about before.",
            "Compared to our earlier conversation, this is interesting.",
            "This reminds me of before, but with a different perspective.",
        ]
        return patterns[hash(memory_reference["reference_text"]) % len(patterns)]

    async def _generate_narrative_weaving(
        self, memory_references: list[dict[str, Any]]
    ) -> str:
        """Generate narrative weaving integration with multiple memories."""
        if len(memory_references) == 1:
            return await self._generate_direct_reference(memory_references[0])

        # Weave multiple references into a narrative
        return "Our conversations have touched on this theme several times, each adding new understanding."

    def _extract_key_topic(self, text: str) -> str:
        """Extract key topic from memory text."""
        # Simple keyword extraction - could be enhanced with NLP
        words = text.lower().split()

        # Filter out common words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "among",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "her",
            "its",
            "our",
            "their",
        }

        meaningful_words = [w for w in words if w not in stop_words and len(w) > 3]

        return meaningful_words[0] if meaningful_words else "that topic"

    async def _integrate_immediately(
        self,
        base_response: str,
        integration_plan: MemoryIntegrationPlan,
        dialogue_context: DialogueContext,
    ) -> str:
        """Integrate memory immediately at start of response."""
        if integration_plan.enhancement_text:
            return f"{integration_plan.enhancement_text} {base_response}"
        return base_response

    async def _integrate_at_natural_pause(
        self,
        base_response: str,
        integration_plan: MemoryIntegrationPlan,
        dialogue_context: DialogueContext,
    ) -> str:
        """Integrate memory at natural pause in response."""
        if integration_plan.enhancement_text:
            # Find natural pause point (after sentence)
            sentences = base_response.split(". ")
            if len(sentences) > 1:
                first_part = sentences[0] + "."
                rest = ". ".join(sentences[1:])
                return f"{first_part} {integration_plan.enhancement_text} {rest}"
        return base_response

    async def _integrate_as_topic_bridge(
        self,
        base_response: str,
        integration_plan: MemoryIntegrationPlan,
        dialogue_context: DialogueContext,
    ) -> str:
        """Use memory as bridge between topics."""
        if integration_plan.enhancement_text:
            return f"{integration_plan.enhancement_text} {base_response}"
        return base_response

    async def _enhance_existing_response(
        self,
        base_response: str,
        integration_plan: MemoryIntegrationPlan,
        dialogue_context: DialogueContext,
    ) -> str:
        """Enhance existing response with memory context."""
        if integration_plan.enhancement_text:
            return f"{base_response} {integration_plan.enhancement_text}"
        return base_response

    async def _detect_emotional_tone(
        self, message: str, recent_exchanges: list[ConversationExchange]
    ) -> str:
        """Detect emotional tone of current message."""
        try:
            context_text = "\n".join([ex.message_text for ex in recent_exchanges[-2:]])

            prompt = f"""
            Analyze the emotional tone of this message in context:
            
            Recent context: {context_text}
            Current message: "{message}"
            
            Respond with one word: happy, sad, angry, worried, curious, neutral, excited, frustrated, determined, confused
            """

            response = await self.llm_client.generate_response(
                prompt, model="qwen3:1.7b"
            )
            return response.strip().lower()
        except Exception as e:
            logger.warning(f"Emotional tone detection failed: {e}")
            return "neutral"

    async def _calculate_emotion_intensity(self, message: str, emotion: str) -> float:
        """Calculate intensity of detected emotion."""
        intensity = 0.5  # Base intensity

        # Increase for exclamation marks
        intensity += message.count("!") * 0.1

        # Increase for question marks (curiosity)
        if emotion == "curious":
            intensity += message.count("?") * 0.1

        # Increase for caps
        if any(c.isupper() for c in message):
            caps_ratio = sum(1 for c in message if c.isupper()) / len(message)
            intensity += caps_ratio * 0.3

        # Increase for emotional keywords
        emotional_intensifiers = {
            "very",
            "really",
            "extremely",
            "incredibly",
            "absolutely",
            "definitely",
            "certainly",
            "surely",
            "completely",
            "totally",
        }

        for intensifier in emotional_intensifiers:
            if intensifier in message.lower():
                intensity += 0.1

        return min(1.0, max(0.0, intensity))

    async def _determine_dialogue_state(
        self,
        conversation: ConversationContext,
        player_message: str,
        dialogue_context: DialogueContext,
    ) -> DialogueState:
        """Determine current dialogue state based on conversation flow."""
        exchange_count = conversation.get_exchange_count()

        # Simple state determination based on conversation length and patterns
        if exchange_count <= 2:
            return DialogueState.OPENING
        elif exchange_count <= 5:
            return DialogueState.BUILDING
        elif exchange_count <= 10:
            # Check for deepening indicators
            if any(
                indicator in player_message.lower()
                for indicator in [
                    "tell me about",
                    "what do you think",
                    "how do you feel",
                    "why",
                ]
            ):
                return DialogueState.DEEPENING
            return DialogueState.BUILDING
        elif exchange_count <= 15:
            return DialogueState.CLIMAX
        elif any(
            indicator in player_message.lower()
            for indicator in ["bye", "goodbye", "thanks", "see you"]
        ):
            return DialogueState.CLOSING
        else:
            return DialogueState.WINDING_DOWN

    async def _calculate_engagement_momentum(
        self,
        player_message: str,
        dialogue_context: DialogueContext,
        threading_context: ThreadingContext | None,
    ) -> float:
        """Calculate current engagement momentum."""
        base_engagement = 0.5

        # Increase for questions
        base_engagement += player_message.count("?") * 0.1

        # Increase for interest indicators
        interest_indicators = [
            "tell me",
            "what about",
            "how",
            "why",
            "interesting",
            "really",
        ]
        for indicator in interest_indicators:
            if indicator in player_message.lower():
                base_engagement += 0.1

        # Increase based on message length (engagement indicator)
        if len(player_message.split()) > 5:
            base_engagement += 0.1

        # Use threading context if available
        if threading_context:
            base_engagement += threading_context.topic_continuity_score * 0.2

        # Apply momentum from previous interactions
        if dialogue_context.emotional_arc:
            recent_emotions = dialogue_context.emotional_arc[-3:]
            positive_emotions = sum(
                1
                for emotion, intensity in recent_emotions
                if emotion in ["happy", "excited", "curious"]
            )
            base_engagement += (positive_emotions / len(recent_emotions)) * 0.2

        return min(1.0, max(0.0, base_engagement))

    async def _extract_current_topic(
        self, message: str, conversation: ConversationContext
    ) -> str | None:
        """Extract current topic from message."""
        try:
            recent_exchanges = conversation.get_recent_exchanges(3)
            context_text = "\n".join(
                [
                    f"{'Player' if ex.speaker_id == conversation.player_id else 'NPC'}: {ex.message_text}"
                    for ex in recent_exchanges
                ]
            )

            prompt = f"""
            Based on this conversation context and message, identify the main topic:
            
            Recent conversation:
            {context_text}
            
            Latest message: "{message}"
            
            Respond with just the topic in 1-3 words.
            """

            response = await self.llm_client.generate_response(
                prompt, model="qwen3:1.7b"
            )
            return response.strip().lower()
        except Exception as e:
            logger.warning(f"Topic extraction failed: {e}")
            return None

    async def _assess_topic_transition_naturalness(
        self, from_topic: str, to_topic: str, player_message: str
    ) -> float:
        """Assess how natural a topic transition is."""
        naturalness = 0.5

        # Check for explicit transition indicators
        transition_words = [
            "speaking of",
            "that reminds me",
            "on that note",
            "also",
            "by the way",
        ]
        for word in transition_words:
            if word in player_message.lower():
                naturalness += 0.3
                break

        # Check semantic similarity (simplified)
        common_words = set(from_topic.split()) & set(to_topic.split())
        if common_words:
            naturalness += 0.2

        return min(1.0, max(0.0, naturalness))

    def _calculate_conversation_depth(
        self, conversation: ConversationContext, dialogue_context: DialogueContext
    ) -> int:
        """Calculate conversation depth level (1-5)."""
        depth = 1

        # Increase based on exchange count
        exchange_count = conversation.get_exchange_count()
        if exchange_count > 5:
            depth += 1
        if exchange_count > 10:
            depth += 1

        # Increase based on emotional variety
        if dialogue_context.emotional_arc:
            unique_emotions = set(
                emotion for emotion, _ in dialogue_context.emotional_arc
            )
            if len(unique_emotions) > 2:
                depth += 1

        # Increase based on relationship level
        if conversation.relationship_level > 0.5:
            depth += 1

        return min(5, depth)

    async def _extract_curiosity_indicators(self, message: str) -> list[str]:
        """Extract indicators of player curiosity."""
        indicators = []
        message_lower = message.lower()

        # Question words
        question_words = ["what", "how", "why", "when", "where", "who", "which"]
        for word in question_words:
            if word in message_lower:
                indicators.append(f"question_word_{word}")

        # Curiosity phrases
        curiosity_phrases = [
            "tell me about",
            "what about",
            "i wonder",
            "curious about",
            "interested in",
            "want to know",
            "can you explain",
        ]
        for phrase in curiosity_phrases:
            if phrase in message_lower:
                indicators.append(f"curiosity_phrase_{phrase.replace(' ', '_')}")

        return indicators

    async def _validate_response_naturalness(
        self,
        original_response: str,
        enhanced_response: str,
        dialogue_context: DialogueContext,
    ) -> float:
        """Validate that enhanced response maintains naturalness."""
        try:
            length_ratio = len(enhanced_response) / len(original_response)

            # Penalize if response becomes too long
            if length_ratio > 2.0:
                return 0.4
            elif length_ratio > 1.5:
                return 0.6

            # Check for awkward transitions (simplified)
            awkward_patterns = ["  ", "...", "and and", "the the"]
            awkwardness = sum(
                enhanced_response.count(pattern) for pattern in awkward_patterns
            )

            naturalness = 0.8 - (awkwardness * 0.1)

            return max(0.0, min(1.0, naturalness))

        except Exception as e:
            logger.warning(f"Naturalness validation failed: {e}")
            return 0.5

    async def _calculate_engagement_impact(
        self, integration_plan: MemoryIntegrationPlan, dialogue_context: DialogueContext
    ) -> float:
        """Calculate expected engagement impact of memory integration."""
        base_impact = 0.1  # Small positive impact by default

        # Increase impact based on integration style
        style_impacts = {
            IntegrationStyle.SUBTLE_HINT: 0.05,
            IntegrationStyle.DIRECT_REFERENCE: 0.15,
            IntegrationStyle.EMOTIONAL_CONNECTION: 0.25,
            IntegrationStyle.COMPARATIVE: 0.20,
            IntegrationStyle.BUILDUP: 0.30,
            IntegrationStyle.NARRATIVE_WEAVING: 0.35,
        }
        base_impact += style_impacts[integration_plan.integration_style]

        # Adjust based on confidence
        base_impact *= integration_plan.confidence_score

        # Adjust based on current engagement
        base_impact *= (1.0 + dialogue_context.engagement_momentum) / 2.0

        return min(0.5, max(-0.1, base_impact))

    async def _update_dialogue_context_post_integration(
        self,
        dialogue_context: DialogueContext,
        integration_plan: MemoryIntegrationPlan,
        result: DialogueEnhancementResult,
    ) -> None:
        """Update dialogue context after successful integration."""
        # Update memory reference density
        dialogue_context.memory_reference_density = (
            dialogue_context.memory_reference_density * 0.8
            + result.memories_referenced * 0.2
        )

        # Update last integration time
        dialogue_context.last_memory_integration = time.time()
        self._last_integration_times[dialogue_context.conversation_id] = time.time()

        # Update engagement momentum
        dialogue_context.engagement_momentum += result.engagement_impact

    async def resolve_memory_conflicts(
        self,
        conflicting_memories: list[dict[str, Any]],
        conversation: ConversationContext,
        npc_personality: NPCPersonality,
    ) -> dict[str, Any]:
        """Resolve conflicts between contradictory memories."""
        if len(conflicting_memories) <= 1:
            return {
                "conflict_resolution": "no_conflicts",
                "resolved_memories": conflicting_memories,
            }

        try:
            # Analyze memory conflicts
            conflicts = []
            for i, memory1 in enumerate(conflicting_memories):
                for j, memory2 in enumerate(conflicting_memories[i + 1 :], i + 1):
                    conflict_score = await self._assess_memory_conflict(
                        memory1, memory2
                    )
                    if conflict_score > 0.5:  # Significant conflict
                        conflicts.append(
                            {
                                "memory_1": memory1,
                                "memory_2": memory2,
                                "conflict_score": conflict_score,
                                "conflict_type": await self._classify_conflict_type(
                                    memory1, memory2
                                ),
                            }
                        )

            if not conflicts:
                return {
                    "conflict_resolution": "no_significant_conflicts",
                    "resolved_memories": conflicting_memories,
                }

            # Apply conflict resolution strategy
            resolution_strategy = self._select_conflict_resolution_strategy(
                conflicts, npc_personality
            )
            resolved_memories = await self._apply_conflict_resolution(
                conflicting_memories, conflicts, resolution_strategy
            )

            return {
                "conflict_resolution": resolution_strategy,
                "conflicts_found": len(conflicts),
                "resolved_memories": resolved_memories,
                "conflict_details": conflicts,
            }

        except Exception as e:
            logger.error(f"Memory conflict resolution failed: {e}")
            return {
                "conflict_resolution": "error",
                "error": str(e),
                "resolved_memories": conflicting_memories[
                    :1
                ],  # Use first memory as fallback
            }

    async def _assess_memory_conflict(
        self, memory1: dict[str, Any], memory2: dict[str, Any]
    ) -> float:
        """Assess the level of conflict between two memories."""
        conflict_score = 0.0

        # Check temporal conflicts
        if memory1.get("timestamp") and memory2.get("timestamp"):
            time_diff = abs(memory1["timestamp"] - memory2["timestamp"])
            if time_diff < 3600:  # Same hour but different content
                conflict_score += 0.3

        # Check emotional conflicts
        emotion1 = memory1.get("emotional_context", "neutral")
        emotion2 = memory2.get("emotional_context", "neutral")
        if self._are_emotions_conflicting(emotion1, emotion2):
            conflict_score += 0.4

        # Check semantic conflicts (simplified)
        text1 = memory1.get("reference_text", "")
        text2 = memory2.get("reference_text", "")
        if await self._detect_semantic_contradiction(text1, text2):
            conflict_score += 0.5

        return min(1.0, conflict_score)

    def _are_emotions_conflicting(self, emotion1: str, emotion2: str) -> bool:
        """Check if two emotions are conflicting."""
        conflicting_pairs = [
            ("happy", "sad"),
            ("excited", "bored"),
            ("angry", "calm"),
            ("worried", "confident"),
            ("frustrated", "satisfied"),
        ]

        for e1, e2 in conflicting_pairs:
            if (emotion1 == e1 and emotion2 == e2) or (
                emotion1 == e2 and emotion2 == e1
            ):
                return True
        return False

    async def _detect_semantic_contradiction(self, text1: str, text2: str) -> bool:
        """Detect semantic contradictions between two texts (simplified)."""
        # Simple contradiction detection
        contradiction_patterns = [
            ("yes", "no"),
            ("like", "dislike"),
            ("love", "hate"),
            ("agree", "disagree"),
            ("accept", "reject"),
            ("want", "don't want"),
        ]

        text1_lower = text1.lower()
        text2_lower = text2.lower()

        for pos, neg in contradiction_patterns:
            if pos in text1_lower and neg in text2_lower:
                return True
            if neg in text1_lower and pos in text2_lower:
                return True

        return False

    async def _classify_conflict_type(
        self, memory1: dict[str, Any], memory2: dict[str, Any]
    ) -> str:
        """Classify the type of conflict between memories."""
        emotion1 = memory1.get("emotional_context", "neutral")
        emotion2 = memory2.get("emotional_context", "neutral")

        if self._are_emotions_conflicting(emotion1, emotion2):
            return "emotional_conflict"

        text1 = memory1.get("reference_text", "")
        text2 = memory2.get("reference_text", "")

        if await self._detect_semantic_contradiction(text1, text2):
            return "semantic_conflict"

        time1 = memory1.get("timestamp", 0)
        time2 = memory2.get("timestamp", 0)

        if abs(time1 - time2) < 3600:  # Same timeframe
            return "temporal_conflict"

        return "general_conflict"

    def _select_conflict_resolution_strategy(
        self, conflicts: list[dict[str, Any]], npc_personality: NPCPersonality
    ) -> str:
        """Select appropriate conflict resolution strategy based on NPC personality."""
        # Personality-based resolution
        if npc_personality.get_trait_strength("wise") > 0.7:
            return "synthesis"  # Combine conflicting memories wisely
        elif npc_personality.get_trait_strength("honest") > 0.7:
            return "acknowledge_uncertainty"  # Admit confusion
        elif npc_personality.get_trait_strength("diplomatic") > 0.7:
            return "focus_on_common_ground"  # Find agreement
        else:
            return "most_recent"  # Use most recent memory

    async def _apply_conflict_resolution(
        self,
        original_memories: list[dict[str, Any]],
        conflicts: list[dict[str, Any]],
        strategy: str,
    ) -> list[dict[str, Any]]:
        """Apply conflict resolution strategy to memories."""
        if strategy == "most_recent":
            # Keep most recent memory from each conflict
            resolved = []
            conflicted_memories = set()

            for conflict in conflicts:
                mem1 = conflict["memory_1"]
                mem2 = conflict["memory_2"]

                # Compare timestamps
                if mem1.get("timestamp", 0) > mem2.get("timestamp", 0):
                    resolved.append(mem1)
                    conflicted_memories.add(id(mem2))
                else:
                    resolved.append(mem2)
                    conflicted_memories.add(id(mem1))

            # Add non-conflicted memories
            for memory in original_memories:
                if id(memory) not in conflicted_memories:
                    resolved.append(memory)

            return resolved

        elif strategy == "synthesis":
            # Create synthesized memories that acknowledge both perspectives
            synthesized = []
            processed_conflicts = set()

            for conflict in conflicts:
                if id(conflict) in processed_conflicts:
                    continue

                mem1 = conflict["memory_1"]
                mem2 = conflict["memory_2"]

                synthesized_memory = {
                    **mem1,
                    "reference_text": f"We've discussed {self._extract_key_topic(mem1['reference_text'])} multiple times with different perspectives",
                    "emotional_context": "thoughtful",
                    "synthesis": True,
                }
                synthesized.append(synthesized_memory)
                processed_conflicts.add(id(conflict))

            return synthesized

        elif strategy == "acknowledge_uncertainty":
            # Create memories that acknowledge uncertainty
            uncertain_memories = []

            for memory in original_memories:
                uncertain_memory = {
                    **memory,
                    "reference_text": f"If I recall correctly about {self._extract_key_topic(memory['reference_text'])}...",
                    "uncertainty_acknowledged": True,
                }
                uncertain_memories.append(uncertain_memory)

            return uncertain_memories[:1]  # Limit to most relevant

        else:  # focus_on_common_ground
            # Find memories with common themes
            common_themes = self._extract_common_themes(original_memories)
            if common_themes:
                return [
                    {
                        "reference_text": f"What we often discuss is {common_themes[0]}",
                        "emotional_context": "neutral",
                        "common_ground": True,
                        "similarity": 1.0,
                    }
                ]
            else:
                return original_memories[:1]

    def _extract_common_themes(self, memories: list[dict[str, Any]]) -> list[str]:
        """Extract common themes across memories."""
        all_topics = []
        for memory in memories:
            topic = self._extract_key_topic(memory.get("reference_text", ""))
            all_topics.append(topic)

        # Find most common topics
        from collections import Counter

        topic_counts = Counter(all_topics)
        return [topic for topic, count in topic_counts.most_common(3)]

    async def resolve_memory_conflicts(
        self,
        memories: list[dict[str, Any]],
        npc_personality: NPCPersonality,
        dialogue_context: DialogueContext,
    ) -> list[dict[str, Any]]:
        """Resolve conflicts between memories before integration."""
        if len(memories) <= 1:
            return memories

        # Detect conflicts
        conflicts = []
        for i, memory1 in enumerate(memories):
            for j, memory2 in enumerate(memories[i + 1 :], i + 1):
                conflict_score = await self._assess_memory_conflict(memory1, memory2)
                if conflict_score > 0.5:  # Significant conflict
                    conflicts.append((i, j, conflict_score))

        if not conflicts:
            return memories

        # Apply conflict resolution strategy
        strategy = npc_personality.speech_patterns.get(
            "conflict_resolution_strategy", "diplomatic"
        )

        if strategy == "diplomatic":
            return self._resolve_diplomatically(memories, conflicts)
        elif strategy == "selective":
            return self._resolve_selectively(memories, conflicts)
        elif strategy == "comprehensive":
            return self._resolve_comprehensively(memories, conflicts)
        else:
            # Default: keep all memories but add uncertainty markers
            return self._mark_uncertain_memories(memories, conflicts)

    def _resolve_diplomatically(
        self, memories: list[dict[str, Any]], conflicts: list[tuple[int, int, float]]
    ) -> list[dict[str, Any]]:
        """Resolve conflicts diplomatically by acknowledging uncertainty."""
        resolved_memories = []
        conflicted_indices = set()

        for i, j, score in conflicts:
            conflicted_indices.update([i, j])

        for idx, memory in enumerate(memories):
            if idx in conflicted_indices:
                # Add uncertainty marker to conflicted memories
                memory["uncertainty_marker"] = True
                memory["diplomatic_phrase"] = "if I remember correctly"
            resolved_memories.append(memory)

        return resolved_memories

    def _resolve_selectively(
        self, memories: list[dict[str, Any]], conflicts: list[tuple[int, int, float]]
    ) -> list[dict[str, Any]]:
        """Resolve conflicts by selecting the most confident memory."""
        to_remove = set()

        for i, j, score in conflicts:
            # Keep the memory with higher similarity score
            if memories[i]["similarity"] >= memories[j]["similarity"]:
                to_remove.add(j)
            else:
                to_remove.add(i)

        return [memory for idx, memory in enumerate(memories) if idx not in to_remove]

    def _resolve_comprehensively(
        self, memories: list[dict[str, Any]], conflicts: list[tuple[int, int, float]]
    ) -> list[dict[str, Any]]:
        """Resolve conflicts by combining related memories."""
        if not conflicts:
            return memories

        # For now, use diplomatic approach
        # A more sophisticated implementation would synthesize memories
        return self._resolve_diplomatically(memories, conflicts)

    def _mark_uncertain_memories(
        self, memories: list[dict[str, Any]], conflicts: list[tuple[int, int, float]]
    ) -> list[dict[str, Any]]:
        """Mark uncertain memories without removing them."""
        conflicted_indices = set()
        for i, j, score in conflicts:
            conflicted_indices.update([i, j])

        for idx, memory in enumerate(memories):
            if idx in conflicted_indices:
                memory["uncertainty_marker"] = True

        return memories

    def _determine_dialogue_state(
        self,
        conversation_history: list[ConversationExchange],
        current_topic: str | None,
        relationship_level: float,
    ) -> DialogueState:
        """Determine current dialogue state based on conversation context."""
        if not conversation_history:
            return DialogueState.OPENING

        exchange_count = len(conversation_history)

        # Opening phase: first few exchanges
        if exchange_count <= 2:
            return DialogueState.OPENING

        # Analyze recent conversation patterns
        recent_exchanges = conversation_history[-3:]

        # Look for building patterns (questions, information sharing)
        building_indicators = 0
        for exchange in recent_exchanges:
            text = exchange.message_text.lower()
            if any(
                indicator in text for indicator in ["what", "how", "why", "tell me"]
            ):
                building_indicators += 1

        # Look for deepening patterns (personal information, emotions)
        deepening_indicators = 0
        emotional_words = ["feel", "think", "believe", "remember", "experience"]
        for exchange in recent_exchanges:
            text = exchange.message_text.lower()
            if any(word in text for word in emotional_words):
                deepening_indicators += 1

        # Determine state based on patterns and relationship
        if relationship_level > 0.8 and deepening_indicators >= 2:
            return DialogueState.CLIMAX
        elif relationship_level > 0.6 and deepening_indicators >= 1:
            return DialogueState.DEEPENING
        elif building_indicators >= 1 or exchange_count <= 5:
            return DialogueState.BUILDING
        elif exchange_count > 20:
            return DialogueState.WINDING_DOWN
        else:
            return DialogueState.BUILDING

    async def _create_natural_memory_reference(
        self,
        memory: ConversationExchange,
        integration_style: IntegrationStyle,
        npc_personality: NPCPersonality,
        dialogue_context: DialogueContext,
    ) -> str:
        """Create natural memory reference based on integration style."""
        try:
            memory_dict = {
                "memory": memory,
                "reference_text": memory.message_text,
                "emotional_context": memory.emotion or "neutral",
            }

            if integration_style == IntegrationStyle.SUBTLE_HINT:
                return await self._generate_subtle_hint(memory_dict)
            elif integration_style == IntegrationStyle.DIRECT_REFERENCE:
                return await self._generate_direct_reference(memory_dict)
            elif integration_style == IntegrationStyle.EMOTIONAL_CONNECTION:
                return await self._generate_emotional_connection(memory_dict)
            elif integration_style == IntegrationStyle.COMPARATIVE:
                return await self._generate_comparative_reference(memory_dict)
            elif integration_style == IntegrationStyle.NARRATIVE_WEAVING:
                return await self._generate_narrative_weaving([memory_dict])
            else:
                # Default to direct reference
                return await self._generate_direct_reference(memory_dict)

        except Exception as e:
            logger.error(f"Error creating natural memory reference: {e}")
            return "that reminds me of something"

    def _weave_memory_into_response(
        self,
        base_response: str,
        memory_reference: str,
        timing: MemoryIntegrationTiming,
        style: IntegrationStyle,
    ) -> str:
        """Weave memory reference into response based on timing and style."""
        try:
            if timing == MemoryIntegrationTiming.IMMEDIATE:
                return f"{memory_reference}, {base_response.lower()}"
            elif timing == MemoryIntegrationTiming.NATURAL_PAUSE:
                sentences = base_response.split(". ")
                if len(sentences) > 1:
                    # Insert after first sentence
                    return f"{sentences[0]}. {memory_reference}, {'. '.join(sentences[1:])}"
                else:
                    return f"{base_response} {memory_reference}."
            elif timing == MemoryIntegrationTiming.TOPIC_BRIDGE:
                return f"{memory_reference} - {base_response.lower()}"
            elif timing == MemoryIntegrationTiming.RESPONSE_ENHANCEMENT:
                return f"{base_response} {memory_reference}."
            elif timing == MemoryIntegrationTiming.DELAYED:
                return f"{base_response} Oh, and {memory_reference.lower()}."
            else:
                # Default
                return f"{base_response} {memory_reference}."

        except Exception as e:
            logger.error(f"Error weaving memory into response: {e}")
            return f"{base_response} {memory_reference}"

    async def _integrate_immediately(
        self,
        base_response: str,
        integration_plan: MemoryIntegrationPlan,
        dialogue_context: DialogueContext,
        npc_personality: NPCPersonality,
    ) -> str:
        """Integrate memories immediately at start of response."""
        if not integration_plan.memory_references:
            return base_response

        memory_ref = integration_plan.memory_references[0]
        memory_text = await self._create_natural_memory_reference(
            memory_ref["memory"],
            integration_plan.integration_style,
            npc_personality,
            dialogue_context,
        )

        return f"{memory_text}, {base_response.lower()}"

    async def _integrate_at_natural_pause(
        self,
        base_response: str,
        integration_plan: MemoryIntegrationPlan,
        dialogue_context: DialogueContext,
        npc_personality: NPCPersonality,
    ) -> str:
        """Integrate memories at natural pause in conversation."""
        if not integration_plan.memory_references:
            return base_response

        memory_ref = integration_plan.memory_references[0]
        memory_text = await self._create_natural_memory_reference(
            memory_ref["memory"],
            integration_plan.integration_style,
            npc_personality,
            dialogue_context,
        )

        # Insert at natural pause (after first sentence)
        sentences = base_response.split(". ")
        if len(sentences) > 1:
            return f"{sentences[0]}. {memory_text}, {'. '.join(sentences[1:])}"
        else:
            return f"{base_response} {memory_text}."

    async def _integrate_as_topic_bridge(
        self,
        base_response: str,
        integration_plan: MemoryIntegrationPlan,
        dialogue_context: DialogueContext,
        npc_personality: NPCPersonality,
    ) -> str:
        """Use memory as bridge between topics."""
        if not integration_plan.memory_references:
            return base_response

        memory_ref = integration_plan.memory_references[0]
        memory_text = await self._create_natural_memory_reference(
            memory_ref["memory"],
            integration_plan.integration_style,
            npc_personality,
            dialogue_context,
        )

        return f"{memory_text} - {base_response.lower()}"

    async def _enhance_existing_response(
        self,
        base_response: str,
        integration_plan: MemoryIntegrationPlan,
        dialogue_context: DialogueContext,
        npc_personality: NPCPersonality,
    ) -> str:
        """Enhance existing response with memory context."""
        if not integration_plan.memory_references:
            return base_response

        memory_ref = integration_plan.memory_references[0]
        memory_text = await self._create_natural_memory_reference(
            memory_ref["memory"],
            integration_plan.integration_style,
            npc_personality,
            dialogue_context,
        )

        return f"{base_response} {memory_text}."

    async def _validate_response_naturalness(
        self,
        base_response: str,
        enhanced_response: str,
        dialogue_context: DialogueContext,
    ) -> float:
        """Validate naturalness of enhanced response."""
        # Simple heuristic - could be enhanced with LLM evaluation
        length_ratio = len(enhanced_response) / len(base_response)

        # Be more lenient with length ratios for memory integration
        if length_ratio > 3.0:
            return 0.3
        elif length_ratio > 2.5:
            return 0.5
        elif length_ratio > 2.0:
            return 0.7  # Still acceptable for memory integration
        elif length_ratio > 1.5:
            return 0.8

        # Check for awkward transitions
        if " - " in enhanced_response and enhanced_response.count(" - ") > 1:
            return 0.4

        # Bonus for reasonable length increase (indicates good integration)
        if 1.2 <= length_ratio <= 2.0:
            return 0.85

        # Default reasonable naturalness
        return 0.8

    async def _assess_topic_transition_naturalness(
        self, last_topic: str, current_topic: str, player_message: str
    ) -> float:
        """Assess how natural a topic transition is."""
        # Simple heuristic - could use LLM for better assessment
        if last_topic == current_topic:
            return 1.0

        # Check for related topics
        related_pairs = [
            ("weather", "outdoors"),
            ("weather", "activities"),
            ("food", "cooking"),
            ("travel", "places"),
        ]

        for topic1, topic2 in related_pairs:
            if (topic1 in last_topic.lower() and topic2 in current_topic.lower()) or (
                topic2 in last_topic.lower() and topic1 in current_topic.lower()
            ):
                return 0.8

        # Check for question-based transitions
        if "?" in player_message:
            return 0.7

        return 0.5

    async def _generate_integration_text(
        self,
        memory_references: list[dict[str, Any]],
        integration_style: IntegrationStyle,
        dialogue_context: DialogueContext,
    ) -> str:
        """Generate integration text from memory references."""
        if not memory_references:
            return ""

        memory_ref = memory_references[0]

        if integration_style == IntegrationStyle.SUBTLE_HINT:
            return await self._generate_subtle_hint(memory_ref)
        elif integration_style == IntegrationStyle.DIRECT_REFERENCE:
            return await self._generate_direct_reference(memory_ref)
        elif integration_style == IntegrationStyle.EMOTIONAL_CONNECTION:
            return await self._generate_emotional_connection(memory_ref)
        else:
            return await self._generate_direct_reference(memory_ref)
