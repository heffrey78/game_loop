"""Integration of emotional memory context with dialogue system and personality archetypes."""

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from game_loop.core.conversation.conversation_models import (
    ConversationContext,
    ConversationExchange,
    NPCPersonality,
)
from game_loop.core.conversation.memory_integration import (
    ConversationFlowState,
    MemoryContext,
    MemoryDisclosureLevel,
    MemoryIntegrationInterface,
    MemoryRetrievalResult,
)
from game_loop.database.session_factory import DatabaseSessionFactory
from game_loop.llm.ollama.client import OllamaClient

from .affective_weighting import AffectiveMemoryWeightingEngine
from .config import MemoryAlgorithmConfig
from .constants import EmotionalThresholds
from .emotional_clustering import EmotionalMemoryClusteringEngine
from .emotional_context import EmotionalMemoryContextEngine, EmotionalSignificance, MoodState
from .emotional_preservation import EmotionalMemoryRecord, EmotionalPreservationEngine, EmotionalRetrievalQuery
from .exceptions import (
    DialogueIntegrationError, InvalidEmotionalDataError, 
    handle_emotional_memory_error
)
from .mood_memory_engine import MoodDependentMemoryEngine, MoodStateRecord
from .trauma_protection import TraumaProtectionEngine, TraumaAccessRequest
from .validation import (
    validate_uuid, validate_probability, validate_string_content, 
    validate_mood_state, default_validator
)

logger = logging.getLogger(__name__)


class EmotionalResponseMode(Enum):
    """Modes for emotional memory-informed responses."""
    
    EMPATHETIC = "empathetic"               # Deep emotional understanding and resonance
    SUPPORTIVE = "supportive"               # Providing emotional support and comfort
    REFLECTIVE = "reflective"               # Helping process and understand emotions
    PROTECTIVE = "protective"               # Protecting from emotional harm
    THERAPEUTIC = "therapeutic"             # Therapeutic processing and healing
    NEUTRAL = "neutral"                     # Emotionally neutral, factual responses
    AVOIDANT = "avoidant"                   # Avoiding emotional content


class DialogueEmotionalState(Enum):
    """Emotional states that affect dialogue generation."""
    
    EMOTIONALLY_ENGAGED = "engaged"         # Actively processing emotions
    EMOTIONALLY_OVERWHELMED = "overwhelmed" # Too much emotional content
    EMOTIONALLY_WITHDRAWN = "withdrawn"     # Pulling back from emotions
    EMOTIONALLY_CURIOUS = "curious"        # Exploring emotional content
    EMOTIONALLY_RESISTANT = "resistant"    # Resisting emotional processing
    EMOTIONALLY_INTEGRATED = "integrated"  # Well-integrated emotional state


@dataclass
class EmotionalDialogueContext:
    """Context for emotionally-informed dialogue generation."""
    
    npc_id: str
    conversation_id: str
    
    # Current emotional state
    current_mood: MoodState
    mood_intensity: float = EmotionalThresholds.MODERATE_SIGNIFICANCE
    emotional_stability: float = EmotionalThresholds.HIGH_MOOD_ACCESSIBILITY
    
    # Memory context
    accessible_emotional_memories: List[EmotionalMemoryRecord] = field(default_factory=list)
    triggered_memory_clusters: List[str] = field(default_factory=list)
    emotional_associations: List[Tuple[str, float]] = field(default_factory=list)
    
    # Dialogue state
    emotional_response_mode: EmotionalResponseMode = EmotionalResponseMode.NEUTRAL
    dialogue_emotional_state: DialogueEmotionalState = DialogueEmotionalState.EMOTIONALLY_ENGAGED
    disclosure_level: MemoryDisclosureLevel = MemoryDisclosureLevel.NONE
    
    # Safety and protection
    trauma_protection_active: bool = False
    safety_protocols_enabled: bool = False
    therapeutic_context: bool = False
    
    # Response guidance
    emotional_themes: List[str] = field(default_factory=list)
    recommended_tone: str = "neutral"
    avoid_topics: List[str] = field(default_factory=list)
    supportive_elements: List[str] = field(default_factory=list)
    
    # Context metadata
    created_at: float = field(default_factory=time.time)
    confidence: float = EmotionalThresholds.HIGH_SIGNIFICANCE_THRESHOLD


@dataclass
class EmotionalResponseGuidance:
    """Guidance for generating emotionally-informed responses."""
    
    # Core response characteristics
    emotional_tone: str                     # Primary emotional tone to use
    empathy_level: float                   # 0.0-1.0 level of empathy to show
    vulnerability_sharing: float           # 0.0-1.0 how much vulnerability to share
    emotional_validation: bool             # Whether to validate player emotions
    
    # Memory integration
    memory_references: List[str] = field(default_factory=list)  # Memories to reference
    emotional_insights: List[str] = field(default_factory=list) # Insights to share
    therapeutic_elements: List[str] = field(default_factory=list) # Therapeutic responses
    
    # Response structure
    opening_approach: str = "neutral"      # How to open the response
    core_message_type: str = "informative" # Type of core message
    emotional_bridge: Optional[str] = None  # Bridge to emotional content
    closing_approach: str = "supportive"   # How to close the response
    
    # Safety considerations
    trigger_warnings: List[str] = field(default_factory=list)
    protective_measures: List[str] = field(default_factory=list)
    safety_checks: List[str] = field(default_factory=list)
    
    # Personality expression
    personality_emphasis: List[str] = field(default_factory=list) # Traits to emphasize
    emotional_archetype: str = "balanced"   # Emotional archetype to express
    response_complexity: str = "moderate"   # Simple, moderate, or complex response


class EmotionalDialogueIntegrationEngine:
    """Engine for integrating emotional memory context with dialogue generation."""
    
    def __init__(
        self,
        session_factory: DatabaseSessionFactory,
        llm_client: OllamaClient,
        config: MemoryAlgorithmConfig,
        emotional_context_engine: EmotionalMemoryContextEngine,
        affective_engine: AffectiveMemoryWeightingEngine,
        preservation_engine: EmotionalPreservationEngine,
        clustering_engine: EmotionalMemoryClusteringEngine,
        mood_engine: MoodDependentMemoryEngine,
        trauma_engine: TraumaProtectionEngine,
        memory_integration: MemoryIntegrationInterface,
    ):
        self.session_factory = session_factory
        self.llm_client = llm_client
        self.config = config
        self.emotional_context_engine = emotional_context_engine
        self.affective_engine = affective_engine
        self.preservation_engine = preservation_engine
        self.clustering_engine = clustering_engine
        self.mood_engine = mood_engine
        self.trauma_engine = trauma_engine
        self.memory_integration = memory_integration
        
        # Integration state tracking
        self._dialogue_contexts: Dict[str, EmotionalDialogueContext] = {}
        self._response_history: Dict[str, List[EmotionalResponseGuidance]] = {}
        
        # Performance tracking
        self._performance_stats = {
            "dialogue_integrations": 0,
            "emotional_responses_generated": 0,
            "trauma_protections_activated": 0,
            "therapeutic_interventions": 0,
            "memory_references_made": 0,
            "avg_integration_time_ms": 0.0,
        }

    async def create_emotional_dialogue_context(
        self,
        npc_id: uuid.UUID,
        conversation_context: ConversationContext,
        player_message: str,
        personality: NPCPersonality,
    ) -> EmotionalDialogueContext:
        """Create comprehensive emotional dialogue context."""
        try:
            # Validate inputs
            if not npc_id:
                raise InvalidEmotionalDataError("NPC ID is required")
            if not conversation_context:
                raise InvalidEmotionalDataError("Conversation context is required")
            if not personality:
                raise InvalidEmotionalDataError("NPC personality is required")
            
            # Validate and sanitize inputs
            npc_id_str = str(validate_uuid(npc_id, "npc_id"))
            player_message = validate_string_content(
                player_message, "player_message", min_length=1, max_length=5000
            ) if player_message else ""
            
        except Exception as e:
            raise handle_emotional_memory_error(
                e, "Failed to validate inputs for emotional dialogue context creation"
            )
        
        start_time = time.perf_counter()
        
        try:
            # Get current mood state
            current_mood_record = self.mood_engine.get_current_mood(npc_id)
            current_mood = current_mood_record.mood_state if current_mood_record else MoodState.NEUTRAL
            
            # Analyze emotional significance of the player's message
            last_exchange = conversation_context.get_recent_exchanges(1)
            if last_exchange:
                exchange = last_exchange[0]
                # Create a mock exchange for the player message
                mock_exchange = ConversationExchange(
                    exchange_id=uuid.uuid4(),
                    conversation_id=conversation_context.conversation_id,
                    speaker_id=conversation_context.player_id,
                    message_text=player_message,
                    message_type="statement",
                    timestamp=datetime.now(timezone.utc),
                )
                
                emotional_significance = await self.emotional_context_engine.analyze_emotional_significance(
                    mock_exchange, conversation_context, personality
                )
            else:
                emotional_significance = None
            
            # Get mood-adjusted memories
            retrieval_query = EmotionalRetrievalQuery(
                target_mood=current_mood,
                significance_threshold=EmotionalThresholds.MODERATE_SIGNIFICANCE - 0.1,  # 0.4
                max_results=8,
                trust_level=conversation_context.relationship_level,
            )
        
        accessible_memories, mood_context = await self.mood_engine.get_mood_adjusted_memories(
            npc_id, retrieval_query, include_therapeutic=False
        )
        
        # Check for trauma protection needs
        trauma_protection_needed = False
        if emotional_significance:
            trauma_request = TraumaAccessRequest(
                npc_id=npc_id_str,
                requesting_context="dialogue_generation",
                trauma_memory_ids=[m.exchange_id for m in accessible_memories 
                                 if m.emotional_significance.emotional_type.value == "traumatic"],
                trust_level=float(conversation_context.relationship_level),
                therapeutic_intent=False,
            )
            
            if trauma_request.trauma_memory_ids:
                trauma_decision = await self.trauma_engine.evaluate_trauma_access_request(
                    npc_id, trauma_request, personality, current_mood
                )
                trauma_protection_needed = not trauma_decision.access_granted
                
                # Filter out denied trauma memories
                accessible_memories = [
                    m for m in accessible_memories 
                    if m.exchange_id not in trauma_decision.denied_memories
                ]
        
        # Get emotional associations
        emotional_associations = []
        if accessible_memories:
            for memory in accessible_memories[:3]:  # Top 3 memories
                associations = await self.clustering_engine.get_associated_memories(
                    npc_id, memory.exchange_id, max_associations=2
                )
                emotional_associations.extend(associations)
        
        # Determine emotional response mode
        response_mode = await self._determine_emotional_response_mode(
            emotional_significance, current_mood_record, personality, conversation_context
        )
        
        # Determine dialogue emotional state
        dialogue_state = self._determine_dialogue_emotional_state(
            current_mood_record, accessible_memories, trauma_protection_needed
        )
        
        # Extract emotional themes
        emotional_themes = self._extract_emotional_themes(accessible_memories, emotional_significance)
        
        # Create dialogue context
        dialogue_context = EmotionalDialogueContext(
            npc_id=npc_id_str,
            conversation_id=str(conversation_context.conversation_id),
            current_mood=current_mood,
            mood_intensity=current_mood_record.intensity if current_mood_record else EmotionalThresholds.MODERATE_SIGNIFICANCE,
            emotional_stability=current_mood_record.stability if current_mood_record else EmotionalThresholds.HIGH_MOOD_ACCESSIBILITY,
            accessible_emotional_memories=accessible_memories,
            emotional_associations=emotional_associations,
            emotional_response_mode=response_mode,
            dialogue_emotional_state=dialogue_state,
            disclosure_level=MemoryDisclosureLevel.SUBTLE_HINTS,  # Default conservative
            trauma_protection_active=trauma_protection_needed,
            emotional_themes=emotional_themes,
            recommended_tone=self._determine_recommended_tone(current_mood, response_mode),
        )
        
            # Store context
            context_key = f"{npc_id_str}_{conversation_context.conversation_id}"
            self._dialogue_contexts[context_key] = dialogue_context
            
            # Update performance stats
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            self._performance_stats["dialogue_integrations"] += 1
            self._update_avg_integration_time(processing_time_ms)
            
            logger.debug(f"Created emotional dialogue context for NPC {npc_id}")
            return dialogue_context
            
        except Exception as e:
            raise DialogueIntegrationError(
                f"Failed to create emotional dialogue context for NPC {npc_id}",
                npc_id=npc_id_str,
                conversation_id=str(conversation_context.conversation_id),
                integration_stage="context_creation"
            )

    async def generate_emotional_response_guidance(
        self,
        dialogue_context: EmotionalDialogueContext,
        conversation_context: ConversationContext,
        personality: NPCPersonality,
        response_intent: str = "conversational",
    ) -> EmotionalResponseGuidance:
        """Generate guidance for emotionally-informed responses."""
        
        # Determine core response characteristics
        emotional_tone = await self._determine_emotional_tone(
            dialogue_context, personality, response_intent
        )
        
        empathy_level = self._calculate_empathy_level(dialogue_context, personality)
        vulnerability_sharing = self._calculate_vulnerability_sharing(
            dialogue_context, personality, conversation_context
        )
        
        # Memory integration decisions
        memory_references = await self._select_memory_references(
            dialogue_context, personality, response_intent
        )
        
        emotional_insights = self._generate_emotional_insights(
            dialogue_context, memory_references, personality
        )
        
        # Response structure guidance
        opening_approach = self._determine_opening_approach(dialogue_context, personality)
        core_message_type = self._determine_core_message_type(dialogue_context, response_intent)
        closing_approach = self._determine_closing_approach(dialogue_context, personality)
        
        # Safety and protection considerations
        trigger_warnings, protective_measures = self._assess_safety_considerations(
            dialogue_context, memory_references
        )
        
        # Personality expression guidance
        personality_emphasis = self._select_personality_emphasis(
            dialogue_context, personality, response_intent
        )
        
        emotional_archetype = self._determine_emotional_archetype(personality, dialogue_context)
        
        # Create response guidance
        guidance = EmotionalResponseGuidance(
            emotional_tone=emotional_tone,
            empathy_level=empathy_level,
            vulnerability_sharing=vulnerability_sharing,
            emotional_validation=dialogue_context.emotional_response_mode in [
                EmotionalResponseMode.EMPATHETIC, EmotionalResponseMode.SUPPORTIVE
            ],
            memory_references=memory_references,
            emotional_insights=emotional_insights,
            opening_approach=opening_approach,
            core_message_type=core_message_type,
            closing_approach=closing_approach,
            trigger_warnings=trigger_warnings,
            protective_measures=protective_measures,
            personality_emphasis=personality_emphasis,
            emotional_archetype=emotional_archetype,
        )
        
        # Add therapeutic elements if appropriate
        if dialogue_context.therapeutic_context or dialogue_context.emotional_response_mode == EmotionalResponseMode.THERAPEUTIC:
            guidance.therapeutic_elements = await self._generate_therapeutic_elements(
                dialogue_context, personality
            )
        
        # Store guidance in history
        npc_key = dialogue_context.npc_id
        if npc_key not in self._response_history:
            self._response_history[npc_key] = []
        self._response_history[npc_key].append(guidance)
        
        # Limit history size
        if len(self._response_history[npc_key]) > 20:
            self._response_history[npc_key] = self._response_history[npc_key][-10:]
        
        self._performance_stats["emotional_responses_generated"] += 1
        if memory_references:
            self._performance_stats["memory_references_made"] += len(memory_references)
        
        return guidance

    async def update_emotional_state_from_response(
        self,
        npc_id: uuid.UUID,
        dialogue_context: EmotionalDialogueContext,
        generated_response: str,
        player_reaction_predicted: str,
    ) -> None:
        """Update emotional state based on generated response and predicted reaction."""
        
        # Analyze emotional impact of the response
        response_emotional_impact = await self._analyze_response_emotional_impact(
            generated_response, dialogue_context
        )
        
        # Predict mood changes from the interaction
        predicted_mood_changes = await self._predict_mood_changes(
            dialogue_context, response_emotional_impact, player_reaction_predicted
        )
        
        # Update mood if significant change predicted
        if abs(predicted_mood_changes.get("intensity_change", 0)) > 0.1:
            new_intensity = max(0.0, min(1.0, 
                dialogue_context.mood_intensity + predicted_mood_changes["intensity_change"]
            ))
            
            # Update mood through mood engine
            await self.mood_engine.update_npc_mood(
                npc_id,
                dialogue_context.current_mood,
                new_intensity,
                "dialogue_interaction",
                conversation_context=None  # Would need to pass if available
            )
        
        # Update dialogue emotional state based on response
        new_dialogue_state = self._calculate_post_response_emotional_state(
            dialogue_context, response_emotional_impact
        )
        dialogue_context.dialogue_emotional_state = new_dialogue_state
        
        logger.debug(f"Updated emotional state for NPC {npc_id} after response generation")

    async def handle_emotional_memory_integration(
        self,
        npc_id: uuid.UUID,
        new_exchange: ConversationExchange,
        conversation_context: ConversationContext,
        personality: NPCPersonality,
    ) -> Optional[EmotionalMemoryRecord]:
        """Handle integration of new emotional memory from conversation."""
        
        # Analyze emotional significance of the new exchange
        emotional_significance = await self.emotional_context_engine.analyze_emotional_significance(
            new_exchange, conversation_context, personality
        )
        
        # Only create emotional memory records for significant exchanges
        if emotional_significance.overall_significance < 0.3:
            return None
        
        # Calculate affective weight
        current_mood_record = self.mood_engine.get_current_mood(npc_id)
        current_mood = current_mood_record.mood_state if current_mood_record else MoodState.NEUTRAL
        
        affective_weight = await self.affective_engine.calculate_affective_weight(
            emotional_significance,
            personality,
            current_mood,
            float(conversation_context.relationship_level),
        )
        
        # Create and preserve emotional memory record
        emotional_record = await self.preservation_engine.preserve_emotional_context(
            new_exchange,
            emotional_significance,
            affective_weight,
        )
        
        # Update clustering if this is a significant memory
        if emotional_significance.overall_significance > 0.6:
            # Get all memories for re-clustering
            all_memories_query = EmotionalRetrievalQuery(
                significance_threshold=0.2,
                max_results=100,
                trust_level=1.0,  # Get all memories for clustering
            )
            
            all_memories_result = await self.preservation_engine.retrieve_emotional_memories(
                npc_id, all_memories_query
            )
            
            if len(all_memories_result.emotional_records) >= 5:  # Enough for clustering
                await self.clustering_engine.update_emotional_network(
                    npc_id, all_memories_result.emotional_records, personality, incremental=True
                )
        
        logger.debug(f"Integrated emotional memory for exchange {new_exchange.exchange_id}")
        return emotional_record

    async def _determine_emotional_response_mode(
        self,
        emotional_significance: Optional[EmotionalSignificance],
        current_mood_record: Optional[MoodStateRecord],
        personality: NPCPersonality,
        conversation_context: ConversationContext,
    ) -> EmotionalResponseMode:
        """Determine the appropriate emotional response mode."""
        
        # Default to neutral
        if not emotional_significance or not current_mood_record:
            return EmotionalResponseMode.NEUTRAL
        
        # Check for trauma content - use protective mode
        if emotional_significance.emotional_type.value == "traumatic":
            return EmotionalResponseMode.PROTECTIVE
        
        # High emotional significance with supportive personality
        if (emotional_significance.overall_significance > 0.7 and 
            personality.get_trait_strength("supportive") > 0.6):
            return EmotionalResponseMode.SUPPORTIVE
        
        # High empathy personalities use empathetic mode
        empathy_score = personality.get_trait_strength("empathetic")
        if empathy_score > 0.7 and emotional_significance.intensity_score > 0.5:
            return EmotionalResponseMode.EMPATHETIC
        
        # Therapeutic personalities in emotional situations
        therapeutic_score = personality.get_trait_strength("therapeutic")
        if therapeutic_score > 0.6 and emotional_significance.formative_influence > 0.5:
            return EmotionalResponseMode.THERAPEUTIC
        
        # Reflective personalities with formative experiences
        reflective_score = personality.get_trait_strength("reflective")
        if reflective_score > 0.6 and emotional_significance.emotional_type.value == "formative":
            return EmotionalResponseMode.REFLECTIVE
        
        # Avoidant personalities or low trust
        if (personality.get_trait_strength("conflict_averse") > 0.7 or 
            conversation_context.relationship_level < 0.3):
            return EmotionalResponseMode.AVOIDANT
        
        # Default to supportive for moderate emotional content
        if emotional_significance.overall_significance > 0.5:
            return EmotionalResponseMode.SUPPORTIVE
        
        return EmotionalResponseMode.NEUTRAL

    def _determine_dialogue_emotional_state(
        self,
        current_mood_record: Optional[MoodStateRecord],
        accessible_memories: List[EmotionalMemoryRecord],
        trauma_protection_needed: bool,
    ) -> DialogueEmotionalState:
        """Determine current dialogue emotional state."""
        
        if trauma_protection_needed:
            return DialogueEmotionalState.EMOTIONALLY_WITHDRAWN
        
        if not current_mood_record:
            return DialogueEmotionalState.EMOTIONALLY_ENGAGED
        
        # Very unstable mood = overwhelmed
        if current_mood_record.stability < 0.3:
            return DialogueEmotionalState.EMOTIONALLY_OVERWHELMED
        
        # High intensity negative moods = withdrawn
        if (current_mood_record.intensity > 0.8 and 
            current_mood_record.mood_state in [MoodState.FEARFUL, MoodState.ANXIOUS, MoodState.MELANCHOLY]):
            return DialogueEmotionalState.EMOTIONALLY_WITHDRAWN
        
        # Many accessible emotional memories = engaged
        emotional_memory_count = sum(1 for m in accessible_memories if m.emotional_significance.intensity_score > 0.5)
        if emotional_memory_count > 3:
            return DialogueEmotionalState.EMOTIONALLY_ENGAGED
        
        # Moderate emotional content = curious
        if emotional_memory_count > 0:
            return DialogueEmotionalState.EMOTIONALLY_CURIOUS
        
        # High stability and positive mood = integrated
        if (current_mood_record.stability > 0.8 and 
            current_mood_record.mood_state in [MoodState.CONTENT, MoodState.JOYFUL]):
            return DialogueEmotionalState.EMOTIONALLY_INTEGRATED
        
        return DialogueEmotionalState.EMOTIONALLY_ENGAGED

    def _extract_emotional_themes(
        self,
        accessible_memories: List[EmotionalMemoryRecord],
        current_significance: Optional[EmotionalSignificance],
    ) -> List[str]:
        """Extract emotional themes from accessible memories and current context."""
        
        themes = []
        
        # Add themes from current emotional significance
        if current_significance:
            themes.append(current_significance.emotional_type.value)
            
            # Add intensity-based themes
            if current_significance.intensity_score > 0.8:
                themes.append("high_intensity_emotion")
            elif current_significance.intensity_score > 0.6:
                themes.append("moderate_emotion")
            
            # Add relationship themes
            if abs(current_significance.relationship_impact) > 0.7:
                themes.append("relationship_significant")
        
        # Add themes from accessible memories
        memory_types = [m.emotional_significance.emotional_type.value for m in accessible_memories]
        type_counts = {}
        for mem_type in memory_types:
            type_counts[mem_type] = type_counts.get(mem_type, 0) + 1
        
        # Add dominant memory types as themes
        for mem_type, count in type_counts.items():
            if count >= 2:  # At least 2 memories of this type
                themes.append(f"recurring_{mem_type}")
        
        # Add protection level themes
        high_protection_count = sum(
            1 for m in accessible_memories 
            if m.emotional_significance.protection_level.value in ["protected", "traumatic"]
        )
        if high_protection_count > 0:
            themes.append("sensitive_content")
        
        return list(set(themes))  # Remove duplicates

    def _determine_recommended_tone(self, mood: MoodState, response_mode: EmotionalResponseMode) -> str:
        """Determine recommended tone based on mood and response mode."""
        
        # Response mode overrides
        if response_mode == EmotionalResponseMode.EMPATHETIC:
            return "empathetic"
        elif response_mode == EmotionalResponseMode.SUPPORTIVE:
            return "supportive"
        elif response_mode == EmotionalResponseMode.THERAPEUTIC:
            return "therapeutic"
        elif response_mode == EmotionalResponseMode.PROTECTIVE:
            return "protective"
        elif response_mode == EmotionalResponseMode.AVOIDANT:
            return "neutral"
        
        # Mood-based tones
        mood_tones = {
            MoodState.JOYFUL: "cheerful",
            MoodState.CONTENT: "warm",
            MoodState.NEUTRAL: "neutral",
            MoodState.MELANCHOLY: "gentle",
            MoodState.ANXIOUS: "calming",
            MoodState.ANGRY: "understanding",
            MoodState.FEARFUL: "reassuring",
            MoodState.EXCITED: "enthusiastic",
            MoodState.NOSTALGIC: "reflective",
            MoodState.HOPEFUL: "encouraging",
        }
        
        return mood_tones.get(mood, "neutral")

    async def _determine_emotional_tone(
        self,
        dialogue_context: EmotionalDialogueContext,
        personality: NPCPersonality,
        response_intent: str,
    ) -> str:
        """Determine specific emotional tone for response."""
        
        base_tone = dialogue_context.recommended_tone
        
        # Adjust based on personality traits
        if personality.get_trait_strength("warm") > 0.7:
            if base_tone in ["neutral", "gentle"]:
                base_tone = "warm"
        
        if personality.get_trait_strength("analytical") > 0.7:
            if base_tone in ["warm", "cheerful"]:
                base_tone = "thoughtful"
        
        if personality.get_trait_strength("playful") > 0.7:
            if base_tone in ["neutral", "warm"]:
                base_tone = "playful"
        
        # Adjust based on dialogue emotional state
        if dialogue_context.dialogue_emotional_state == DialogueEmotionalState.EMOTIONALLY_OVERWHELMED:
            return "gentle"
        elif dialogue_context.dialogue_emotional_state == DialogueEmotionalState.EMOTIONALLY_WITHDRAWN:
            return "patient"
        elif dialogue_context.dialogue_emotional_state == DialogueEmotionalState.EMOTIONALLY_CURIOUS:
            return "encouraging"
        
        # Adjust for trauma protection
        if dialogue_context.trauma_protection_active:
            return "protective"
        
        return base_tone

    def _calculate_empathy_level(
        self, dialogue_context: EmotionalDialogueContext, personality: NPCPersonality
    ) -> float:
        """Calculate appropriate level of empathy to display."""
        
        base_empathy = personality.get_trait_strength("empathetic")
        
        # Adjust based on emotional response mode
        mode_adjustments = {
            EmotionalResponseMode.EMPATHETIC: 0.3,
            EmotionalResponseMode.SUPPORTIVE: 0.2,
            EmotionalResponseMode.THERAPEUTIC: 0.1,
            EmotionalResponseMode.REFLECTIVE: 0.0,
            EmotionalResponseMode.PROTECTIVE: -0.1,
            EmotionalResponseMode.NEUTRAL: -0.2,
            EmotionalResponseMode.AVOIDANT: -0.4,
        }
        
        adjustment = mode_adjustments.get(dialogue_context.emotional_response_mode, 0.0)
        
        # Adjust based on emotional themes
        if "high_intensity_emotion" in dialogue_context.emotional_themes:
            adjustment += 0.2
        if "sensitive_content" in dialogue_context.emotional_themes:
            adjustment += 0.1
        
        # Trauma protection reduces empathy display
        if dialogue_context.trauma_protection_active:
            adjustment -= 0.2
        
        final_empathy = max(0.0, min(1.0, base_empathy + adjustment))
        return final_empathy

    def _calculate_vulnerability_sharing(
        self,
        dialogue_context: EmotionalDialogueContext,
        personality: NPCPersonality,
        conversation_context: ConversationContext,
    ) -> float:
        """Calculate appropriate level of vulnerability sharing."""
        
        base_openness = personality.get_trait_strength("open")
        trust_level = float(conversation_context.relationship_level)
        
        # Trust level strongly affects vulnerability sharing
        trust_factor = trust_level * 0.8
        
        # Personality factors
        vulnerability_willingness = (
            personality.get_trait_strength("vulnerable") * 0.3 +
            personality.get_trait_strength("authentic") * 0.2 +
            base_openness * 0.2
        )
        
        # Mode adjustments
        if dialogue_context.emotional_response_mode == EmotionalResponseMode.EMPATHETIC:
            vulnerability_willingness += 0.2
        elif dialogue_context.emotional_response_mode == EmotionalResponseMode.AVOIDANT:
            vulnerability_willingness -= 0.3
        elif dialogue_context.emotional_response_mode == EmotionalResponseMode.PROTECTIVE:
            vulnerability_willingness -= 0.2
        
        # Trauma protection heavily reduces vulnerability
        if dialogue_context.trauma_protection_active:
            vulnerability_willingness *= 0.3
        
        # Emotional stability affects willingness to be vulnerable
        stability_factor = dialogue_context.emotional_stability * 0.3
        
        final_vulnerability = max(0.0, min(1.0, 
            trust_factor + vulnerability_willingness + stability_factor
        ))
        
        return final_vulnerability

    async def _select_memory_references(
        self,
        dialogue_context: EmotionalDialogueContext,
        personality: NPCPersonality,
        response_intent: str,
    ) -> List[str]:
        """Select which memories to reference in the response."""
        
        if dialogue_context.emotional_response_mode == EmotionalResponseMode.AVOIDANT:
            return []
        
        if dialogue_context.trauma_protection_active:
            return []  # No memory references when trauma protection is active
        
        # Select memories based on disclosure level
        max_memories = {
            MemoryDisclosureLevel.NONE: 0,
            MemoryDisclosureLevel.SUBTLE_HINTS: 1,
            MemoryDisclosureLevel.DIRECT_REFERENCES: 2,
            MemoryDisclosureLevel.DETAILED_MEMORIES: 3,
        }.get(dialogue_context.disclosure_level, 0)
        
        if max_memories == 0:
            return []
        
        # Filter memories by appropriateness
        appropriate_memories = []
        for memory in dialogue_context.accessible_emotional_memories:
            # Skip traumatic memories unless therapeutic context
            if (memory.emotional_significance.emotional_type.value == "traumatic" and
                not dialogue_context.therapeutic_context):
                continue
            
            # Skip highly protected memories in casual context
            if (memory.emotional_significance.protection_level.value in ["protected", "traumatic"] and
                response_intent == "conversational"):
                continue
            
            # Include memories that match current emotional themes
            if memory.emotional_significance.emotional_type.value in dialogue_context.emotional_themes:
                appropriate_memories.append(memory.exchange_id)
            
            # Include moderately significant memories
            elif memory.emotional_significance.overall_significance > 0.5:
                appropriate_memories.append(memory.exchange_id)
        
        # Limit to max_memories
        return appropriate_memories[:max_memories]

    def _generate_emotional_insights(
        self,
        dialogue_context: EmotionalDialogueContext,
        memory_references: List[str],
        personality: NPCPersonality,
    ) -> List[str]:
        """Generate emotional insights to share."""
        
        insights = []
        
        # Don't generate insights in protective or avoidant modes
        if dialogue_context.emotional_response_mode in [
            EmotionalResponseMode.PROTECTIVE, EmotionalResponseMode.AVOIDANT
        ]:
            return insights
        
        # Generate insights based on emotional themes
        if "recurring_traumatic" in dialogue_context.emotional_themes and dialogue_context.therapeutic_context:
            insights.append("trauma_pattern_recognition")
        
        if "relationship_significant" in dialogue_context.emotional_themes:
            insights.append("relationship_dynamics_insight")
        
        if "high_intensity_emotion" in dialogue_context.emotional_themes:
            insights.append("emotional_intensity_awareness")
        
        # Personality-based insights
        if personality.get_trait_strength("wise") > 0.7:
            insights.append("life_experience_wisdom")
        
        if personality.get_trait_strength("analytical") > 0.7:
            insights.append("emotional_pattern_analysis")
        
        # Limit insights based on trust and relationship
        max_insights = min(len(insights), len(memory_references) + 1)
        return insights[:max_insights]

    def _determine_opening_approach(
        self, dialogue_context: EmotionalDialogueContext, personality: NPCPersonality
    ) -> str:
        """Determine how to open the response."""
        
        if dialogue_context.trauma_protection_active:
            return "gentle_check_in"
        
        if dialogue_context.dialogue_emotional_state == DialogueEmotionalState.EMOTIONALLY_OVERWHELMED:
            return "calming_presence"
        
        if dialogue_context.emotional_response_mode == EmotionalResponseMode.EMPATHETIC:
            return "emotional_reflection"
        
        if dialogue_context.emotional_response_mode == EmotionalResponseMode.SUPPORTIVE:
            return "supportive_acknowledgment"
        
        if personality.get_trait_strength("warm") > 0.7:
            return "warm_connection"
        
        return "neutral_acknowledgment"

    def _determine_core_message_type(
        self, dialogue_context: EmotionalDialogueContext, response_intent: str
    ) -> str:
        """Determine the type of core message to deliver."""
        
        if dialogue_context.emotional_response_mode == EmotionalResponseMode.THERAPEUTIC:
            return "therapeutic_insight"
        
        if dialogue_context.emotional_response_mode == EmotionalResponseMode.EMPATHETIC:
            return "empathetic_understanding"
        
        if dialogue_context.emotional_response_mode == EmotionalResponseMode.SUPPORTIVE:
            return "supportive_guidance"
        
        if dialogue_context.emotional_response_mode == EmotionalResponseMode.REFLECTIVE:
            return "reflective_exploration"
        
        if response_intent == "informational":
            return "informative_response"
        
        return "conversational_engagement"

    def _determine_closing_approach(
        self, dialogue_context: EmotionalDialogueContext, personality: NPCPersonality
    ) -> str:
        """Determine how to close the response."""
        
        if dialogue_context.trauma_protection_active:
            return "safety_reinforcement"
        
        if dialogue_context.emotional_response_mode == EmotionalResponseMode.THERAPEUTIC:
            return "therapeutic_check_in"
        
        if dialogue_context.emotional_response_mode == EmotionalResponseMode.SUPPORTIVE:
            return "ongoing_support_offer"
        
        if personality.get_trait_strength("caring") > 0.7:
            return "caring_connection"
        
        return "open_continuation"

    def _assess_safety_considerations(
        self, dialogue_context: EmotionalDialogueContext, memory_references: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Assess safety considerations for the response."""
        
        trigger_warnings = []
        protective_measures = []
        
        if dialogue_context.trauma_protection_active:
            protective_measures.extend([
                "trauma_aware_language",
                "safety_priority",
                "gentle_pacing"
            ])
        
        if "sensitive_content" in dialogue_context.emotional_themes:
            trigger_warnings.append("emotional_content_warning")
            protective_measures.append("content_sensitivity")
        
        if dialogue_context.dialogue_emotional_state == DialogueEmotionalState.EMOTIONALLY_OVERWHELMED:
            protective_measures.extend([
                "emotional_regulation_support",
                "pacing_adjustment"
            ])
        
        if memory_references and any("traumatic" in theme for theme in dialogue_context.emotional_themes):
            protective_measures.append("trauma_informed_approach")
        
        return trigger_warnings, protective_measures

    def _select_personality_emphasis(
        self,
        dialogue_context: EmotionalDialogueContext,
        personality: NPCPersonality,
        response_intent: str,
    ) -> List[str]:
        """Select which personality traits to emphasize."""
        
        emphasized_traits = []
        
        # Always emphasize core traits above 0.7
        for trait, strength in personality.traits.items():
            if strength > 0.7:
                emphasized_traits.append(trait)
        
        # Emphasize traits aligned with emotional response mode
        mode_trait_alignments = {
            EmotionalResponseMode.EMPATHETIC: ["empathetic", "caring", "sensitive"],
            EmotionalResponseMode.SUPPORTIVE: ["supportive", "nurturing", "helpful"],
            EmotionalResponseMode.THERAPEUTIC: ["wise", "insightful", "therapeutic"],
            EmotionalResponseMode.REFLECTIVE: ["reflective", "thoughtful", "analytical"],
            EmotionalResponseMode.PROTECTIVE: ["protective", "strong", "reliable"],
        }
        
        aligned_traits = mode_trait_alignments.get(dialogue_context.emotional_response_mode, [])
        for trait in aligned_traits:
            if personality.get_trait_strength(trait) > 0.5:
                emphasized_traits.append(trait)
        
        return list(set(emphasized_traits))  # Remove duplicates

    def _determine_emotional_archetype(
        self, personality: NPCPersonality, dialogue_context: EmotionalDialogueContext
    ) -> str:
        """Determine emotional archetype to express."""
        
        # Map personality combinations to archetypes
        if (personality.get_trait_strength("wise") > 0.7 and 
            personality.get_trait_strength("caring") > 0.6):
            return "wise_mentor"
        
        if (personality.get_trait_strength("empathetic") > 0.8 and 
            personality.get_trait_strength("supportive") > 0.7):
            return "empathetic_healer"
        
        if (personality.get_trait_strength("protective") > 0.7 and 
            personality.get_trait_strength("strong") > 0.6):
            return "protective_guardian"
        
        if (personality.get_trait_strength("playful") > 0.7 and 
            personality.get_trait_strength("optimistic") > 0.6):
            return "joyful_companion"
        
        if (personality.get_trait_strength("analytical") > 0.7 and 
            personality.get_trait_strength("insightful") > 0.6):
            return "thoughtful_analyst"
        
        # Adjust based on dialogue context
        if dialogue_context.trauma_protection_active:
            return "gentle_protector"
        
        if dialogue_context.emotional_response_mode == EmotionalResponseMode.THERAPEUTIC:
            return "therapeutic_guide"
        
        return "balanced_companion"

    async def _generate_therapeutic_elements(
        self, dialogue_context: EmotionalDialogueContext, personality: NPCPersonality
    ) -> List[str]:
        """Generate therapeutic elements for the response."""
        
        elements = []
        
        # Basic therapeutic elements
        elements.append("validation_and_normalization")
        
        if "recurring_traumatic" in dialogue_context.emotional_themes:
            elements.extend([
                "trauma_psychoeducation",
                "safety_and_grounding",
                "resource_building"
            ])
        
        if dialogue_context.current_mood in [MoodState.ANXIOUS, MoodState.FEARFUL]:
            elements.extend([
                "anxiety_management_techniques",
                "breathing_guidance"
            ])
        
        if dialogue_context.current_mood == MoodState.MELANCHOLY:
            elements.extend([
                "depression_support",
                "meaning_making_support"
            ])
        
        # Personality-informed therapeutic approach
        if personality.get_trait_strength("analytical") > 0.6:
            elements.append("cognitive_restructuring")
        
        if personality.get_trait_strength("creative") > 0.6:
            elements.append("creative_expression_encouragement")
        
        if personality.get_trait_strength("spiritual") > 0.6:
            elements.append("spiritual_coping_resources")
        
        return elements

    async def _analyze_response_emotional_impact(
        self, response_text: str, dialogue_context: EmotionalDialogueContext
    ) -> Dict[str, float]:
        """Analyze the emotional impact of a generated response."""
        
        # Simplified emotional impact analysis
        # In a full implementation, this would use NLP to analyze the response
        
        impact = {
            "emotional_intensity": 0.5,
            "supportiveness": 0.5,
            "vulnerability_shown": 0.3,
            "therapeutic_value": 0.0,
            "potential_triggering": 0.1,
        }
        
        # Adjust based on dialogue context
        if dialogue_context.emotional_response_mode == EmotionalResponseMode.SUPPORTIVE:
            impact["supportiveness"] = 0.8
        
        if dialogue_context.emotional_response_mode == EmotionalResponseMode.THERAPEUTIC:
            impact["therapeutic_value"] = 0.7
        
        if dialogue_context.trauma_protection_active:
            impact["potential_triggering"] = 0.0
            impact["emotional_intensity"] = 0.3
        
        return impact

    async def _predict_mood_changes(
        self,
        dialogue_context: EmotionalDialogueContext,
        response_impact: Dict[str, float],
        player_reaction: str,
    ) -> Dict[str, float]:
        """Predict mood changes from the interaction."""
        
        changes = {
            "intensity_change": 0.0,
            "stability_change": 0.0,
            "mood_shift_probability": 0.0,
        }
        
        # Supportive responses generally stabilize mood
        if response_impact["supportiveness"] > 0.7:
            changes["stability_change"] = 0.1
        
        # High emotional intensity can destabilize
        if response_impact["emotional_intensity"] > 0.8:
            changes["stability_change"] = -0.1
            changes["intensity_change"] = 0.1
        
        # Therapeutic value promotes positive change
        if response_impact["therapeutic_value"] > 0.6:
            changes["stability_change"] = 0.15
            
            # May shift toward more positive mood
            if dialogue_context.current_mood in [MoodState.MELANCHOLY, MoodState.ANXIOUS]:
                changes["mood_shift_probability"] = 0.3
        
        # Positive player reactions improve mood
        if "positive" in player_reaction.lower() or "thank" in player_reaction.lower():
            changes["intensity_change"] = 0.05
            changes["stability_change"] = 0.05
        
        return changes

    def _calculate_post_response_emotional_state(
        self,
        dialogue_context: EmotionalDialogueContext,
        response_impact: Dict[str, float],
    ) -> DialogueEmotionalState:
        """Calculate emotional state after generating a response."""
        
        current_state = dialogue_context.dialogue_emotional_state
        
        # Therapeutic responses move toward integration
        if response_impact["therapeutic_value"] > 0.7:
            if current_state == DialogueEmotionalState.EMOTIONALLY_OVERWHELMED:
                return DialogueEmotionalState.EMOTIONALLY_ENGAGED
            elif current_state == DialogueEmotionalState.EMOTIONALLY_ENGAGED:
                return DialogueEmotionalState.EMOTIONALLY_INTEGRATED
        
        # High supportiveness helps overwhelmed state
        if (response_impact["supportiveness"] > 0.8 and 
            current_state == DialogueEmotionalState.EMOTIONALLY_OVERWHELMED):
            return DialogueEmotionalState.EMOTIONALLY_ENGAGED
        
        # Successful vulnerable sharing promotes integration
        if (response_impact["vulnerability_shown"] > 0.6 and 
            current_state == DialogueEmotionalState.EMOTIONALLY_ENGAGED):
            return DialogueEmotionalState.EMOTIONALLY_INTEGRATED
        
        # Return current state if no significant change
        return current_state

    def _update_avg_integration_time(self, processing_time_ms: float) -> None:
        """Update average integration time statistic."""
        total_integrations = self._performance_stats["dialogue_integrations"]
        if total_integrations > 0:
            current_avg = self._performance_stats["avg_integration_time_ms"]
            self._performance_stats["avg_integration_time_ms"] = (
                current_avg * (total_integrations - 1) + processing_time_ms
            ) / total_integrations

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "dialogue_integrations": self._performance_stats["dialogue_integrations"],
            "emotional_responses_generated": self._performance_stats["emotional_responses_generated"],
            "trauma_protections_activated": self._performance_stats["trauma_protections_activated"],
            "therapeutic_interventions": self._performance_stats["therapeutic_interventions"],
            "memory_references_made": self._performance_stats["memory_references_made"],
            "avg_integration_time_ms": round(self._performance_stats["avg_integration_time_ms"], 2),
            "active_dialogue_contexts": len(self._dialogue_contexts),
            "total_response_history": sum(len(history) for history in self._response_history.values()),
        }

    def clear_caches(self) -> None:
        """Clear all caches and temporary data."""
        self._dialogue_contexts.clear()
        # Keep recent response history but limit size
        for npc_id in list(self._response_history.keys()):
            self._response_history[npc_id] = self._response_history[npc_id][-5:]  # Keep last 5
        
        self._performance_stats = {
            "dialogue_integrations": 0,
            "emotional_responses_generated": 0,
            "trauma_protections_activated": 0,
            "therapeutic_interventions": 0,
            "memory_references_made": 0,
            "avg_integration_time_ms": 0.0,
        }