"""Enhanced emotional memory context and affective weighting system."""

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from game_loop.core.conversation.conversation_models import (
    ConversationContext,
    ConversationExchange,
    NPCPersonality,
)
from game_loop.database.session_factory import DatabaseSessionFactory
from game_loop.llm.ollama.client import OllamaClient

from .config import MemoryAlgorithmConfig
from .constants import (
    EmotionalThresholds,
    EmotionalTypeThresholds,
    MoodTransitionPatterns,
)
from .emotional_analysis import EmotionalAnalysisResult, EmotionalWeightingAnalyzer
from .exceptions import (
    EmotionalAnalysisError,
    InvalidEmotionalDataError,
    handle_emotional_memory_error,
)
# Circular import issue - using basic validation for now
# from .validation import (
#     validate_uuid,
#     validate_probability,
#     validate_string_content,
#     validate_mood_state,
#     validate_emotional_memory_type,
#     default_validator,
# )

logger = logging.getLogger(__name__)


class MoodState(Enum):
    """Current mood state that affects memory accessibility."""

    JOYFUL = "joyful"
    CONTENT = "content"
    NEUTRAL = "neutral"
    MELANCHOLY = "melancholy"
    ANXIOUS = "anxious"
    ANGRY = "angry"
    FEARFUL = "fearful"
    EXCITED = "excited"
    NOSTALGIC = "nostalgic"
    HOPEFUL = "hopeful"


class EmotionalMemoryType(Enum):
    """Types of emotional memories with different significance levels."""

    PEAK_POSITIVE = "peak_positive"  # Life-defining positive moments
    CORE_ATTACHMENT = "core_attachment"  # Deep relationship bonds
    TRAUMATIC = "traumatic"  # Traumatic or deeply negative experiences
    FORMATIVE = "formative"  # Personality-shaping experiences
    SIGNIFICANT_LOSS = "significant_loss"  # Grief, loss, major disappointments
    BREAKTHROUGH = "breakthrough"  # Personal growth, achievements
    CONFLICT = "conflict"  # Interpersonal conflicts
    TRUST_EVENT = "trust_event"  # Trust establishment or betrayal
    EVERYDAY_POSITIVE = "everyday_positive"  # Regular positive interactions
    ROUTINE_NEGATIVE = "routine_negative"  # Minor negative experiences


class MemoryProtectionLevel(Enum):
    """Protection levels for sensitive memories."""

    PUBLIC = "public"  # Freely accessible
    PRIVATE = "private"  # Requires moderate trust
    SENSITIVE = "sensitive"  # Requires high trust
    PROTECTED = "protected"  # Requires very high trust
    TRAUMATIC = "traumatic"  # Special handling required


@dataclass
class EmotionalSignificance:
    """Comprehensive emotional significance scoring for a memory."""

    overall_significance: float  # 0.0-1.0 overall emotional importance
    emotional_type: EmotionalMemoryType  # Primary emotional classification
    intensity_score: float  # 0.0-1.0 emotional intensity
    personal_relevance: float  # 0.0-1.0 relevance to personality
    relationship_impact: float  # -1.0 to 1.0 impact on relationships
    formative_influence: float  # 0.0-1.0 influence on personality development
    protection_level: MemoryProtectionLevel  # Required trust level for access
    mood_accessibility: Dict[MoodState, float]  # Accessibility in different moods
    decay_resistance: float  # 0.0-1.0 resistance to memory decay
    triggering_potential: float  # 0.0-1.0 likelihood to trigger related memories

    # Metadata
    confidence_score: float = 0.0
    analysis_timestamp: float = field(default_factory=time.time)
    contributing_factors: List[str] = field(default_factory=list)


@dataclass
class MoodDependentAccess:
    """Mood-dependent memory accessibility patterns."""

    base_accessibility: float  # Base accessibility (0.0-1.0)
    mood_modifiers: Dict[MoodState, float]  # Mood-specific modifiers
    emotional_resonance: Dict[str, float]  # Resonance with emotional keywords
    contextual_triggers: List[str]  # Triggers that increase accessibility
    blocking_factors: List[str]  # Factors that decrease accessibility


@dataclass
class EmotionalMemoryCluster:
    """Cluster of emotionally related memories."""

    cluster_id: str
    emotional_theme: str  # Primary emotional theme
    dominant_type: EmotionalMemoryType  # Dominant memory type in cluster
    emotional_coherence: float  # 0.0-1.0 emotional consistency
    temporal_span: Tuple[float, float]  # (start_time, end_time) of memories
    triggering_strength: float  # How strongly memories trigger each other
    protection_level: MemoryProtectionLevel  # Highest protection level in cluster
    member_memories: List[str] = field(default_factory=list)
    associated_moods: List[MoodState] = field(default_factory=list)
    narrative_coherence: float = 0.0  # How well memories form a story


class EmotionalMemoryContextEngine:
    """Advanced emotional memory context and affective weighting system."""

    def __init__(
        self,
        session_factory: DatabaseSessionFactory,
        llm_client: OllamaClient,
        config: MemoryAlgorithmConfig,
        emotional_analyzer: Optional[EmotionalWeightingAnalyzer] = None,
    ):
        self.session_factory = session_factory
        self.llm_client = llm_client
        self.config = config
        self.emotional_analyzer = emotional_analyzer or EmotionalWeightingAnalyzer(
            config
        )

        # Caches for performance (with size limits)
        self._significance_cache: Dict[str, EmotionalSignificance] = {}
        self._mood_access_cache: Dict[str, MoodDependentAccess] = {}
        self._memory_clusters: Dict[str, List[EmotionalMemoryCluster]] = {}
        self._max_cache_size = (
            config.get_processing_limit(
                "max_cache_size", EmotionalThresholds.DEFAULT_CACHE_SIZE
            )
            if hasattr(config, "get_processing_limit")
            else EmotionalThresholds.DEFAULT_CACHE_SIZE
        )

        # Performance tracking
        self._performance_stats = {
            "significance_analyses": 0,
            "cache_hits": 0,
            "avg_processing_time_ms": 0.0,
        }

    async def analyze_emotional_significance(
        self,
        exchange: ConversationExchange,
        conversation_context: ConversationContext,
        npc_personality: NPCPersonality,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> EmotionalSignificance:
        """
        Analyze comprehensive emotional significance of a conversation exchange.

        This goes beyond basic emotional analysis to understand:
        - Life significance and formative impact
        - Relationship implications
        - Memory protection needs
        - Mood-dependent accessibility patterns
        """
        try:
            # Validate inputs
            if not exchange or not exchange.exchange_id:
                raise InvalidEmotionalDataError(
                    "Exchange must have a valid exchange_id"
                )
            if not conversation_context:
                raise InvalidEmotionalDataError("Conversation context is required")
            if not npc_personality:
                raise InvalidEmotionalDataError("NPC personality is required")

            # Validate exchange content
            if not exchange.message_text or not exchange.message_text.strip():
                raise InvalidEmotionalDataError("Exchange message text cannot be empty")

            # Basic validation without importing validation module
            if len(exchange.message_text) > 10000:
                raise InvalidEmotionalDataError("Exchange message text too long (max 10000 chars)")

        except Exception as e:
            raise handle_emotional_memory_error(
                e, "Failed to validate inputs for emotional significance analysis"
            )

        start_time = time.perf_counter()

        # Create cache key
        cache_key = self._create_significance_cache_key(
            exchange, conversation_context, npc_personality
        )

        try:
            # Check cache
            if cache_key in self._significance_cache:
                self._performance_stats["cache_hits"] += 1
                return self._significance_cache[cache_key]

            # Perform basic emotional analysis first
            basic_analysis = self.emotional_analyzer.analyze_emotional_weight(
                exchange.message_text,
                conversation_context.to_dict(),
                npc_personality.to_dict(),
            )

            # Enhanced significance analysis
            significance = await self._analyze_comprehensive_significance(
                exchange,
                conversation_context,
                npc_personality,
                basic_analysis,
                additional_context,
            )

            # Cache result with size management
            if len(self._significance_cache) >= self._max_cache_size:
                # Remove oldest entries (simple FIFO)
                oldest_keys = list(self._significance_cache.keys())[
                    : int(self._max_cache_size * 0.1)
                ]
                for old_key in oldest_keys:
                    self._significance_cache.pop(old_key, None)

            self._significance_cache[cache_key] = significance

            # Update performance stats
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            self._performance_stats["significance_analyses"] += 1
            total_analyses = self._performance_stats["significance_analyses"]
            self._performance_stats["avg_processing_time_ms"] = (
                self._performance_stats["avg_processing_time_ms"] * (total_analyses - 1)
                + processing_time_ms
            ) / total_analyses

            return significance

        except Exception as e:
            raise handle_emotional_memory_error(
                e, "Failed to analyze emotional significance"
            )

    async def _analyze_comprehensive_significance(
        self,
        exchange: ConversationExchange,
        conversation_context: ConversationContext,
        npc_personality: NPCPersonality,
        basic_analysis: EmotionalAnalysisResult,
        additional_context: Optional[Dict[str, Any]],
    ) -> EmotionalSignificance:
        """Perform comprehensive emotional significance analysis."""

        # Determine emotional memory type
        emotional_type = await self._classify_emotional_memory_type(
            exchange, basic_analysis, conversation_context
        )

        # Calculate personal relevance based on personality
        personal_relevance = self._calculate_personal_relevance(
            basic_analysis, npc_personality, emotional_type
        )

        # Assess formative influence
        formative_influence = await self._assess_formative_influence(
            exchange, conversation_context, emotional_type, personal_relevance
        )

        # Determine protection level
        protection_level = self._determine_protection_level(
            emotional_type, basic_analysis, formative_influence
        )

        # Calculate mood-dependent accessibility
        mood_accessibility = self._calculate_mood_accessibility(
            emotional_type, basic_analysis, npc_personality
        )

        # Calculate decay resistance (emotional memories resist forgetting)
        decay_resistance = self._calculate_decay_resistance(
            emotional_type, basic_analysis, formative_influence
        )

        # Calculate triggering potential
        triggering_potential = self._calculate_triggering_potential(
            emotional_type, basic_analysis, conversation_context
        )

        # Overall significance combines multiple factors
        overall_significance = self._calculate_overall_significance(
            basic_analysis.emotional_weight,
            personal_relevance,
            formative_influence,
            emotional_type,
        )

        # Contributing factors for explainability
        contributing_factors = self._identify_contributing_factors(
            basic_analysis, emotional_type, personal_relevance, formative_influence
        )

        return EmotionalSignificance(
            overall_significance=overall_significance,
            emotional_type=emotional_type,
            intensity_score=basic_analysis.emotional_intensity,
            personal_relevance=personal_relevance,
            relationship_impact=basic_analysis.relationship_impact,
            formative_influence=formative_influence,
            protection_level=protection_level,
            mood_accessibility=mood_accessibility,
            decay_resistance=decay_resistance,
            triggering_potential=triggering_potential,
            confidence_score=basic_analysis.analysis_confidence,
            contributing_factors=contributing_factors,
        )

    async def _classify_emotional_memory_type(
        self,
        exchange: ConversationExchange,
        basic_analysis: EmotionalAnalysisResult,
        conversation_context: ConversationContext,
    ) -> EmotionalMemoryType:
        """Classify the type of emotional memory based on content and context."""

        text = exchange.message_text.lower()
        emotional_weight = basic_analysis.emotional_weight
        relationship_impact = abs(basic_analysis.relationship_impact)

        # Check for traumatic indicators
        trauma_keywords = {
            "trauma",
            "traumatic",
            "nightmare",
            "terrified",
            "horrified",
            "devastated",
            "shattered",
            "broken",
            "destroyed",
            "ruined",
            "abused",
            "attacked",
            "betrayed",
            "abandoned",
            "rejected",
        }
        if (
            any(keyword in text for keyword in trauma_keywords)
            and emotional_weight > EmotionalThresholds.TRAUMA_WEIGHT_THRESHOLD
        ):
            return EmotionalMemoryType.TRAUMATIC

        # Check for peak positive experiences
        peak_positive_keywords = {
            "best day",
            "happiest",
            "amazing",
            "incredible",
            "unforgettable",
            "perfect",
            "wonderful",
            "magical",
            "dream come true",
            "blessed",
        }
        if (
            any(keyword in text for keyword in peak_positive_keywords)
            and emotional_weight > EmotionalThresholds.PEAK_EXPERIENCE_THRESHOLD
        ):
            return EmotionalMemoryType.PEAK_POSITIVE

        # Check for core attachment/relationship moments
        attachment_keywords = {
            "love",
            "loved",
            "care",
            "caring",
            "devoted",
            "cherish",
            "family",
            "parent",
            "child",
            "partner",
            "best friend",
        }
        if (
            any(keyword in text for keyword in attachment_keywords)
            and relationship_impact > EmotionalThresholds.HIGH_RELATIONSHIP_IMPACT
        ):
            return EmotionalMemoryType.CORE_ATTACHMENT

        # Check for formative experiences
        formative_keywords = {
            "first time",
            "learned",
            "realized",
            "understood",
            "changed me",
            "grew up",
            "matured",
            "became",
            "discovered",
            "awakened",
        }
        if (
            any(keyword in text for keyword in formative_keywords)
            and emotional_weight > EmotionalThresholds.FORMATIVE_WEIGHT_THRESHOLD
        ):
            return EmotionalMemoryType.FORMATIVE

        # Check for significant loss
        loss_keywords = {
            "died",
            "death",
            "lost",
            "goodbye",
            "farewell",
            "ended",
            "grief",
            "mourning",
            "miss",
            "gone",
            "never again",
        }
        if (
            any(keyword in text for keyword in loss_keywords)
            and emotional_weight > EmotionalThresholds.FORMATIVE_WEIGHT_THRESHOLD
        ):
            return EmotionalMemoryType.SIGNIFICANT_LOSS

        # Check for breakthrough moments
        breakthrough_keywords = {
            "achieved",
            "accomplished",
            "succeeded",
            "breakthrough",
            "victory",
            "proud",
            "overcome",
            "conquered",
            "mastered",
            "breakthrough",
        }
        if (
            any(keyword in text for keyword in breakthrough_keywords)
            and emotional_weight > EmotionalThresholds.FORMATIVE_WEIGHT_THRESHOLD
        ):
            return EmotionalMemoryType.BREAKTHROUGH

        # Check for conflict
        conflict_keywords = {
            "argue",
            "argued",
            "fight",
            "fought",
            "angry",
            "mad",
            "disagreed",
            "conflict",
            "confrontation",
            "quarrel",
        }
        if (
            any(keyword in text for keyword in conflict_keywords)
            and emotional_weight > EmotionalThresholds.MODERATE_SIGNIFICANCE
        ):
            return EmotionalMemoryType.CONFLICT

        # Check for trust events
        trust_keywords = {
            "trust",
            "trusted",
            "betrayed",
            "loyal",
            "faithful",
            "honest",
            "lied",
            "deceived",
            "reliable",
            "depend",
        }
        if (
            any(keyword in text for keyword in trust_keywords)
            and relationship_impact > EmotionalThresholds.FORMATIVE_WEIGHT_THRESHOLD
        ):
            return EmotionalMemoryType.TRUST_EVENT

        # Default categorization based on emotional weight
        if (
            basic_analysis.sentiment_score > EmotionalThresholds.LOW_SIGNIFICANCE
            and emotional_weight > (EmotionalThresholds.MODERATE_SIGNIFICANCE - 0.1)
        ):
            return EmotionalMemoryType.EVERYDAY_POSITIVE
        elif (
            basic_analysis.sentiment_score < -EmotionalThresholds.LOW_SIGNIFICANCE
            and emotional_weight > (EmotionalThresholds.MODERATE_SIGNIFICANCE - 0.1)
        ):
            return EmotionalMemoryType.ROUTINE_NEGATIVE
        else:
            return (
                EmotionalMemoryType.EVERYDAY_POSITIVE
            )  # Default to positive for neutral

    def _calculate_personal_relevance(
        self,
        basic_analysis: EmotionalAnalysisResult,
        npc_personality: NPCPersonality,
        emotional_type: EmotionalMemoryType,
    ) -> float:
        """Calculate how personally relevant this memory is to the NPC's personality."""

        base_relevance = basic_analysis.emotional_weight

        # Personality-based modifiers
        emotional_sensitivity = npc_personality.get_trait_strength(
            "emotional_sensitivity"
        )
        if emotional_sensitivity == 0:
            emotional_sensitivity = npc_personality.get_trait_strength("emotional")

        # Apply emotional sensitivity
        relevance = base_relevance * (
            EmotionalThresholds.MODERATE_SIGNIFICANCE + emotional_sensitivity
        )

        # Type-specific personality alignments
        if emotional_type == EmotionalMemoryType.CORE_ATTACHMENT:
            social_trait = npc_personality.get_trait_strength("social")
            loving_trait = npc_personality.get_trait_strength("loving")
            relevance *= 1.0 + max(social_trait, loving_trait) * 0.5

        elif emotional_type == EmotionalMemoryType.TRAUMATIC:
            trauma_sensitive = npc_personality.get_trait_strength("trauma_sensitive")
            anxious_trait = npc_personality.get_trait_strength("anxious")
            relevance *= 1.0 + max(trauma_sensitive, anxious_trait) * 0.8

        elif emotional_type == EmotionalMemoryType.FORMATIVE:
            analytical_trait = npc_personality.get_trait_strength("analytical")
            reflective_trait = npc_personality.get_trait_strength("reflective")
            relevance *= 1.0 + max(analytical_trait, reflective_trait) * 0.6

        elif emotional_type == EmotionalMemoryType.CONFLICT:
            conflict_averse = npc_personality.get_trait_strength("conflict_averse")
            aggressive_trait = npc_personality.get_trait_strength("aggressive")
            # Conflict-averse NPCs remember conflicts more vividly
            relevance *= 1.0 + max(conflict_averse, aggressive_trait) * 0.7

        return min(1.0, relevance)

    async def _assess_formative_influence(
        self,
        exchange: ConversationExchange,
        conversation_context: ConversationContext,
        emotional_type: EmotionalMemoryType,
        personal_relevance: float,
    ) -> float:
        """Assess how much this memory influenced or could influence personality development."""

        base_influence = 0.0

        # Type-based formative influence
        formative_weights = {
            EmotionalMemoryType.TRAUMATIC: 0.9,
            EmotionalMemoryType.FORMATIVE: 0.8,
            EmotionalMemoryType.PEAK_POSITIVE: 0.7,
            EmotionalMemoryType.CORE_ATTACHMENT: 0.7,
            EmotionalMemoryType.SIGNIFICANT_LOSS: 0.6,
            EmotionalMemoryType.BREAKTHROUGH: 0.6,
            EmotionalMemoryType.TRUST_EVENT: 0.5,
            EmotionalMemoryType.CONFLICT: 0.4,
            EmotionalMemoryType.EVERYDAY_POSITIVE: 0.2,
            EmotionalMemoryType.ROUTINE_NEGATIVE: 0.1,
        }

        base_influence = formative_weights.get(emotional_type, 0.1)

        # Relationship level amplifies formative influence
        relationship_amplifier = (
            1.0 + float(conversation_context.relationship_level) * 0.5
        )

        # First-time experiences are more formative
        text = exchange.message_text.lower()
        if "first time" in text or "never" in text:
            base_influence *= 1.3

        # Age/experience context (would need more context, using relationship as proxy)
        experience_modifier = (
            1.0 + (1.0 - float(conversation_context.relationship_level)) * 0.2
        )

        formative_influence = (
            base_influence
            * personal_relevance
            * relationship_amplifier
            * experience_modifier
        )

        return min(1.0, formative_influence)

    def _determine_protection_level(
        self,
        emotional_type: EmotionalMemoryType,
        basic_analysis: EmotionalAnalysisResult,
        formative_influence: float,
    ) -> MemoryProtectionLevel:
        """Determine the protection level required for accessing this memory."""

        # Type-based protection levels
        type_protection = {
            EmotionalMemoryType.TRAUMATIC: MemoryProtectionLevel.TRAUMATIC,
            EmotionalMemoryType.CORE_ATTACHMENT: MemoryProtectionLevel.PROTECTED,
            EmotionalMemoryType.SIGNIFICANT_LOSS: MemoryProtectionLevel.PROTECTED,
            EmotionalMemoryType.FORMATIVE: MemoryProtectionLevel.SENSITIVE,
            EmotionalMemoryType.PEAK_POSITIVE: MemoryProtectionLevel.SENSITIVE,
            EmotionalMemoryType.TRUST_EVENT: MemoryProtectionLevel.SENSITIVE,
            EmotionalMemoryType.BREAKTHROUGH: MemoryProtectionLevel.PRIVATE,
            EmotionalMemoryType.CONFLICT: MemoryProtectionLevel.PRIVATE,
            EmotionalMemoryType.EVERYDAY_POSITIVE: MemoryProtectionLevel.PUBLIC,
            EmotionalMemoryType.ROUTINE_NEGATIVE: MemoryProtectionLevel.PRIVATE,
        }

        base_protection = type_protection.get(
            emotional_type, MemoryProtectionLevel.PUBLIC
        )

        # Upgrade protection based on formative influence
        if (
            formative_influence > EmotionalThresholds.HIGH_SIGNIFICANCE_THRESHOLD
            and base_protection != MemoryProtectionLevel.TRAUMATIC
        ):
            if base_protection == MemoryProtectionLevel.PUBLIC:
                return MemoryProtectionLevel.PRIVATE
            elif base_protection == MemoryProtectionLevel.PRIVATE:
                return MemoryProtectionLevel.SENSITIVE
            elif base_protection == MemoryProtectionLevel.SENSITIVE:
                return MemoryProtectionLevel.PROTECTED

        return base_protection

    def _calculate_mood_accessibility(
        self,
        emotional_type: EmotionalMemoryType,
        basic_analysis: EmotionalAnalysisResult,
        npc_personality: NPCPersonality,
    ) -> Dict[MoodState, float]:
        """Calculate how accessible this memory is in different mood states."""

        base_accessibility = (
            EmotionalThresholds.MODERATE_SIGNIFICANCE
        )  # Neutral baseline

        # Mood compatibility patterns based on emotional type
        mood_patterns = {
            EmotionalMemoryType.PEAK_POSITIVE: {
                MoodState.JOYFUL: 0.9,
                MoodState.CONTENT: 0.8,
                MoodState.HOPEFUL: 0.8,
                MoodState.EXCITED: 0.9,
                MoodState.NOSTALGIC: 0.7,
                MoodState.NEUTRAL: 0.5,
                MoodState.MELANCHOLY: 0.3,
                MoodState.ANXIOUS: 0.2,
                MoodState.ANGRY: 0.1,
                MoodState.FEARFUL: 0.1,
            },
            EmotionalMemoryType.TRAUMATIC: {
                MoodState.FEARFUL: 0.9,
                MoodState.ANXIOUS: 0.8,
                MoodState.ANGRY: 0.7,
                MoodState.MELANCHOLY: 0.6,
                MoodState.NEUTRAL: 0.3,
                MoodState.NOSTALGIC: 0.2,
                MoodState.CONTENT: 0.1,
                MoodState.JOYFUL: 0.05,
                MoodState.EXCITED: 0.05,
                MoodState.HOPEFUL: 0.1,
            },
            EmotionalMemoryType.SIGNIFICANT_LOSS: {
                MoodState.MELANCHOLY: 0.9,
                MoodState.NOSTALGIC: 0.8,
                MoodState.ANXIOUS: 0.6,
                MoodState.NEUTRAL: 0.4,
                MoodState.FEARFUL: 0.3,
                MoodState.CONTENT: 0.2,
                MoodState.ANGRY: 0.2,
                MoodState.HOPEFUL: 0.15,
                MoodState.JOYFUL: 0.05,
                MoodState.EXCITED: 0.05,
            },
            # Add more patterns for other types...
        }

        # Default pattern for unlisted types
        default_pattern = {mood: base_accessibility for mood in MoodState}

        accessibility = mood_patterns.get(emotional_type, default_pattern)

        # Apply personality modifiers
        emotional_sensitivity = npc_personality.get_trait_strength(
            "emotional_sensitivity"
        )
        if emotional_sensitivity == 0:
            emotional_sensitivity = npc_personality.get_trait_strength("emotional")

        # Highly emotional personalities have stronger mood effects
        for mood in accessibility:
            mood_effect = accessibility[mood] - base_accessibility
            accessibility[mood] = base_accessibility + mood_effect * (
                1.0 + emotional_sensitivity
            )
            accessibility[mood] = max(0.0, min(1.0, accessibility[mood]))

        return accessibility

    def _calculate_decay_resistance(
        self,
        emotional_type: EmotionalMemoryType,
        basic_analysis: EmotionalAnalysisResult,
        formative_influence: float,
    ) -> float:
        """Calculate how resistant this memory is to decay over time."""

        # Type-based decay resistance
        type_resistance = {
            EmotionalMemoryType.TRAUMATIC: 0.95,  # Trauma rarely fades
            EmotionalMemoryType.FORMATIVE: 0.9,  # Life-changing moments persist
            EmotionalMemoryType.PEAK_POSITIVE: 0.85,  # Peak experiences are vivid
            EmotionalMemoryType.CORE_ATTACHMENT: 0.9,  # Deep relationships remembered
            EmotionalMemoryType.SIGNIFICANT_LOSS: 0.8,  # Loss is remembered
            EmotionalMemoryType.BREAKTHROUGH: 0.75,  # Achievements remembered
            EmotionalMemoryType.TRUST_EVENT: 0.8,  # Trust events are significant
            EmotionalMemoryType.CONFLICT: 0.7,  # Conflicts stick in memory
            EmotionalMemoryType.EVERYDAY_POSITIVE: 0.4,  # Regular positive fades
            EmotionalMemoryType.ROUTINE_NEGATIVE: 0.5,  # Mild negatives remembered longer
        }

        base_resistance = type_resistance.get(emotional_type, 0.5)

        # Emotional weight and formative influence strengthen resistance
        emotional_boost = basic_analysis.emotional_weight * 0.3
        formative_boost = formative_influence * 0.4

        resistance = base_resistance + emotional_boost + formative_boost
        return min(1.0, resistance)

    def _calculate_triggering_potential(
        self,
        emotional_type: EmotionalMemoryType,
        basic_analysis: EmotionalAnalysisResult,
        conversation_context: ConversationContext,
    ) -> float:
        """Calculate how likely this memory is to trigger related memories."""

        base_triggering = basic_analysis.emotional_weight * 0.6

        # Type-based triggering strength
        type_triggers = {
            EmotionalMemoryType.TRAUMATIC: 0.9,  # Trauma strongly triggers related memories
            EmotionalMemoryType.CORE_ATTACHMENT: 0.8,  # Attachment memories chain together
            EmotionalMemoryType.SIGNIFICANT_LOSS: 0.8,  # Loss memories connect
            EmotionalMemoryType.FORMATIVE: 0.7,  # Formative experiences link
            EmotionalMemoryType.PEAK_POSITIVE: 0.7,  # Peak experiences chain
            EmotionalMemoryType.CONFLICT: 0.6,  # Conflicts trigger other conflicts
            EmotionalMemoryType.TRUST_EVENT: 0.6,  # Trust events connect
            EmotionalMemoryType.BREAKTHROUGH: 0.5,  # Achievements may link
            EmotionalMemoryType.EVERYDAY_POSITIVE: 0.3,  # Regular positives less triggering
            EmotionalMemoryType.ROUTINE_NEGATIVE: 0.4,  # Minor negatives somewhat triggering
        }

        type_modifier = type_triggers.get(emotional_type, 0.5)

        # Relationship context amplifies triggering
        relationship_amplifier = (
            1.0 + float(conversation_context.relationship_level) * 0.3
        )

        triggering = base_triggering * type_modifier * relationship_amplifier
        return min(1.0, triggering)

    def _calculate_overall_significance(
        self,
        emotional_weight: float,
        personal_relevance: float,
        formative_influence: float,
        emotional_type: EmotionalMemoryType,
    ) -> float:
        """Calculate overall emotional significance score."""

        # Weighted combination of factors
        significance = (
            emotional_weight * 0.4
            + personal_relevance * 0.3
            + formative_influence * 0.3
        )

        # Type-based significance boosts
        type_boosts = {
            EmotionalMemoryType.TRAUMATIC: 0.2,
            EmotionalMemoryType.FORMATIVE: 0.15,
            EmotionalMemoryType.PEAK_POSITIVE: 0.1,
            EmotionalMemoryType.CORE_ATTACHMENT: 0.15,
            EmotionalMemoryType.SIGNIFICANT_LOSS: 0.1,
        }

        significance += type_boosts.get(emotional_type, 0.0)

        return min(1.0, significance)

    def _identify_contributing_factors(
        self,
        basic_analysis: EmotionalAnalysisResult,
        emotional_type: EmotionalMemoryType,
        personal_relevance: float,
        formative_influence: float,
    ) -> List[str]:
        """Identify factors that contributed to the significance score."""

        factors = []

        if (
            basic_analysis.emotional_weight
            > EmotionalThresholds.TRAUMA_WEIGHT_THRESHOLD
        ):
            factors.append("high_emotional_intensity")

        if personal_relevance > EmotionalThresholds.HIGH_RELATIONSHIP_IMPACT:
            factors.append("strong_personal_relevance")

        if formative_influence > EmotionalThresholds.FORMATIVE_WEIGHT_THRESHOLD:
            factors.append("formative_life_experience")

        if (
            abs(basic_analysis.relationship_impact)
            > EmotionalThresholds.FORMATIVE_WEIGHT_THRESHOLD
        ):
            factors.append("significant_relationship_impact")

        if basic_analysis.emotional_keywords:
            factors.append(
                f"emotional_keywords: {', '.join(basic_analysis.emotional_keywords[:3])}"
            )

        factors.append(f"memory_type: {emotional_type.value}")

        return factors

    def _create_significance_cache_key(
        self,
        exchange: ConversationExchange,
        conversation_context: ConversationContext,
        npc_personality: NPCPersonality,
    ) -> str:
        """Create cache key for significance analysis."""
        # Use hash of key factors
        key_data = f"{exchange.exchange_id}_{conversation_context.relationship_level}_{hash(frozenset(npc_personality.traits.items()) if npc_personality.traits else 0)}"
        return str(hash(key_data))

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the emotional context engine."""
        total_requests = (
            self._performance_stats["significance_analyses"]
            + self._performance_stats["cache_hits"]
        )
        cache_hit_rate = (
            (self._performance_stats["cache_hits"] / total_requests * 100)
            if total_requests > 0
            else 0.0
        )

        return {
            "total_significance_analyses": self._performance_stats[
                "significance_analyses"
            ],
            "cache_hits": self._performance_stats["cache_hits"],
            "cache_hit_rate_percent": round(cache_hit_rate, 1),
            "avg_processing_time_ms": round(
                self._performance_stats["avg_processing_time_ms"], 2
            ),
            "cached_significances": len(self._significance_cache),
            "cached_mood_access": len(self._mood_access_cache),
        }

    def clear_caches(self) -> None:
        """Clear all caches."""
        self._significance_cache.clear()
        self._mood_access_cache.clear()
        self._memory_clusters.clear()
        self._performance_stats = {
            "significance_analyses": 0,
            "cache_hits": 0,
            "avg_processing_time_ms": 0.0,
        }
