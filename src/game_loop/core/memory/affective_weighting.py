"""Affective memory weighting system based on emotional intensity and significance."""

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from game_loop.core.conversation.conversation_models import NPCPersonality
from game_loop.database.session_factory import DatabaseSessionFactory

from .config import MemoryAlgorithmConfig
from .constants import EmotionalThresholds, ProtectionMechanismConfig
from .exceptions import (
    EmotionalAnalysisError, InvalidEmotionalDataError, PerformanceError, 
    handle_emotional_memory_error
)
# from .validation import (
#     validate_probability, validate_positive_number, validate_mood_state,
#     validate_uuid, default_validator
# )
from .emotional_context import (
    EmotionalMemoryContextEngine,
    EmotionalMemoryType,
    EmotionalSignificance,
    MoodState,
    MemoryProtectionLevel,
)

logger = logging.getLogger(__name__)


class AffectiveWeightingStrategy(Enum):
    """Strategies for applying affective weighting to memories."""
    
    LINEAR = "linear"                    # Linear scaling based on emotional weight
    EXPONENTIAL = "exponential"          # Exponential scaling for high-emotion memories  
    THRESHOLD = "threshold"              # Step function at emotional thresholds
    PERSONALITY_ADAPTIVE = "adaptive"    # Adapts to NPC personality traits
    MOOD_SENSITIVE = "mood_sensitive"    # Adjusts based on current mood state


class MemoryAccessStrategy(Enum):
    """Strategies for accessing memories based on emotional state."""
    
    MOOD_CONGRUENT = "mood_congruent"    # Access memories matching current mood
    MOOD_CONTRASTING = "contrasting"     # Access memories opposing current mood
    BALANCED = "balanced"                # Balance congruent and contrasting
    THERAPEUTIC = "therapeutic"          # Access memories that help mood regulation
    AVOIDANT = "avoidant"               # Avoid triggering memories


@dataclass
class AffectiveWeight:
    """Comprehensive affective weighting for a memory."""
    
    # Core weighting
    base_affective_weight: float          # Base emotional weight (0.0-1.0)
    intensity_multiplier: float           # Multiplier based on emotional intensity
    personality_modifier: float           # Personality-based adjustment
    mood_accessibility_modifier: float    # Current mood accessibility adjustment
    
    # Contextual factors
    recency_boost: float                  # Boost for recent emotional memories
    relationship_amplifier: float         # Amplification based on relationship context
    formative_importance: float           # Weight for formative life experiences
    
    # Protection and access
    access_threshold: float               # Minimum trust required for access
    trauma_sensitivity: float             # Special handling for traumatic content
    
    # Meta information
    final_weight: float                   # Combined final weight
    weighting_strategy: AffectiveWeightingStrategy
    confidence: float                     # Confidence in weighting accuracy
    computed_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "base_affective_weight": self.base_affective_weight,
            "intensity_multiplier": self.intensity_multiplier,
            "personality_modifier": self.personality_modifier,
            "mood_accessibility_modifier": self.mood_accessibility_modifier,
            "recency_boost": self.recency_boost,
            "relationship_amplifier": self.relationship_amplifier,
            "formative_importance": self.formative_importance,
            "access_threshold": self.access_threshold,
            "trauma_sensitivity": self.trauma_sensitivity,
            "final_weight": self.final_weight,
            "weighting_strategy": self.weighting_strategy.value,
            "confidence": self.confidence,
            "computed_at": self.computed_at,
        }


@dataclass
class MoodBasedAccessibility:
    """Mood-based memory accessibility calculation."""
    
    current_mood: MoodState
    base_accessibility: float
    mood_congruent_boost: float           # Boost for mood-matching memories
    mood_contrasting_penalty: float       # Penalty for mood-opposing memories
    therapeutic_value: float              # Value for mood regulation
    triggering_risk: float               # Risk of triggering negative states
    
    # Final accessibility
    adjusted_accessibility: float
    access_strategy: MemoryAccessStrategy
    
    def is_accessible(self, trust_level: float) -> bool:
        """Check if memory is accessible given trust level."""
        return self.adjusted_accessibility > EmotionalThresholds.LOW_SIGNIFICANCE and trust_level >= self.access_threshold
    
    @property 
    def access_threshold(self) -> float:
        """Get the trust threshold needed for access."""
        # Higher triggering risk requires more trust
        base_threshold = EmotionalThresholds.LOW_SIGNIFICANCE
        risk_adjustment = self.triggering_risk * 0.4
        return min(EmotionalThresholds.CRISIS_SENSITIVITY_THRESHOLD, base_threshold + risk_adjustment)


class AffectiveMemoryWeightingEngine:
    """Engine for computing affective memory weights based on emotional significance."""
    
    def __init__(
        self,
        session_factory: DatabaseSessionFactory,
        config: MemoryAlgorithmConfig,
        emotional_context_engine: EmotionalMemoryContextEngine,
        default_strategy: AffectiveWeightingStrategy = AffectiveWeightingStrategy.PERSONALITY_ADAPTIVE,
    ):
        self.session_factory = session_factory
        self.config = config
        self.emotional_context_engine = emotional_context_engine
        self.default_strategy = default_strategy
        
        # Caches with size limits
        self._weight_cache: Dict[str, AffectiveWeight] = {}
        self._mood_access_cache: Dict[str, MoodBasedAccessibility] = {}
        self._max_cache_size = getattr(config, 'max_cache_size', EmotionalThresholds.DEFAULT_CACHE_SIZE)
        
        # Performance tracking
        self._performance_stats = {
            "weight_calculations": 0,
            "cache_hits": 0,
            "avg_processing_time_ms": 0.0,
        }

    async def calculate_affective_weight(
        self,
        emotional_significance: EmotionalSignificance,
        npc_personality: NPCPersonality,
        current_mood: MoodState,
        relationship_level: float,
        memory_age_hours: float = 0.0,
        trust_level: float = 0.0,
        strategy: Optional[AffectiveWeightingStrategy] = None,
    ) -> AffectiveWeight:
        """
        Calculate comprehensive affective weighting for a memory.
        
        Args:
            emotional_significance: Emotional significance analysis
            npc_personality: NPC personality traits
            current_mood: Current mood state
            relationship_level: Current relationship level
            memory_age_hours: Age of memory in hours
            trust_level: Current trust level
            strategy: Weighting strategy to use
        """
        try:
            # Validate inputs
            if not emotional_significance:
                raise InvalidEmotionalDataError("EmotionalSignificance is required")
            if not npc_personality:
                raise InvalidEmotionalDataError("NPC personality is required")
            
            # Validate numerical inputs (basic validation without imports)
            if not (0.0 <= relationship_level <= 1.0):
                raise InvalidEmotionalDataError("relationship_level must be between 0.0 and 1.0")
            if memory_age_hours < 0:
                raise InvalidEmotionalDataError("memory_age_hours cannot be negative")
            if not (0.0 <= trust_level <= 1.0):
                raise InvalidEmotionalDataError("trust_level must be between 0.0 and 1.0")
            
            # Validate mood state (basic validation)
            if not isinstance(current_mood, MoodState):
                raise InvalidEmotionalDataError("current_mood must be a MoodState enum")
            
            # Check for reasonable memory age limits
            if memory_age_hours > EmotionalThresholds.VERY_OLD_MEMORY_HOURS * 24:  # Over 2 years
                logger.warning(f"Memory age unusually large: {memory_age_hours} hours")
                
        except Exception as e:
            raise handle_emotional_memory_error(
                e, "Failed to validate inputs for affective weight calculation"
            )
        
        start_time = time.perf_counter()
        
        # Use provided strategy or default
        strategy = strategy or self.default_strategy
        
        # Create cache key
        cache_key = self._create_weight_cache_key(
            emotional_significance, npc_personality.npc_id, current_mood, strategy
        )
        
        # Check cache
        if cache_key in self._weight_cache:
            cached_weight = self._weight_cache[cache_key]
            # Update context-dependent values
            cached_weight.relationship_amplifier = self._calculate_relationship_amplifier(
                relationship_level, emotional_significance
            )
            cached_weight.recency_boost = self._calculate_recency_boost(
                memory_age_hours, emotional_significance
            )
            cached_weight.final_weight = self._combine_final_weight(cached_weight)
            
            self._performance_stats["cache_hits"] += 1
            return cached_weight
        
        # Calculate base affective weight
        base_weight = self._calculate_base_affective_weight(
            emotional_significance, strategy
        )
        
        # Calculate intensity multiplier
        intensity_multiplier = self._calculate_intensity_multiplier(
            emotional_significance, npc_personality, strategy
        )
        
        # Calculate personality modifier
        personality_modifier = self._calculate_personality_modifier(
            emotional_significance, npc_personality, strategy
        )
        
        # Calculate mood accessibility modifier
        mood_modifier = self._calculate_mood_accessibility_modifier(
            emotional_significance, current_mood, npc_personality
        )
        
        # Calculate contextual factors
        recency_boost = self._calculate_recency_boost(memory_age_hours, emotional_significance)
        relationship_amplifier = self._calculate_relationship_amplifier(
            relationship_level, emotional_significance
        )
        
        # Calculate access controls
        access_threshold = self._calculate_access_threshold(emotional_significance)
        trauma_sensitivity = self._calculate_trauma_sensitivity(
            emotional_significance, npc_personality
        )
        
        # Create affective weight
        affective_weight = AffectiveWeight(
            base_affective_weight=base_weight,
            intensity_multiplier=intensity_multiplier,
            personality_modifier=personality_modifier,
            mood_accessibility_modifier=mood_modifier,
            recency_boost=recency_boost,
            relationship_amplifier=relationship_amplifier,
            formative_importance=emotional_significance.formative_influence,
            access_threshold=access_threshold,
            trauma_sensitivity=trauma_sensitivity,
            final_weight=0.0,  # Will be calculated below
            weighting_strategy=strategy,
            confidence=emotional_significance.confidence_score,
        )
        
        # Calculate final combined weight
        affective_weight.final_weight = self._combine_final_weight(affective_weight)
        
        # Cache result with size management
        if len(self._weight_cache) >= self._max_cache_size:
            # Remove oldest entries (simple FIFO)
            oldest_keys = list(self._weight_cache.keys())[:int(self._max_cache_size * 0.1)]
            for old_key in oldest_keys:
                self._weight_cache.pop(old_key, None)
        
        self._weight_cache[cache_key] = affective_weight
        
        # Update performance stats
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        self._performance_stats["weight_calculations"] += 1
        total_calculations = self._performance_stats["weight_calculations"]
        self._performance_stats["avg_processing_time_ms"] = (
            self._performance_stats["avg_processing_time_ms"] * (total_calculations - 1)
            + processing_time_ms
        ) / total_calculations
        
        return affective_weight

    def _calculate_base_affective_weight(
        self,
        emotional_significance: EmotionalSignificance,
        strategy: AffectiveWeightingStrategy,
    ) -> float:
        """Calculate base affective weight using specified strategy."""
        
        base_significance = emotional_significance.overall_significance
        
        if strategy == AffectiveWeightingStrategy.LINEAR:
            return base_significance
        
        elif strategy == AffectiveWeightingStrategy.EXPONENTIAL:
            # Exponential scaling emphasizes high-emotion memories
            return base_significance ** 0.7  # Slightly compress lower values, boost higher
        
        elif strategy == AffectiveWeightingStrategy.THRESHOLD:
            # Step function at thresholds
            if base_significance >= EmotionalThresholds.HIGH_SIGNIFICANCE_THRESHOLD:
                return 1.0
            elif base_significance >= EmotionalThresholds.FORMATIVE_WEIGHT_THRESHOLD:
                return EmotionalThresholds.HIGH_SIGNIFICANCE_THRESHOLD
            elif base_significance >= (EmotionalThresholds.MODERATE_SIGNIFICANCE - 0.1):
                return EmotionalThresholds.FORMATIVE_WEIGHT_THRESHOLD
            elif base_significance >= (EmotionalThresholds.LOW_SIGNIFICANCE - 0.1):
                return EmotionalThresholds.MODERATE_SIGNIFICANCE - 0.1
            else:
                return EmotionalThresholds.LOW_SIGNIFICANCE - 0.1
        
        else:  # PERSONALITY_ADAPTIVE and MOOD_SENSITIVE use linear as base
            return base_significance

    def _calculate_intensity_multiplier(
        self,
        emotional_significance: EmotionalSignificance,
        npc_personality: NPCPersonality,
        strategy: AffectiveWeightingStrategy,
    ) -> float:
        """Calculate intensity-based multiplier."""
        
        base_intensity = emotional_significance.intensity_score
        
        # Personality-based intensity sensitivity
        emotional_sensitivity = npc_personality.get_trait_strength("emotional_sensitivity")
        if emotional_sensitivity == 0:
            emotional_sensitivity = npc_personality.get_trait_strength("emotional")
        
        # Base multiplier from intensity
        intensity_multiplier = 1.0 + base_intensity * 0.5
        
        # Apply personality sensitivity
        if strategy == AffectiveWeightingStrategy.PERSONALITY_ADAPTIVE:
            intensity_multiplier *= (1.0 + emotional_sensitivity * 0.3)
        
        # Exponential strategy amplifies intensity more
        if strategy == AffectiveWeightingStrategy.EXPONENTIAL:
            intensity_multiplier = 1.0 + (base_intensity ** 0.8) * 0.8
        
        return min(2.0, intensity_multiplier)  # Cap at 2x multiplier

    def _calculate_personality_modifier(
        self,
        emotional_significance: EmotionalSignificance,
        npc_personality: NPCPersonality,
        strategy: AffectiveWeightingStrategy,
    ) -> float:
        """Calculate personality-based modifier."""
        
        if strategy != AffectiveWeightingStrategy.PERSONALITY_ADAPTIVE:
            return 1.0  # No personality adjustment for other strategies
        
        modifier = 1.0
        memory_type = emotional_significance.emotional_type
        
        # Type-specific personality alignments
        if memory_type == EmotionalMemoryType.TRAUMATIC:
            trauma_sensitivity = npc_personality.get_trait_strength("trauma_sensitive")
            anxious = npc_personality.get_trait_strength("anxious")
            modifier *= (1.0 + max(trauma_sensitivity, anxious) * 0.8)
        
        elif memory_type == EmotionalMemoryType.CORE_ATTACHMENT:
            loving = npc_personality.get_trait_strength("loving")
            social = npc_personality.get_trait_strength("social")
            modifier *= (1.0 + max(loving, social) * 0.6)
        
        elif memory_type == EmotionalMemoryType.FORMATIVE:
            analytical = npc_personality.get_trait_strength("analytical")
            reflective = npc_personality.get_trait_strength("reflective")
            modifier *= (1.0 + max(analytical, reflective) * 0.5)
        
        elif memory_type == EmotionalMemoryType.CONFLICT:
            conflict_averse = npc_personality.get_trait_strength("conflict_averse")
            aggressive = npc_personality.get_trait_strength("aggressive")
            # Both conflict-averse and aggressive personalities remember conflicts vividly
            modifier *= (1.0 + max(conflict_averse, aggressive) * 0.7)
        
        elif memory_type in [EmotionalMemoryType.PEAK_POSITIVE, EmotionalMemoryType.BREAKTHROUGH]:
            optimistic = npc_personality.get_trait_strength("optimistic")
            achievement_oriented = npc_personality.get_trait_strength("achievement_oriented")
            modifier *= (1.0 + max(optimistic, achievement_oriented) * 0.4)
        
        # General emotional sensitivity
        emotional_trait = npc_personality.get_trait_strength("emotional")
        modifier *= (1.0 + emotional_trait * 0.3)
        
        return min(2.0, modifier)

    def _calculate_mood_accessibility_modifier(
        self,
        emotional_significance: EmotionalSignificance,
        current_mood: MoodState,
        npc_personality: NPCPersonality,
    ) -> float:
        """Calculate mood-based accessibility modifier."""
        
        mood_accessibility = emotional_significance.mood_accessibility.get(current_mood, 0.5)
        
        # Convert accessibility to weight modifier
        # High accessibility (>0.7) boosts weight, low accessibility (<0.3) reduces it
        if mood_accessibility > 0.7:
            modifier = 1.0 + (mood_accessibility - 0.7) * 1.5  # Up to 1.45x boost
        elif mood_accessibility < 0.3:
            modifier = 0.3 + mood_accessibility * 0.7  # Down to 0.3x weight
        else:
            modifier = 0.7 + mood_accessibility * 0.6  # Linear in middle range
        
        # Personality affects mood sensitivity
        emotional_trait = npc_personality.get_trait_strength("emotional")
        if emotional_trait > 0.5:
            # More emotional personalities have stronger mood effects
            mood_effect = modifier - 1.0
            modifier = 1.0 + mood_effect * (1.0 + emotional_trait * 0.5)
        
        return max(0.1, min(2.0, modifier))

    def _calculate_recency_boost(
        self,
        memory_age_hours: float,
        emotional_significance: EmotionalSignificance,
    ) -> float:
        """Calculate boost for recent emotional memories."""
        
        if memory_age_hours <= 0:
            return 1.0  # No age information
        
        # Recent emotional memories get a boost
        # Boost decays exponentially with time
        max_boost_hours = EmotionalThresholds.RECENT_MEMORY_HOURS  # Maximum boost for memories < 24 hours old
        boost_strength = emotional_significance.intensity_score * 0.5
        
        if memory_age_hours < max_boost_hours:
            decay_factor = memory_age_hours / max_boost_hours
            boost = 1.0 + boost_strength * (1.0 - decay_factor)
        else:
            boost = 1.0  # No boost for older memories
        
        return boost

    def _calculate_relationship_amplifier(
        self,
        relationship_level: float,
        emotional_significance: EmotionalSignificance,
    ) -> float:
        """Calculate relationship-based amplification."""
        
        # Strong relationships amplify emotional memories
        base_amplifier = 1.0 + relationship_level * 0.4
        
        # Relationship-relevant memory types get extra amplification
        if emotional_significance.emotional_type in [
            EmotionalMemoryType.CORE_ATTACHMENT,
            EmotionalMemoryType.TRUST_EVENT,
            EmotionalMemoryType.CONFLICT,
        ]:
            base_amplifier *= 1.2
        
        # Personal relevance increases amplification
        relevance_boost = emotional_significance.personal_relevance * 0.3
        
        return base_amplifier + relevance_boost

    def _calculate_access_threshold(
        self,
        emotional_significance: EmotionalSignificance,
    ) -> float:
        """Calculate trust threshold required for memory access."""
        
        protection_thresholds = {
            MemoryProtectionLevel.PUBLIC: ProtectionMechanismConfig.TRUST_THRESHOLDS['public'],
            MemoryProtectionLevel.PRIVATE: ProtectionMechanismConfig.TRUST_THRESHOLDS['private'],
            MemoryProtectionLevel.SENSITIVE: ProtectionMechanismConfig.TRUST_THRESHOLDS['sensitive'],
            MemoryProtectionLevel.PROTECTED: ProtectionMechanismConfig.TRUST_THRESHOLDS['protected'],
            MemoryProtectionLevel.TRAUMATIC: ProtectionMechanismConfig.TRUST_THRESHOLDS['traumatic'],
        }
        
        base_threshold = protection_thresholds.get(
            emotional_significance.protection_level, 0.3
        )
        
        # Formative memories require extra trust
        formative_adjustment = emotional_significance.formative_influence * 0.1
        
        return min(0.95, base_threshold + formative_adjustment)

    def _calculate_trauma_sensitivity(
        self,
        emotional_significance: EmotionalSignificance,
        npc_personality: NPCPersonality,
    ) -> float:
        """Calculate trauma sensitivity level."""
        
        if emotional_significance.emotional_type != EmotionalMemoryType.TRAUMATIC:
            return 0.0
        
        # Base trauma sensitivity
        base_sensitivity = emotional_significance.overall_significance
        
        # Personality affects trauma sensitivity
        trauma_sensitive_trait = npc_personality.get_trait_strength("trauma_sensitive")
        anxious_trait = npc_personality.get_trait_strength("anxious")
        resilient_trait = npc_personality.get_trait_strength("resilient")
        
        # Trauma-sensitive and anxious personalities have higher sensitivity
        sensitivity_modifier = 1.0 + max(trauma_sensitive_trait, anxious_trait) * 0.5
        
        # Resilient personalities have lower sensitivity
        sensitivity_modifier *= (1.0 - resilient_trait * 0.3)
        
        return min(1.0, base_sensitivity * sensitivity_modifier)

    def _combine_final_weight(self, affective_weight: AffectiveWeight) -> float:
        """Combine all weight factors into final weight."""
        
        # Multiplicative combination of factors
        final_weight = (
            affective_weight.base_affective_weight
            * affective_weight.intensity_multiplier
            * affective_weight.personality_modifier
            * affective_weight.mood_accessibility_modifier
            * affective_weight.recency_boost
            * affective_weight.relationship_amplifier
        )
        
        # Add formative importance (additive)
        final_weight += affective_weight.formative_importance * 0.2
        
        # Trauma sensitivity can reduce final weight if not appropriate context
        if affective_weight.trauma_sensitivity > 0.5:
            # Traumatic memories may be suppressed in certain contexts
            suppression_factor = 1.0 - affective_weight.trauma_sensitivity * 0.3
            final_weight *= suppression_factor
        
        return max(0.01, min(1.0, final_weight))

    async def calculate_mood_based_accessibility(
        self,
        emotional_significance: EmotionalSignificance,
        current_mood: MoodState,
        npc_personality: NPCPersonality,
        access_strategy: MemoryAccessStrategy = MemoryAccessStrategy.BALANCED,
    ) -> MoodBasedAccessibility:
        """Calculate mood-based memory accessibility."""
        
        cache_key = f"{hash(str(emotional_significance))}_{current_mood.value}_{access_strategy.value}"
        
        if cache_key in self._mood_access_cache:
            return self._mood_access_cache[cache_key]
        
        base_accessibility = emotional_significance.mood_accessibility.get(current_mood, 0.5)
        
        # Strategy-based adjustments
        if access_strategy == MemoryAccessStrategy.MOOD_CONGRUENT:
            # Boost memories that match current mood
            mood_congruent_boost = max(0.0, base_accessibility - 0.5) * 2.0
            mood_contrasting_penalty = max(0.0, 0.5 - base_accessibility) * 1.5
        
        elif access_strategy == MemoryAccessStrategy.MOOD_CONTRASTING:
            # Boost memories that contrast current mood (for mood regulation)
            mood_congruent_boost = max(0.0, 0.5 - base_accessibility) * 1.5
            mood_contrasting_penalty = max(0.0, base_accessibility - 0.5) * 1.0
        
        elif access_strategy == MemoryAccessStrategy.THERAPEUTIC:
            # Access memories that help with mood regulation
            therapeutic_value = self._calculate_therapeutic_value(
                emotional_significance, current_mood
            )
            mood_congruent_boost = therapeutic_value * 0.3
            mood_contrasting_penalty = (1.0 - therapeutic_value) * 0.2
        
        elif access_strategy == MemoryAccessStrategy.AVOIDANT:
            # Avoid triggering or difficult memories
            triggering_risk = emotional_significance.triggering_potential
            mood_congruent_boost = 0.0
            mood_contrasting_penalty = triggering_risk * 0.8
        
        else:  # BALANCED
            mood_congruent_boost = max(0.0, base_accessibility - 0.5) * 0.8
            mood_contrasting_penalty = max(0.0, 0.5 - base_accessibility) * 0.6
        
        # Calculate triggering risk
        triggering_risk = self._calculate_triggering_risk(
            emotional_significance, current_mood, npc_personality
        )
        
        # Calculate therapeutic value
        therapeutic_value = self._calculate_therapeutic_value(
            emotional_significance, current_mood
        )
        
        # Adjust accessibility
        adjusted_accessibility = base_accessibility + mood_congruent_boost - mood_contrasting_penalty
        adjusted_accessibility = max(0.0, min(1.0, adjusted_accessibility))
        
        # Apply triggering risk penalty
        if triggering_risk > 0.5:
            adjusted_accessibility *= (1.0 - triggering_risk * 0.3)
        
        accessibility = MoodBasedAccessibility(
            current_mood=current_mood,
            base_accessibility=base_accessibility,
            mood_congruent_boost=mood_congruent_boost,
            mood_contrasting_penalty=mood_contrasting_penalty,
            therapeutic_value=therapeutic_value,
            triggering_risk=triggering_risk,
            adjusted_accessibility=adjusted_accessibility,
            access_strategy=access_strategy,
        )
        
        self._mood_access_cache[cache_key] = accessibility
        return accessibility

    def _calculate_therapeutic_value(
        self,
        emotional_significance: EmotionalSignificance,
        current_mood: MoodState,
    ) -> float:
        """Calculate how therapeutically valuable this memory is for current mood."""
        
        memory_type = emotional_significance.emotional_type
        
        # Therapeutic value patterns based on mood and memory type
        therapeutic_patterns = {
            MoodState.MELANCHOLY: {
                EmotionalMemoryType.PEAK_POSITIVE: 0.8,    # Positive memories help sadness
                EmotionalMemoryType.CORE_ATTACHMENT: 0.7,  # Love memories provide comfort
                EmotionalMemoryType.BREAKTHROUGH: 0.6,     # Achievements provide hope
                EmotionalMemoryType.TRAUMATIC: 0.1,        # Trauma not helpful when sad
            },
            MoodState.ANXIOUS: {
                EmotionalMemoryType.CORE_ATTACHMENT: 0.8,  # Attachment provides security
                EmotionalMemoryType.BREAKTHROUGH: 0.7,     # Achievements build confidence
                EmotionalMemoryType.TRAUMATIC: 0.0,        # Trauma worsens anxiety
                EmotionalMemoryType.CONFLICT: 0.2,         # Conflict memories increase anxiety
            },
            MoodState.ANGRY: {
                EmotionalMemoryType.PEAK_POSITIVE: 0.6,    # Positive memories cool anger
                EmotionalMemoryType.CORE_ATTACHMENT: 0.5,  # Love memories provide perspective
                EmotionalMemoryType.CONFLICT: 0.2,         # Conflict memories fuel anger
            },
            # Add more patterns as needed...
        }
        
        pattern = therapeutic_patterns.get(current_mood, {})
        return pattern.get(memory_type, 0.4)  # Default neutral therapeutic value

    def _calculate_triggering_risk(
        self,
        emotional_significance: EmotionalSignificance,
        current_mood: MoodState,
        npc_personality: NPCPersonality,
    ) -> float:
        """Calculate risk of triggering difficult emotional states."""
        
        base_risk = emotional_significance.triggering_potential
        
        # Memory type risks
        type_risks = {
            EmotionalMemoryType.TRAUMATIC: 0.9,
            EmotionalMemoryType.SIGNIFICANT_LOSS: 0.7,
            EmotionalMemoryType.CONFLICT: 0.6,
            EmotionalMemoryType.TRUST_EVENT: 0.5,  # Can trigger trust issues
        }
        
        type_risk = type_risks.get(emotional_significance.emotional_type, 0.2)
        
        # Mood vulnerability
        vulnerable_moods = {
            MoodState.ANXIOUS: 0.8,
            MoodState.MELANCHOLY: 0.7,
            MoodState.FEARFUL: 0.9,
            MoodState.ANGRY: 0.6,
        }
        
        mood_vulnerability = vulnerable_moods.get(current_mood, 0.3)
        
        # Personality resilience
        resilient = npc_personality.get_trait_strength("resilient")
        trauma_sensitive = npc_personality.get_trait_strength("trauma_sensitive")
        
        personality_vulnerability = 1.0 - resilient + trauma_sensitive
        
        # Combine risk factors
        triggering_risk = base_risk * type_risk * mood_vulnerability * personality_vulnerability
        
        return min(1.0, triggering_risk)

    def _create_weight_cache_key(
        self,
        emotional_significance: EmotionalSignificance,
        npc_id: str,
        current_mood: MoodState,
        strategy: AffectiveWeightingStrategy,
    ) -> str:
        """Create cache key for affective weight."""
        key_data = f"{hash(str(emotional_significance))}_{npc_id}_{current_mood.value}_{strategy.value}"
        return str(hash(key_data))

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        total_requests = self._performance_stats["weight_calculations"] + self._performance_stats["cache_hits"]
        cache_hit_rate = (
            (self._performance_stats["cache_hits"] / total_requests * 100)
            if total_requests > 0
            else 0.0
        )
        
        return {
            "total_weight_calculations": self._performance_stats["weight_calculations"],
            "cache_hits": self._performance_stats["cache_hits"],
            "cache_hit_rate_percent": round(cache_hit_rate, 1),
            "avg_processing_time_ms": round(self._performance_stats["avg_processing_time_ms"], 2),
            "cached_weights": len(self._weight_cache),
            "cached_mood_access": len(self._mood_access_cache),
        }

    def clear_caches(self) -> None:
        """Clear all caches."""
        self._weight_cache.clear()
        self._mood_access_cache.clear()
        self._performance_stats = {
            "weight_calculations": 0,
            "cache_hits": 0,
            "avg_processing_time_ms": 0.0,
        }