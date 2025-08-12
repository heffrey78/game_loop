"""Mood-dependent memory accessibility patterns and mood state management."""

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from game_loop.core.conversation.conversation_models import (
    ConversationContext,
    ConversationExchange,
    NPCPersonality,
)
from game_loop.database.session_factory import DatabaseSessionFactory

from .affective_weighting import AffectiveMemoryWeightingEngine, MoodBasedAccessibility
from .config import MemoryAlgorithmConfig
from .constants import EmotionalThresholds, MoodTransitionPatterns
from .emotional_context import EmotionalSignificance, MoodState
from .emotional_preservation import (
    EmotionalMemoryRecord,
    EmotionalPreservationEngine,
    EmotionalRetrievalQuery,
)
from .exceptions import (
    MoodEngineError,
    InvalidEmotionalDataError,
    PerformanceError,
    handle_emotional_memory_error,
)
from .validation import (
    validate_uuid,
    validate_probability,
    validate_mood_state,
    validate_string_content,
    default_validator,
)

logger = logging.getLogger(__name__)


class MoodTransition(Enum):
    """Types of mood transitions."""

    NATURAL_PROGRESSION = "natural_progression"  # Gradual mood change
    TRIGGERED_RESPONSE = "triggered_response"  # Mood triggered by memory/event
    EXTERNAL_INFLUENCE = "external_influence"  # Mood changed by conversation
    REGULATORY_ATTEMPT = "regulatory_attempt"  # Conscious mood regulation
    STRESS_RESPONSE = "stress_response"  # Response to stress/pressure


@dataclass
class MoodStateRecord:
    """Record of NPC mood state at a specific time."""

    npc_id: str
    mood_state: MoodState
    intensity: float  # 0.0-1.0 intensity of the mood
    stability: float  # 0.0-1.0 how stable/long-lasting the mood is
    trigger_source: str  # What triggered this mood state
    transition_type: MoodTransition  # How the mood changed

    # Context
    timestamp: float = field(default_factory=time.time)
    duration_prediction: float = 0.0  # Predicted duration in hours
    regulation_attempts: int = 0  # Number of attempts to change mood

    # Memory impact
    memory_accessibility_changes: Dict[str, float] = field(default_factory=dict)
    suppressed_memory_types: Set[str] = field(default_factory=set)
    enhanced_memory_types: Set[str] = field(default_factory=set)


@dataclass
class MoodMemoryPattern:
    """Pattern defining how moods affect memory accessibility."""

    mood_state: MoodState
    base_accessibility: float

    # Memory type accessibility modifiers
    enhanced_types: Dict[str, float]  # Memory types that become more accessible
    suppressed_types: Dict[str, float]  # Memory types that become less accessible

    # Threshold adjustments
    trust_threshold_modifier: float  # How trust requirements change
    emotional_intensity_filter: float  # Minimum emotional intensity to access

    # Temporal effects
    recent_memory_boost: float  # Boost for recent memories
    distant_memory_penalty: float  # Penalty for old memories

    # Recovery patterns - memories that help mood regulation
    therapeutic_patterns: List[str]  # Memory types that help this mood
    contraindicated_patterns: List[str]  # Memory types that worsen this mood


class MoodDependentMemoryEngine:
    """Engine for mood-dependent memory accessibility and mood state management."""

    def __init__(
        self,
        session_factory: DatabaseSessionFactory,
        config: MemoryAlgorithmConfig,
        affective_engine: AffectiveMemoryWeightingEngine,
        preservation_engine: EmotionalPreservationEngine,
    ):
        self.session_factory = session_factory
        self.config = config
        self.affective_engine = affective_engine
        self.preservation_engine = preservation_engine

        # Mood state tracking
        self._current_moods: Dict[str, MoodStateRecord] = {}
        self._mood_history: Dict[str, List[MoodStateRecord]] = {}

        # Memory accessibility patterns
        self._mood_patterns = self._initialize_mood_patterns()

        # Caches with size limits
        self._accessibility_cache: Dict[str, Dict[str, float]] = {}
        self._mood_prediction_cache: Dict[str, Tuple[MoodState, float]] = {}
        self._max_cache_size = EmotionalThresholds.DEFAULT_CACHE_SIZE

        # Performance tracking
        self._performance_stats = {
            "mood_updates": 0,
            "accessibility_calculations": 0,
            "mood_predictions": 0,
            "therapeutic_retrievals": 0,
            "cache_hits": 0,
            "avg_processing_time_ms": 0.0,
        }

    async def update_npc_mood(
        self,
        npc_id: uuid.UUID,
        new_mood: MoodState,
        intensity: float,
        trigger_source: str,
        transition_type: MoodTransition = MoodTransition.NATURAL_PROGRESSION,
        conversation_context: Optional[ConversationContext] = None,
    ) -> MoodStateRecord:
        """Update NPC mood state and calculate memory accessibility changes."""
        try:
            # Validate inputs
            if not npc_id:
                raise InvalidEmotionalDataError("NPC ID is required")

            # Validate inputs using validation utilities
            npc_id_str = str(validate_uuid(npc_id, "npc_id"))
            new_mood = validate_mood_state(new_mood, "new_mood")
            intensity = validate_probability(intensity, "intensity")
            trigger_source = validate_string_content(
                trigger_source, "trigger_source", min_length=1, max_length=200
            )

            if not isinstance(transition_type, MoodTransition):
                raise InvalidEmotionalDataError(
                    f"Invalid transition_type: {transition_type}"
                )

        except Exception as e:
            raise handle_emotional_memory_error(
                e, "Failed to validate inputs for mood update"
            )

        start_time = time.perf_counter()

        try:
            # Get current mood for comparison
            previous_mood = self._current_moods.get(npc_id_str)

            # Calculate stability based on mood history
            stability = self._calculate_mood_stability(npc_id_str, new_mood, intensity)

            # Predict duration of this mood
            duration_prediction = self._predict_mood_duration(
                new_mood, intensity, stability, previous_mood
            )

            # Calculate memory accessibility changes
            accessibility_changes = await self._calculate_accessibility_changes(
                npc_id_str, new_mood, previous_mood, intensity, conversation_context
            )

            # Create mood state record
            mood_record = MoodStateRecord(
                npc_id=npc_id_str,
                mood_state=new_mood,
                intensity=intensity,
                stability=stability,
                trigger_source=trigger_source,
                transition_type=transition_type,
                duration_prediction=duration_prediction,
                memory_accessibility_changes=accessibility_changes,
            )

            # Update current mood
            self._current_moods[npc_id_str] = mood_record

            # Add to mood history
            if npc_id_str not in self._mood_history:
                self._mood_history[npc_id_str] = []
            self._mood_history[npc_id_str].append(mood_record)

            # Limit mood history size using constants
            max_history_size = (
                EmotionalThresholds.MAX_MEMORY_CAPACITY // 100
            )  # 100 entries max
            if len(self._mood_history[npc_id_str]) > max_history_size:
                self._mood_history[npc_id_str] = self._mood_history[npc_id_str][
                    -(max_history_size // 2) :
                ]

            # Clear relevant caches
            self._clear_npc_caches(npc_id_str)

            # Update performance stats
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            self._performance_stats["mood_updates"] += 1
            self._update_avg_processing_time(processing_time_ms)

            logger.debug(
                f"Updated mood for NPC {npc_id}: {new_mood.value} (intensity: {intensity:.2f})"
            )
            return mood_record

        except Exception as e:
            raise handle_emotional_memory_error(
                e, f"Failed to update mood for NPC {npc_id}"
            )

    async def get_mood_adjusted_memories(
        self,
        npc_id: uuid.UUID,
        retrieval_query: EmotionalRetrievalQuery,
        include_therapeutic: bool = True,
    ) -> Tuple[List[EmotionalMemoryRecord], Dict[str, Any]]:
        """Retrieve memories adjusted for current mood state with accessibility filtering."""

        start_time = time.perf_counter()
        npc_id_str = str(npc_id)

        # Get current mood
        current_mood_record = self._current_moods.get(npc_id_str)
        if not current_mood_record:
            # Default to neutral mood
            current_mood_record = MoodStateRecord(
                npc_id=npc_id_str,
                mood_state=MoodState.NEUTRAL,
                intensity=EmotionalThresholds.MODERATE_SIGNIFICANCE,
                stability=EmotionalThresholds.HIGH_MOOD_ACCESSIBILITY,
                trigger_source="default",
                transition_type=MoodTransition.NATURAL_PROGRESSION,
            )

        # Update query with mood context
        mood_adjusted_query = self._adjust_query_for_mood(
            retrieval_query, current_mood_record
        )

        # Retrieve base memories
        retrieval_result = await self.preservation_engine.retrieve_emotional_memories(
            npc_id, mood_adjusted_query
        )

        # Apply mood-based accessibility filtering
        accessible_memories = await self._filter_by_mood_accessibility(
            retrieval_result.emotional_records,
            current_mood_record,
            npc_id_str,
        )

        # Add therapeutic memories if requested and appropriate
        if include_therapeutic and self._should_include_therapeutic_memories(
            current_mood_record
        ):
            therapeutic_memories = await self._get_therapeutic_memories(
                npc_id, current_mood_record, mood_adjusted_query
            )
            accessible_memories.extend(therapeutic_memories)

        # Remove duplicates and sort by combined relevance and mood accessibility
        accessible_memories = self._deduplicate_and_sort_memories(
            accessible_memories, current_mood_record
        )

        # Create mood context metadata
        mood_context = {
            "current_mood": current_mood_record.mood_state.value,
            "mood_intensity": current_mood_record.intensity,
            "mood_stability": current_mood_record.stability,
            "accessibility_adjustments": current_mood_record.memory_accessibility_changes,
            "therapeutic_included": include_therapeutic,
            "total_accessible": len(accessible_memories),
            "mood_duration_prediction": current_mood_record.duration_prediction,
        }

        # Update performance stats
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        self._performance_stats["accessibility_calculations"] += 1
        self._update_avg_processing_time(processing_time_ms)

        return accessible_memories[: mood_adjusted_query.max_results], mood_context

    async def predict_mood_impact_of_memory(
        self,
        npc_id: uuid.UUID,
        memory_record: EmotionalMemoryRecord,
        current_context: Optional[ConversationContext] = None,
    ) -> Dict[str, Any]:
        """Predict how accessing a specific memory will affect NPC mood."""

        npc_id_str = str(npc_id)
        current_mood = self._current_moods.get(npc_id_str)

        if not current_mood:
            return {"prediction": "neutral", "confidence": 0.0}

        # Analyze memory emotional content
        memory_emotion_type = memory_record.emotional_significance.emotional_type
        memory_intensity = memory_record.emotional_significance.intensity_score
        memory_triggering = memory_record.emotional_significance.triggering_potential

        # Calculate mood impact based on current state and memory characteristics
        impact_prediction = self._calculate_mood_impact_prediction(
            current_mood, memory_record, current_context
        )

        return {
            "predicted_mood_change": impact_prediction["mood_change"],
            "intensity_change": impact_prediction["intensity_change"],
            "stability_impact": impact_prediction["stability_impact"],
            "triggering_risk": memory_triggering,
            "therapeutic_potential": impact_prediction["therapeutic_potential"],
            "confidence": impact_prediction["confidence"],
            "recommended_access": impact_prediction["recommended"],
        }

    def get_current_mood(self, npc_id: uuid.UUID) -> Optional[MoodStateRecord]:
        """Get current mood state for an NPC."""
        return self._current_moods.get(str(npc_id))

    def get_mood_history(
        self, npc_id: uuid.UUID, hours_back: float = 24.0
    ) -> List[MoodStateRecord]:
        """Get mood history for an NPC within specified timeframe."""
        npc_id_str = str(npc_id)
        if npc_id_str not in self._mood_history:
            return []

        cutoff_time = time.time() - (hours_back * 3600)
        return [
            mood
            for mood in self._mood_history[npc_id_str]
            if mood.timestamp >= cutoff_time
        ]

    async def recommend_mood_regulation_memories(
        self,
        npc_id: uuid.UUID,
        target_mood: MoodState,
        max_memories: int = 5,
    ) -> List[Tuple[EmotionalMemoryRecord, float]]:
        """Recommend memories that could help regulate mood toward target state."""

        npc_id_str = str(npc_id)
        current_mood = self._current_moods.get(npc_id_str)

        if not current_mood:
            return []

        # Create query for therapeutic memories
        therapeutic_query = EmotionalRetrievalQuery(
            target_mood=target_mood,
            therapeutic_preference=True,
            max_results=max_memories * 2,  # Get extras for filtering
            trust_level=0.5,  # Moderate trust required for regulation
        )

        # Retrieve potential therapeutic memories
        memories_result = await self.preservation_engine.retrieve_emotional_memories(
            npc_id, therapeutic_query
        )

        # Score memories for mood regulation potential
        regulation_scores = []
        for memory in memories_result.emotional_records:
            regulation_score = self._calculate_mood_regulation_score(
                memory, current_mood.mood_state, target_mood
            )
            if regulation_score > 0.3:  # Only include memories with decent potential
                regulation_scores.append((memory, regulation_score))

        # Sort by regulation potential and return top results
        regulation_scores.sort(key=lambda x: x[1], reverse=True)

        self._performance_stats["therapeutic_retrievals"] += 1
        return regulation_scores[:max_memories]

    def _initialize_mood_patterns(self) -> Dict[MoodState, MoodMemoryPattern]:
        """Initialize mood-based memory accessibility patterns."""

        patterns = {}

        # Joyful mood pattern - enhances positive memories
        patterns[MoodState.JOYFUL] = MoodMemoryPattern(
            mood_state=MoodState.JOYFUL,
            base_accessibility=0.8,
            enhanced_types={
                "peak_positive": 0.3,
                "breakthrough": 0.2,
                "everyday_positive": 0.15,
            },
            suppressed_types={
                "traumatic": -0.5,
                "significant_loss": -0.3,
                "conflict": -0.2,
            },
            trust_threshold_modifier=-0.1,  # Lower trust needed when happy
            emotional_intensity_filter=0.2,
            recent_memory_boost=0.2,
            distant_memory_penalty=-0.1,
            therapeutic_patterns=["peak_positive", "core_attachment"],
            contraindicated_patterns=["traumatic", "significant_loss"],
        )

        # Melancholy mood pattern - enhances loss/sad memories
        patterns[MoodState.MELANCHOLY] = MoodMemoryPattern(
            mood_state=MoodState.MELANCHOLY,
            base_accessibility=0.6,
            enhanced_types={
                "significant_loss": 0.4,
                "nostalgic": 0.3,
                "core_attachment": 0.2,
            },
            suppressed_types={"peak_positive": -0.3, "everyday_positive": -0.2},
            trust_threshold_modifier=0.1,  # Higher trust needed when sad
            emotional_intensity_filter=0.3,
            recent_memory_boost=0.1,
            distant_memory_penalty=-0.15,
            therapeutic_patterns=["core_attachment", "breakthrough"],
            contraindicated_patterns=["significant_loss", "traumatic"],
        )

        # Anxious mood pattern - heightens threat-related memories
        patterns[MoodState.ANXIOUS] = MoodMemoryPattern(
            mood_state=MoodState.ANXIOUS,
            base_accessibility=0.5,
            enhanced_types={"conflict": 0.3, "traumatic": 0.4, "trust_event": 0.2},
            suppressed_types={"peak_positive": -0.2, "breakthrough": -0.15},
            trust_threshold_modifier=0.2,  # Much higher trust needed when anxious
            emotional_intensity_filter=0.4,
            recent_memory_boost=0.25,  # Recent threats very salient
            distant_memory_penalty=-0.05,
            therapeutic_patterns=["core_attachment", "peak_positive"],
            contraindicated_patterns=["traumatic", "conflict"],
        )

        # Angry mood pattern - enhances conflict and injustice memories
        patterns[MoodState.ANGRY] = MoodMemoryPattern(
            mood_state=MoodState.ANGRY,
            base_accessibility=0.7,
            enhanced_types={"conflict": 0.5, "trust_event": 0.3, "traumatic": 0.2},
            suppressed_types={"peak_positive": -0.4, "core_attachment": -0.2},
            trust_threshold_modifier=0.15,
            emotional_intensity_filter=0.5,
            recent_memory_boost=0.3,
            distant_memory_penalty=-0.1,
            therapeutic_patterns=["core_attachment", "formative"],
            contraindicated_patterns=["conflict", "trust_event"],
        )

        # Fearful mood pattern - heightens threat memories
        patterns[MoodState.FEARFUL] = MoodMemoryPattern(
            mood_state=MoodState.FEARFUL,
            base_accessibility=0.4,
            enhanced_types={"traumatic": 0.6, "conflict": 0.3, "significant_loss": 0.2},
            suppressed_types={"peak_positive": -0.3, "breakthrough": -0.2},
            trust_threshold_modifier=0.3,  # Highest trust requirements when fearful
            emotional_intensity_filter=0.6,
            recent_memory_boost=0.4,
            distant_memory_penalty=0.0,
            therapeutic_patterns=["core_attachment", "breakthrough"],
            contraindicated_patterns=["traumatic", "conflict"],
        )

        # Neutral mood pattern - balanced accessibility
        patterns[MoodState.NEUTRAL] = MoodMemoryPattern(
            mood_state=MoodState.NEUTRAL,
            base_accessibility=0.6,
            enhanced_types={},
            suppressed_types={},
            trust_threshold_modifier=0.0,
            emotional_intensity_filter=0.3,
            recent_memory_boost=0.1,
            distant_memory_penalty=-0.1,
            therapeutic_patterns=[],
            contraindicated_patterns=[],
        )

        # Add other moods with similar patterns...
        # Content, Excited, Nostalgic, Hopeful can be added as needed

        return patterns

    def _calculate_mood_stability(
        self, npc_id: str, new_mood: MoodState, intensity: float
    ) -> float:
        """Calculate how stable the new mood is likely to be."""

        mood_history = self._mood_history.get(npc_id, [])
        if not mood_history:
            return 0.5  # Default stability for new NPCs

        # Look at recent mood changes
        recent_moods = [
            m for m in mood_history[-10:] if time.time() - m.timestamp < 3600
        ]

        if not recent_moods:
            return 0.7  # Stable if no recent changes

        # Calculate stability based on mood consistency and transition frequency
        mood_changes = 0
        for i in range(1, len(recent_moods)):
            if recent_moods[i].mood_state != recent_moods[i - 1].mood_state:
                mood_changes += 1

        # More changes = less stability
        change_penalty = min(0.5, mood_changes * 0.1)

        # Higher intensity moods tend to be less stable
        intensity_penalty = intensity * 0.2

        stability = max(0.1, 1.0 - change_penalty - intensity_penalty)
        return stability

    def _predict_mood_duration(
        self,
        mood: MoodState,
        intensity: float,
        stability: float,
        previous_mood: Optional[MoodStateRecord],
    ) -> float:
        """Predict how long this mood state will last (in hours)."""

        # Base duration patterns by mood type
        base_durations = {
            MoodState.JOYFUL: 2.0,
            MoodState.CONTENT: 6.0,
            MoodState.NEUTRAL: 8.0,
            MoodState.MELANCHOLY: 4.0,
            MoodState.ANXIOUS: 3.0,
            MoodState.ANGRY: 1.5,
            MoodState.FEARFUL: 2.5,
            MoodState.EXCITED: 1.0,
            MoodState.NOSTALGIC: 3.0,
            MoodState.HOPEFUL: 4.0,
        }

        base_duration = base_durations.get(mood, 3.0)

        # Adjust for intensity (higher intensity = shorter duration for most moods)
        if mood in [MoodState.ANGRY, MoodState.EXCITED, MoodState.FEARFUL]:
            intensity_modifier = 1.0 - (
                intensity * 0.5
            )  # High intensity burns out faster
        else:
            intensity_modifier = 1.0 - (
                intensity * 0.2
            )  # Moderate effect for other moods

        # Adjust for stability
        stability_modifier = 0.5 + (stability * 1.5)

        # Consider previous mood - similar moods last longer
        continuity_modifier = 1.0
        if previous_mood and previous_mood.mood_state == mood:
            continuity_modifier = 1.3  # Continuing same mood extends duration

        predicted_duration = (
            base_duration
            * intensity_modifier
            * stability_modifier
            * continuity_modifier
        )
        return max(
            0.5, min(24.0, predicted_duration)
        )  # Clamp between 30 minutes and 24 hours

    async def _calculate_accessibility_changes(
        self,
        npc_id: str,
        new_mood: MoodState,
        previous_mood: Optional[MoodStateRecord],
        intensity: float,
        conversation_context: Optional[ConversationContext],
    ) -> Dict[str, float]:
        """Calculate how memory accessibility changes with new mood state."""

        pattern = self._mood_patterns.get(new_mood)
        if not pattern:
            return {}

        changes = {}

        # Apply base accessibility changes
        changes["base_accessibility"] = pattern.base_accessibility

        # Apply type-specific changes
        for mem_type, modifier in pattern.enhanced_types.items():
            changes[f"enhanced_{mem_type}"] = modifier * intensity

        for mem_type, modifier in pattern.suppressed_types.items():
            changes[f"suppressed_{mem_type}"] = modifier * intensity

        # Apply threshold changes
        changes["trust_threshold_change"] = pattern.trust_threshold_modifier * intensity
        changes["intensity_filter_change"] = (
            pattern.emotional_intensity_filter * intensity
        )

        # Temporal effect changes
        changes["recent_boost"] = pattern.recent_memory_boost * intensity
        changes["distant_penalty"] = pattern.distant_memory_penalty * intensity

        return changes

    def _adjust_query_for_mood(
        self, query: EmotionalRetrievalQuery, mood_record: MoodStateRecord
    ) -> EmotionalRetrievalQuery:
        """Adjust retrieval query based on current mood state."""

        pattern = self._mood_patterns.get(mood_record.mood_state)
        if not pattern:
            return query

        # Create adjusted query
        adjusted_query = EmotionalRetrievalQuery(
            target_mood=mood_record.mood_state,
            emotional_types=query.emotional_types,
            significance_threshold=max(
                query.significance_threshold, pattern.emotional_intensity_filter
            ),
            protection_level_max=query.protection_level_max,
            trust_level=query.trust_level + pattern.trust_threshold_modifier,
            max_results=query.max_results,
            include_triggering=query.include_triggering,
            therapeutic_preference=query.therapeutic_preference,
            avoid_recent_triggers=query.avoid_recent_triggers,
            max_age_hours=query.max_age_hours,
            min_age_hours=query.min_age_hours,
        )

        return adjusted_query

    async def _filter_by_mood_accessibility(
        self,
        memories: List[EmotionalMemoryRecord],
        mood_record: MoodStateRecord,
        npc_id: str,
    ) -> List[EmotionalMemoryRecord]:
        """Filter memories based on mood-dependent accessibility."""

        pattern = self._mood_patterns.get(mood_record.mood_state)
        if not pattern:
            return memories

        filtered_memories = []

        for memory in memories:
            # Calculate mood-adjusted accessibility
            accessibility = await self._calculate_memory_accessibility(
                memory, mood_record, pattern
            )

            # Apply accessibility threshold
            if accessibility >= 0.3:  # Minimum accessibility threshold
                filtered_memories.append(memory)

        return filtered_memories

    async def _calculate_memory_accessibility(
        self,
        memory: EmotionalMemoryRecord,
        mood_record: MoodStateRecord,
        pattern: MoodMemoryPattern,
    ) -> float:
        """Calculate how accessible a memory is in the current mood state."""

        base_accessibility = pattern.base_accessibility

        # Get memory type
        memory_type = memory.emotional_significance.emotional_type.value

        # Apply type-specific modifiers
        type_modifier = 0.0
        if memory_type in pattern.enhanced_types:
            type_modifier = pattern.enhanced_types[memory_type]
        elif memory_type in pattern.suppressed_types:
            type_modifier = pattern.suppressed_types[memory_type]

        # Apply intensity scaling
        intensity_modifier = type_modifier * mood_record.intensity

        # Apply recency effects
        memory_age_hours = (time.time() - memory.preserved_at) / 3600
        if memory_age_hours < 24:  # Recent memory
            recency_modifier = pattern.recent_memory_boost * mood_record.intensity
        elif memory_age_hours > 168:  # Old memory (>1 week)
            recency_modifier = pattern.distant_memory_penalty * mood_record.intensity
        else:
            recency_modifier = 0.0

        # Calculate final accessibility
        final_accessibility = base_accessibility + intensity_modifier + recency_modifier
        return max(0.0, min(1.0, final_accessibility))

    def _should_include_therapeutic_memories(
        self, mood_record: MoodStateRecord
    ) -> bool:
        """Determine if therapeutic memories should be included for current mood."""

        # Include therapeutic memories for negative moods or low stability
        negative_moods = {
            MoodState.MELANCHOLY,
            MoodState.ANXIOUS,
            MoodState.ANGRY,
            MoodState.FEARFUL,
        }

        return (
            mood_record.mood_state in negative_moods
            or mood_record.stability < 0.4
            or mood_record.intensity > 0.7
        )

    async def _get_therapeutic_memories(
        self,
        npc_id: uuid.UUID,
        mood_record: MoodStateRecord,
        base_query: EmotionalRetrievalQuery,
    ) -> List[EmotionalMemoryRecord]:
        """Get memories that could provide therapeutic value for current mood."""

        pattern = self._mood_patterns.get(mood_record.mood_state)
        if not pattern or not pattern.therapeutic_patterns:
            return []

        # Create therapeutic query
        therapeutic_query = EmotionalRetrievalQuery(
            target_mood=mood_record.mood_state,
            significance_threshold=0.4,  # Lower threshold for therapeutic memories
            therapeutic_preference=True,
            max_results=3,  # Limit therapeutic memories
            trust_level=base_query.trust_level * 0.8,  # Lower trust requirement
        )

        # Retrieve therapeutic memories
        therapeutic_result = await self.preservation_engine.retrieve_emotional_memories(
            npc_id, therapeutic_query
        )

        # Filter for actually therapeutic patterns
        therapeutic_memories = []
        for memory in therapeutic_result.emotional_records:
            memory_type = memory.emotional_significance.emotional_type.value
            if memory_type in pattern.therapeutic_patterns:
                therapeutic_memories.append(memory)

        return therapeutic_memories

    def _deduplicate_and_sort_memories(
        self,
        memories: List[EmotionalMemoryRecord],
        mood_record: MoodStateRecord,
    ) -> List[EmotionalMemoryRecord]:
        """Remove duplicates and sort memories by relevance to current mood."""

        # Remove duplicates by exchange_id
        seen_exchanges = set()
        unique_memories = []
        for memory in memories:
            if memory.exchange_id not in seen_exchanges:
                seen_exchanges.add(memory.exchange_id)
                unique_memories.append(memory)

        # Sort by combined relevance score
        def relevance_score(memory: EmotionalMemoryRecord) -> float:
            # Base significance
            score = memory.emotional_significance.overall_significance

            # Mood accessibility bonus
            mood_accessibility = memory.mood_accessibility.get(
                mood_record.mood_state, 0.5
            )
            score += mood_accessibility * 0.3

            # Recency bonus
            age_hours = (time.time() - memory.preserved_at) / 3600
            if age_hours < 24:
                score += 0.2 * (1.0 - age_hours / 24)

            return score

        unique_memories.sort(key=relevance_score, reverse=True)
        return unique_memories

    def _calculate_mood_impact_prediction(
        self,
        current_mood: MoodStateRecord,
        memory_record: EmotionalMemoryRecord,
        conversation_context: Optional[ConversationContext],
    ) -> Dict[str, Any]:
        """Calculate predicted impact of accessing a memory on mood state."""

        memory_emotion_type = memory_record.emotional_significance.emotional_type
        memory_intensity = memory_record.emotional_significance.intensity_score
        memory_relationship_impact = (
            memory_record.emotional_significance.relationship_impact
        )

        # Base impact calculations
        mood_congruence = self._calculate_mood_memory_congruence(
            current_mood.mood_state, memory_emotion_type
        )

        intensity_impact = memory_intensity * 0.5 - 0.25  # Range: -0.25 to 0.25

        # Stability impact
        stability_change = (
            -abs(intensity_impact) * 0.3
        )  # Accessing memories reduces stability

        # Therapeutic potential
        pattern = self._mood_patterns.get(current_mood.mood_state)
        therapeutic_potential = 0.0
        if pattern and memory_emotion_type.value in pattern.therapeutic_patterns:
            therapeutic_potential = 0.7
        elif pattern and memory_emotion_type.value in pattern.contraindicated_patterns:
            therapeutic_potential = -0.8

        # Overall recommendation
        impact_score = mood_congruence + therapeutic_potential
        recommended = impact_score > 0.3

        return {
            "mood_change": mood_congruence,
            "intensity_change": intensity_impact,
            "stability_impact": stability_change,
            "therapeutic_potential": therapeutic_potential,
            "confidence": min(
                0.9, memory_record.emotional_significance.confidence_score + 0.1
            ),
            "recommended": recommended,
        }

    def _calculate_mood_memory_congruence(self, mood: MoodState, memory_type) -> float:
        """Calculate how congruent a memory type is with current mood."""

        # Mood-memory congruence matrix
        congruence_matrix = {
            MoodState.JOYFUL: {
                "peak_positive": 0.8,
                "breakthrough": 0.6,
                "core_attachment": 0.7,
                "everyday_positive": 0.5,
                "traumatic": -0.9,
                "significant_loss": -0.7,
                "conflict": -0.5,
            },
            MoodState.MELANCHOLY: {
                "significant_loss": 0.8,
                "core_attachment": 0.4,
                "peak_positive": -0.6,
                "traumatic": 0.3,
                "everyday_positive": -0.4,
            },
            MoodState.ANXIOUS: {
                "traumatic": 0.9,
                "conflict": 0.7,
                "trust_event": 0.6,
                "peak_positive": -0.5,
                "breakthrough": -0.4,
            },
            # Add more mappings as needed...
        }

        mood_mappings = congruence_matrix.get(mood, {})
        return mood_mappings.get(memory_type.value, 0.0)

    def _calculate_mood_regulation_score(
        self,
        memory: EmotionalMemoryRecord,
        current_mood: MoodState,
        target_mood: MoodState,
    ) -> float:
        """Calculate how well a memory could help regulate from current to target mood."""

        memory_type = memory.emotional_significance.emotional_type

        # Distance between current and target mood
        mood_distance = self._calculate_mood_distance(current_mood, target_mood)

        # How well memory bridges this gap
        current_congruence = abs(
            self._calculate_mood_memory_congruence(current_mood, memory_type)
        )
        target_congruence = self._calculate_mood_memory_congruence(
            target_mood, memory_type
        )

        # Regulation potential = ability to bridge mood gap
        regulation_potential = target_congruence - (current_congruence * 0.5)

        # Adjust for memory strength
        memory_strength = memory.emotional_significance.overall_significance

        regulation_score = (
            regulation_potential * memory_strength * (1.0 / mood_distance)
        )
        return max(0.0, min(1.0, regulation_score))

    def _calculate_mood_distance(self, mood1: MoodState, mood2: MoodState) -> float:
        """Calculate conceptual distance between two mood states."""

        if mood1 == mood2:
            return 0.1  # Small distance to avoid division by zero

        # Simplified mood space - could be enhanced with more sophisticated modeling
        mood_positions = {
            MoodState.JOYFUL: (1.0, 1.0),  # High energy, high valence
            MoodState.EXCITED: (1.0, 0.8),  # High energy, positive
            MoodState.CONTENT: (0.3, 0.7),  # Low energy, positive
            MoodState.HOPEFUL: (0.5, 0.6),  # Medium energy, positive
            MoodState.NEUTRAL: (0.0, 0.0),  # Neutral
            MoodState.NOSTALGIC: (-0.2, 0.3),  # Low energy, slightly positive
            MoodState.MELANCHOLY: (-0.8, -0.6),  # Low energy, negative
            MoodState.ANXIOUS: (0.7, -0.7),  # High energy, negative
            MoodState.ANGRY: (0.9, -0.8),  # High energy, very negative
            MoodState.FEARFUL: (0.8, -0.9),  # High energy, very negative
        }

        pos1 = mood_positions.get(mood1, (0.0, 0.0))
        pos2 = mood_positions.get(mood2, (0.0, 0.0))

        # Euclidean distance
        distance = ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
        return max(0.1, distance)

    def _clear_npc_caches(self, npc_id: str) -> None:
        """Clear caches related to a specific NPC."""
        if npc_id in self._accessibility_cache:
            del self._accessibility_cache[npc_id]

        # Clear mood prediction cache entries for this NPC
        keys_to_remove = [
            key for key in self._mood_prediction_cache if key.startswith(npc_id)
        ]
        for key in keys_to_remove:
            del self._mood_prediction_cache[key]

    def _update_avg_processing_time(self, processing_time_ms: float) -> None:
        """Update average processing time statistic."""
        total_operations = sum(
            [
                self._performance_stats["mood_updates"],
                self._performance_stats["accessibility_calculations"],
                self._performance_stats["mood_predictions"],
            ]
        )

        if total_operations > 0:
            current_avg = self._performance_stats["avg_processing_time_ms"]
            self._performance_stats["avg_processing_time_ms"] = (
                current_avg * (total_operations - 1) + processing_time_ms
            ) / total_operations

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the mood-dependent memory engine."""
        total_operations = sum(
            [
                self._performance_stats["mood_updates"],
                self._performance_stats["accessibility_calculations"],
                self._performance_stats["mood_predictions"],
                self._performance_stats["therapeutic_retrievals"],
            ]
        )

        cache_hit_rate = (
            (
                self._performance_stats["cache_hits"]
                / (total_operations + self._performance_stats["cache_hits"])
                * 100
            )
            if total_operations > 0
            else 0.0
        )

        return {
            "mood_updates": self._performance_stats["mood_updates"],
            "accessibility_calculations": self._performance_stats[
                "accessibility_calculations"
            ],
            "mood_predictions": self._performance_stats["mood_predictions"],
            "therapeutic_retrievals": self._performance_stats["therapeutic_retrievals"],
            "cache_hits": self._performance_stats["cache_hits"],
            "cache_hit_rate_percent": round(cache_hit_rate, 1),
            "avg_processing_time_ms": round(
                self._performance_stats["avg_processing_time_ms"], 2
            ),
            "active_npc_moods": len(self._current_moods),
            "total_mood_history": sum(
                len(history) for history in self._mood_history.values()
            ),
            "accessibility_cache_size": len(self._accessibility_cache),
        }

    def clear_caches(self) -> None:
        """Clear all caches."""
        self._accessibility_cache.clear()
        self._mood_prediction_cache.clear()
        self._performance_stats = {
            "mood_updates": 0,
            "accessibility_calculations": 0,
            "mood_predictions": 0,
            "therapeutic_retrievals": 0,
            "cache_hits": 0,
            "avg_processing_time_ms": 0.0,
        }
