"""Emotional context preservation and retrieval mechanisms for memory systems."""

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from game_loop.core.conversation.conversation_models import ConversationExchange
from game_loop.database.models.conversation import (
    EmotionalContext as EmotionalContextModel,
)
from game_loop.database.session_factory import DatabaseSessionFactory
from sqlalchemy import select, and_, or_, desc, func
from sqlalchemy.orm import Session

from .affective_weighting import (
    AffectiveWeight,
    AffectiveWeightingStrategy,
    MoodBasedAccessibility,
)
from .constants import EmotionalThresholds, ProtectionMechanismConfig
from .emotional_context import (
    EmotionalSignificance,
    MoodState,
    EmotionalMemoryType,
    MemoryProtectionLevel,
)
from .exceptions import (
    MemoryPreservationError,
    InvalidEmotionalDataError,
    PerformanceError,
    handle_emotional_memory_error,
)
from .validation import (
    validate_uuid,
    validate_probability,
    validate_positive_number,
    validate_string_content,
    default_validator,
)

logger = logging.getLogger(__name__)


@dataclass
class EmotionalMemoryRecord:
    """Complete emotional memory record with context and preservation metadata."""

    # Core identifiers
    exchange_id: str
    emotional_context_id: Optional[str] = None

    # Emotional analysis
    emotional_significance: EmotionalSignificance
    affective_weight: AffectiveWeight
    mood_accessibility: Dict[MoodState, float] = field(default_factory=dict)

    # Preservation metadata
    preserved_at: float = field(default_factory=time.time)
    preservation_version: str = "1.0"
    preservation_confidence: float = 0.0

    # Retrieval optimization
    retrieval_frequency: int = 0
    last_retrieved: Optional[float] = None
    retrieval_contexts: List[str] = field(default_factory=list)

    # Emotional state tracking
    emotional_evolution: List[Dict[str, Any]] = field(default_factory=list)
    triggering_associations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "exchange_id": self.exchange_id,
            "emotional_context_id": self.emotional_context_id,
            "emotional_significance": {
                "overall_significance": self.emotional_significance.overall_significance,
                "emotional_type": self.emotional_significance.emotional_type.value,
                "intensity_score": self.emotional_significance.intensity_score,
                "personal_relevance": self.emotional_significance.personal_relevance,
                "relationship_impact": self.emotional_significance.relationship_impact,
                "formative_influence": self.emotional_significance.formative_influence,
                "protection_level": self.emotional_significance.protection_level.value,
                "decay_resistance": self.emotional_significance.decay_resistance,
                "triggering_potential": self.emotional_significance.triggering_potential,
                "confidence_score": self.emotional_significance.confidence_score,
                "contributing_factors": self.emotional_significance.contributing_factors,
            },
            "affective_weight": self.affective_weight.to_dict(),
            "mood_accessibility": {
                mood.value: weight for mood, weight in self.mood_accessibility.items()
            },
            "preserved_at": self.preserved_at,
            "preservation_version": self.preservation_version,
            "preservation_confidence": self.preservation_confidence,
            "retrieval_frequency": self.retrieval_frequency,
            "last_retrieved": self.last_retrieved,
            "retrieval_contexts": self.retrieval_contexts,
            "emotional_evolution": self.emotional_evolution,
            "triggering_associations": self.triggering_associations,
        }


@dataclass
class EmotionalRetrievalQuery:
    """Query for retrieving emotionally significant memories."""

    # Target emotional characteristics
    target_mood: Optional[MoodState] = None
    emotional_types: List[EmotionalMemoryType] = field(default_factory=list)
    significance_threshold: float = EmotionalThresholds.LOW_SIGNIFICANCE

    # Context filters
    protection_level_max: MemoryProtectionLevel = MemoryProtectionLevel.PROTECTED
    trust_level: float = 0.0
    relationship_level: float = 0.0

    # Retrieval preferences
    max_results: int = 10
    include_triggering: bool = True
    therapeutic_preference: bool = False
    avoid_recent_triggers: bool = True

    # Temporal constraints
    max_age_hours: Optional[float] = None
    min_age_hours: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/debugging."""
        return {
            "target_mood": self.target_mood.value if self.target_mood else None,
            "emotional_types": [et.value for et in self.emotional_types],
            "significance_threshold": self.significance_threshold,
            "protection_level_max": self.protection_level_max.value,
            "trust_level": self.trust_level,
            "max_results": self.max_results,
            "include_triggering": self.include_triggering,
            "therapeutic_preference": self.therapeutic_preference,
        }


@dataclass
class EmotionalRetrievalResult:
    """Result of emotional memory retrieval with context and metadata."""

    # Retrieved records
    emotional_records: List[EmotionalMemoryRecord] = field(default_factory=list)

    # Query metadata
    query_processed: EmotionalRetrievalQuery
    total_available: int = 0
    filtered_count: int = 0
    retrieval_confidence: float = 0.0

    # Emotional context
    dominant_emotion: Optional[str] = None
    emotional_coherence: float = 0.0
    therapeutic_value: float = 0.0
    triggering_risk: float = 0.0

    # Performance
    retrieval_time_ms: float = 0.0
    cache_hit_rate: float = 0.0

    def get_exchanges(self) -> List[str]:
        """Get list of exchange IDs from results."""
        return [record.exchange_id for record in self.emotional_records]

    def get_by_significance(
        self, min_significance: float = 0.5
    ) -> List[EmotionalMemoryRecord]:
        """Get records above significance threshold."""
        return [
            record
            for record in self.emotional_records
            if record.emotional_significance.overall_significance >= min_significance
        ]

    def get_by_type(
        self, emotional_type: EmotionalMemoryType
    ) -> List[EmotionalMemoryRecord]:
        """Get records of specific emotional type."""
        return [
            record
            for record in self.emotional_records
            if record.emotional_significance.emotional_type == emotional_type
        ]


class EmotionalPreservationEngine:
    """Engine for preserving and retrieving emotional memory context."""

    def __init__(
        self,
        session_factory: DatabaseSessionFactory,
        cache_size: int = EmotionalThresholds.DEFAULT_CACHE_SIZE,
        preservation_batch_size: int = 50,
    ):
        self.session_factory = session_factory
        self.cache_size = cache_size
        self.preservation_batch_size = preservation_batch_size

        # Caches for performance
        self._emotional_records_cache: Dict[str, EmotionalMemoryRecord] = {}
        self._retrieval_cache: Dict[str, EmotionalRetrievalResult] = {}

        # Performance tracking
        self._performance_stats = {
            "preservations": 0,
            "retrievals": 0,
            "cache_hits": 0,
            "avg_preservation_time_ms": 0.0,
            "avg_retrieval_time_ms": 0.0,
        }

    async def preserve_emotional_context(
        self,
        exchange: ConversationExchange,
        emotional_significance: EmotionalSignificance,
        affective_weight: AffectiveWeight,
        mood_accessibility: Optional[Dict[MoodState, float]] = None,
    ) -> EmotionalMemoryRecord:
        """
        Preserve emotional context for a conversation exchange.

        This creates both the in-memory emotional record and the database
        emotional context entry for persistence.
        """
        try:
            # Validate inputs
            if not exchange or not exchange.exchange_id:
                raise InvalidEmotionalDataError(
                    "Exchange must have a valid exchange_id"
                )
            if not emotional_significance:
                raise InvalidEmotionalDataError("EmotionalSignificance is required")
            if not affective_weight:
                raise InvalidEmotionalDataError("AffectiveWeight is required")

            # Validate exchange content
            validate_uuid(exchange.exchange_id, "exchange.exchange_id")

            # Validate significance values
            if (
                emotional_significance.confidence_score < 0
                or emotional_significance.confidence_score > 1
            ):
                raise InvalidEmotionalDataError(
                    "Confidence score must be between 0 and 1"
                )

        except Exception as e:
            raise handle_emotional_memory_error(
                e, "Failed to validate inputs for emotional context preservation"
            )

        start_time = time.perf_counter()

        try:
            # Create emotional memory record
            emotional_record = EmotionalMemoryRecord(
                exchange_id=str(exchange.exchange_id),
                emotional_significance=emotional_significance,
                affective_weight=affective_weight,
                mood_accessibility=mood_accessibility or {},
                preservation_confidence=emotional_significance.confidence_score,
            )

            # Persist to database
            async with self.session_factory.get_session() as session:
                # Create database emotional context entry
                emotional_context_db = EmotionalContextModel(
                    exchange_id=exchange.exchange_id,
                    sentiment_score=emotional_significance.relationship_impact,
                    emotional_keywords=emotional_significance.contributing_factors,
                    participant_emotions={
                        "emotional_type": emotional_significance.emotional_type.value,
                        "protection_level": emotional_significance.protection_level.value,
                        "mood_accessibility": {
                            mood.value: weight
                            for mood, weight in (mood_accessibility or {}).items()
                        },
                    },
                    emotional_intensity=emotional_significance.intensity_score,
                    relationship_impact_score=abs(
                        emotional_significance.relationship_impact
                    ),
                )

                session.add(emotional_context_db)
                await session.commit()
                await session.refresh(emotional_context_db)

                emotional_record.emotional_context_id = str(
                    emotional_context_db.context_id
                )

            # Cache the record
            self._emotional_records_cache[emotional_record.exchange_id] = (
                emotional_record
            )

            # Manage cache size
            if len(self._emotional_records_cache) > self.cache_size:
                # Remove oldest entries
                oldest_keys = list(self._emotional_records_cache.keys())[
                    : -self.cache_size // 2
                ]
                for key in oldest_keys:
                    del self._emotional_records_cache[key]

            # Update performance stats
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            self._performance_stats["preservations"] += 1
            total_preservations = self._performance_stats["preservations"]
            self._performance_stats["avg_preservation_time_ms"] = (
                self._performance_stats["avg_preservation_time_ms"]
                * (total_preservations - 1)
                + processing_time_ms
            ) / total_preservations

            logger.debug(
                f"Preserved emotional context for exchange {exchange.exchange_id}"
            )
            return emotional_record

        except Exception as e:
            error = handle_emotional_memory_error(
                e, "Error preserving emotional context"
            )
            logger.error(f"Preservation failed: {error}")
            raise MemoryPreservationError(
                f"Failed to preserve emotional context for exchange {exchange.exchange_id}",
                exchange_id=str(exchange.exchange_id),
                operation="preserve",
                database_error=str(e),
            )

    async def retrieve_emotional_memories(
        self,
        npc_id: uuid.UUID,
        query: EmotionalRetrievalQuery,
    ) -> EmotionalRetrievalResult:
        """
        Retrieve emotionally significant memories based on query parameters.

        This performs both database queries and in-memory filtering to return
        memories that match emotional, trust, and contextual criteria.
        """
        try:
            # Validate inputs
            if not npc_id:
                raise InvalidEmotionalDataError("NPC ID is required")
            if not query:
                raise InvalidEmotionalDataError("Query is required")

            # Validate NPC ID
            validate_uuid(npc_id, "npc_id")

            # Validate query parameters
            validate_probability(query.significance_threshold, "significance_threshold")
            validate_probability(query.trust_level, "trust_level")
            validate_probability(query.relationship_level, "relationship_level")

            # Validate limits
            if (
                query.max_results <= 0
                or query.max_results > EmotionalThresholds.MAX_MEMORY_CAPACITY
            ):
                raise InvalidEmotionalDataError(
                    f"max_results must be between 1 and {EmotionalThresholds.MAX_MEMORY_CAPACITY}"
                )

            # Validate temporal constraints
            if query.max_age_hours is not None:
                validate_positive_number(
                    query.max_age_hours, "max_age_hours", allow_zero=True
                )
            if query.min_age_hours is not None:
                validate_positive_number(
                    query.min_age_hours, "min_age_hours", allow_zero=True
                )

        except Exception as e:
            raise handle_emotional_memory_error(
                e, "Failed to validate inputs for emotional memory retrieval"
            )

        start_time = time.perf_counter()

        # Create cache key for this query
        cache_key = self._create_retrieval_cache_key(npc_id, query)

        # Check cache first
        if cache_key in self._retrieval_cache:
            self._performance_stats["cache_hits"] += 1
            cached_result = self._retrieval_cache[cache_key]
            cached_result.cache_hit_rate = 1.0
            return cached_result

        try:
            async with self.session_factory.get_session() as session:
                # Build base query for exchanges with emotional context
                base_query = (
                    select(EmotionalContextModel)
                    .join(
                        ConversationExchange,
                        EmotionalContextModel.exchange_id
                        == ConversationExchange.exchange_id,
                    )
                    .where(
                        ConversationExchange.conversation_id.in_(
                            select(ConversationExchange.conversation_id).where(
                                ConversationExchange.speaker_id == npc_id
                            )
                        )
                    )
                )

                # Apply filters
                filters = []

                # Significance threshold (using emotional_intensity as proxy)
                if query.significance_threshold > 0:
                    filters.append(
                        EmotionalContextModel.emotional_intensity
                        >= query.significance_threshold
                    )

                # Protection level (would need to be stored in participant_emotions)
                # For now, using relationship_impact_score as proxy
                protection_threshold = self._get_protection_threshold(
                    query.protection_level_max
                )
                if protection_threshold > 0:
                    filters.append(
                        EmotionalContextModel.relationship_impact_score
                        <= protection_threshold
                    )

                # Apply temporal constraints
                if query.max_age_hours:
                    cutoff_time = datetime.now(timezone.utc) - timedelta(
                        hours=query.max_age_hours
                    )
                    filters.append(EmotionalContextModel.created_at >= cutoff_time)

                if query.min_age_hours:
                    cutoff_time = datetime.now(timezone.utc) - timedelta(
                        hours=query.min_age_hours
                    )
                    filters.append(EmotionalContextModel.created_at <= cutoff_time)

                if filters:
                    base_query = base_query.where(and_(*filters))

                # Apply ordering and limits
                base_query = base_query.order_by(
                    desc(EmotionalContextModel.emotional_intensity),
                    desc(EmotionalContextModel.relationship_impact_score),
                ).limit(
                    query.max_results * 2
                )  # Get extra for filtering

                # Execute query
                result = await session.execute(base_query)
                emotional_contexts = result.scalars().all()

                # Convert to emotional records and apply advanced filtering
                emotional_records = await self._build_emotional_records_from_contexts(
                    emotional_contexts, session, query
                )

                # Apply mood-based filtering if specified
                if query.target_mood:
                    emotional_records = self._filter_by_mood_accessibility(
                        emotional_records, query.target_mood, query.trust_level
                    )

                # Apply emotional type filtering
                if query.emotional_types:
                    emotional_records = [
                        record
                        for record in emotional_records
                        if record.emotional_significance.emotional_type
                        in query.emotional_types
                    ]

                # Apply therapeutic filtering
                if query.therapeutic_preference and query.target_mood:
                    emotional_records = self._prioritize_therapeutic_memories(
                        emotional_records, query.target_mood
                    )

                # Limit final results
                emotional_records = emotional_records[: query.max_results]

                # Calculate result metadata
                result_metadata = self._calculate_result_metadata(
                    emotional_records, query
                )

                # Create retrieval result
                retrieval_result = EmotionalRetrievalResult(
                    emotional_records=emotional_records,
                    query_processed=query,
                    total_available=len(emotional_contexts),
                    filtered_count=len(emotional_records),
                    retrieval_confidence=result_metadata["confidence"],
                    dominant_emotion=result_metadata["dominant_emotion"],
                    emotional_coherence=result_metadata["coherence"],
                    therapeutic_value=result_metadata["therapeutic_value"],
                    triggering_risk=result_metadata["triggering_risk"],
                    retrieval_time_ms=(time.perf_counter() - start_time) * 1000,
                    cache_hit_rate=0.0,
                )

                # Update retrieval statistics
                await self._update_retrieval_stats(emotional_records, session)

                # Cache result
                self._retrieval_cache[cache_key] = retrieval_result

                # Manage cache size
                if len(self._retrieval_cache) > self.cache_size // 2:
                    oldest_keys = list(self._retrieval_cache.keys())[
                        : -self.cache_size // 4
                    ]
                    for key in oldest_keys:
                        del self._retrieval_cache[key]

                # Update performance stats
                processing_time_ms = retrieval_result.retrieval_time_ms
                self._performance_stats["retrievals"] += 1
                total_retrievals = self._performance_stats["retrievals"]
                self._performance_stats["avg_retrieval_time_ms"] = (
                    self._performance_stats["avg_retrieval_time_ms"]
                    * (total_retrievals - 1)
                    + processing_time_ms
                ) / total_retrievals

                logger.debug(
                    f"Retrieved {len(emotional_records)} emotional memories for NPC {npc_id}"
                )
                return retrieval_result

        except Exception as e:
            error = handle_emotional_memory_error(
                e, "Error retrieving emotional memories"
            )
            logger.error(f"Retrieval failed: {error}")
            raise MemoryPreservationError(
                f"Failed to retrieve emotional memories for NPC {npc_id}",
                operation="retrieve",
                database_error=str(e),
            )

    async def _build_emotional_records_from_contexts(
        self,
        emotional_contexts: List[EmotionalContextModel],
        session: Session,
        query: EmotionalRetrievalQuery,
    ) -> List[EmotionalMemoryRecord]:
        """Build emotional memory records from database emotional contexts."""

        records = []

        for context in emotional_contexts:
            # Check if we have cached record
            if str(context.exchange_id) in self._emotional_records_cache:
                cached_record = self._emotional_records_cache[str(context.exchange_id)]
                # Update retrieval metadata
                cached_record.retrieval_frequency += 1
                cached_record.last_retrieved = time.time()
                if query.target_mood:
                    cached_record.retrieval_contexts.append(query.target_mood.value)
                records.append(cached_record)
                continue

            # Reconstruct emotional significance from database data
            participant_emotions = context.participant_emotions or {}
            emotional_type_str = participant_emotions.get(
                "emotional_type", "everyday_positive"
            )
            protection_level_str = participant_emotions.get(
                "protection_level", "public"
            )

            try:
                emotional_type = EmotionalMemoryType(emotional_type_str)
            except ValueError:
                emotional_type = EmotionalMemoryType.EVERYDAY_POSITIVE

            try:
                protection_level = MemoryProtectionLevel(protection_level_str)
            except ValueError:
                protection_level = MemoryProtectionLevel.PUBLIC

            # Create simplified emotional significance (some data may be lost)
            emotional_significance = EmotionalSignificance(
                overall_significance=float(context.emotional_intensity),
                emotional_type=emotional_type,
                intensity_score=float(context.emotional_intensity),
                personal_relevance=float(context.relationship_impact_score),
                relationship_impact=float(context.sentiment_score),
                formative_influence=float(context.emotional_intensity)
                * 0.8,  # Estimate
                protection_level=protection_level,
                mood_accessibility={},  # Would need to be reconstructed
                decay_resistance=float(context.emotional_intensity) * 0.7,  # Estimate
                triggering_potential=float(context.relationship_impact_score)
                * 0.6,  # Estimate
                confidence_score=0.8,  # Default for reconstructed data
                contributing_factors=context.emotional_keywords or [],
            )

            # Create simplified affective weight (would ideally be stored)
            affective_weight = AffectiveWeight(
                base_affective_weight=float(context.emotional_intensity),
                intensity_multiplier=1.0 + float(context.emotional_intensity) * 0.5,
                personality_modifier=1.0,
                mood_accessibility_modifier=1.0,
                recency_boost=1.0,
                relationship_amplifier=1.0
                + float(context.relationship_impact_score) * 0.3,
                formative_importance=float(context.emotional_intensity) * 0.8,
                access_threshold=self._get_access_threshold_for_protection(
                    protection_level
                ),
                trauma_sensitivity=(
                    0.8 if emotional_type == EmotionalMemoryType.TRAUMATIC else 0.0
                ),
                final_weight=float(context.emotional_intensity),
                weighting_strategy=AffectiveWeightingStrategy.LINEAR,
                confidence=0.7,  # Lower for reconstructed data
            )

            # Create emotional record
            record = EmotionalMemoryRecord(
                exchange_id=str(context.exchange_id),
                emotional_context_id=str(context.context_id),
                emotional_significance=emotional_significance,
                affective_weight=affective_weight,
                mood_accessibility=participant_emotions.get("mood_accessibility", {}),
                preserved_at=context.created_at.timestamp(),
                preservation_confidence=0.7,  # Lower for reconstructed
                retrieval_frequency=1,
                last_retrieved=time.time(),
                retrieval_contexts=(
                    [query.target_mood.value] if query.target_mood else []
                ),
            )

            records.append(record)

            # Cache the record
            self._emotional_records_cache[record.exchange_id] = record

        return records

    def _filter_by_mood_accessibility(
        self,
        records: List[EmotionalMemoryRecord],
        target_mood: MoodState,
        trust_level: float,
    ) -> List[EmotionalMemoryRecord]:
        """Filter records based on mood accessibility."""

        accessible_records = []

        for record in records:
            # Check mood accessibility
            mood_accessibility = record.mood_accessibility.get(target_mood, 0.5)

            # Check trust requirements
            required_trust = record.affective_weight.access_threshold

            # Apply accessibility and trust filters
            if mood_accessibility >= 0.3 and trust_level >= required_trust:
                accessible_records.append(record)

        # Sort by accessibility and significance
        accessible_records.sort(
            key=lambda r: (
                r.mood_accessibility.get(target_mood, 0.5),
                r.emotional_significance.overall_significance,
            ),
            reverse=True,
        )

        return accessible_records

    def _prioritize_therapeutic_memories(
        self,
        records: List[EmotionalMemoryRecord],
        target_mood: MoodState,
    ) -> List[EmotionalMemoryRecord]:
        """Prioritize memories with therapeutic value for the target mood."""

        # Calculate therapeutic value for each record
        therapeutic_scores = []

        for record in records:
            therapeutic_value = self._calculate_therapeutic_value_for_mood(
                record.emotional_significance, target_mood
            )
            therapeutic_scores.append((record, therapeutic_value))

        # Sort by therapeutic value, then significance
        therapeutic_scores.sort(
            key=lambda x: (x[1], x[0].emotional_significance.overall_significance),
            reverse=True,
        )

        return [record for record, _ in therapeutic_scores]

    def _calculate_therapeutic_value_for_mood(
        self,
        emotional_significance: EmotionalSignificance,
        target_mood: MoodState,
    ) -> float:
        """Calculate therapeutic value of a memory for a specific mood."""

        memory_type = emotional_significance.emotional_type

        # Therapeutic value patterns
        therapeutic_patterns = {
            MoodState.MELANCHOLY: {
                EmotionalMemoryType.PEAK_POSITIVE: 0.9,
                EmotionalMemoryType.CORE_ATTACHMENT: 0.8,
                EmotionalMemoryType.BREAKTHROUGH: 0.7,
                EmotionalMemoryType.FORMATIVE: 0.6,
                EmotionalMemoryType.TRAUMATIC: 0.1,
            },
            MoodState.ANXIOUS: {
                EmotionalMemoryType.CORE_ATTACHMENT: 0.9,
                EmotionalMemoryType.BREAKTHROUGH: 0.8,
                EmotionalMemoryType.PEAK_POSITIVE: 0.7,
                EmotionalMemoryType.TRAUMATIC: 0.0,
                EmotionalMemoryType.CONFLICT: 0.2,
            },
            MoodState.ANGRY: {
                EmotionalMemoryType.PEAK_POSITIVE: 0.8,
                EmotionalMemoryType.CORE_ATTACHMENT: 0.7,
                EmotionalMemoryType.FORMATIVE: 0.5,
                EmotionalMemoryType.CONFLICT: 0.1,
            },
            MoodState.FEARFUL: {
                EmotionalMemoryType.CORE_ATTACHMENT: 0.9,
                EmotionalMemoryType.PEAK_POSITIVE: 0.8,
                EmotionalMemoryType.BREAKTHROUGH: 0.6,
                EmotionalMemoryType.TRAUMATIC: 0.0,
            },
        }

        pattern = therapeutic_patterns.get(target_mood, {})
        base_value = pattern.get(memory_type, 0.4)

        # Adjust for emotional intensity (moderate intensity is more therapeutic)
        intensity = emotional_significance.intensity_score
        if 0.4 <= intensity <= 0.7:
            intensity_modifier = 1.2
        elif intensity > 0.8:
            intensity_modifier = 0.8  # Very intense memories may not be therapeutic
        else:
            intensity_modifier = 1.0

        return base_value * intensity_modifier

    def _calculate_result_metadata(
        self,
        records: List[EmotionalMemoryRecord],
        query: EmotionalRetrievalQuery,
    ) -> Dict[str, Any]:
        """Calculate metadata for retrieval results."""

        if not records:
            return {
                "confidence": 0.0,
                "dominant_emotion": None,
                "coherence": 0.0,
                "therapeutic_value": 0.0,
                "triggering_risk": 0.0,
            }

        # Calculate average confidence
        avg_confidence = sum(
            r.emotional_significance.confidence_score for r in records
        ) / len(records)

        # Find dominant emotional type
        type_counts = {}
        for record in records:
            emotional_type = record.emotional_significance.emotional_type
            type_counts[emotional_type] = type_counts.get(emotional_type, 0) + 1

        dominant_emotion = (
            max(type_counts, key=type_counts.get).value if type_counts else None
        )

        # Calculate emotional coherence (how similar the emotions are)
        if len(type_counts) == 1:
            coherence = 1.0
        else:
            max_count = max(type_counts.values())
            coherence = max_count / len(records)

        # Calculate therapeutic value
        if query.target_mood:
            therapeutic_values = [
                self._calculate_therapeutic_value_for_mood(
                    r.emotional_significance, query.target_mood
                )
                for r in records
            ]
            avg_therapeutic_value = sum(therapeutic_values) / len(therapeutic_values)
        else:
            avg_therapeutic_value = 0.5

        # Calculate triggering risk
        avg_triggering_risk = sum(
            r.emotional_significance.triggering_potential for r in records
        ) / len(records)

        return {
            "confidence": avg_confidence,
            "dominant_emotion": dominant_emotion,
            "coherence": coherence,
            "therapeutic_value": avg_therapeutic_value,
            "triggering_risk": avg_triggering_risk,
        }

    async def _update_retrieval_stats(
        self,
        records: List[EmotionalMemoryRecord],
        session: Session,
    ) -> None:
        """Update retrieval statistics in database."""

        try:
            for record in records:
                # Update access count in conversation_exchanges table
                # This would require a separate query to update the exchange
                # For now, we'll just log the access
                logger.debug(f"Memory {record.exchange_id} accessed")

        except Exception as e:
            logger.warning(f"Error updating retrieval stats: {e}")

    def _get_protection_threshold(
        self, protection_level: MemoryProtectionLevel
    ) -> float:
        """Get relationship impact threshold for protection level."""
        thresholds = {
            MemoryProtectionLevel.PUBLIC: 1.0,
            MemoryProtectionLevel.PRIVATE: 0.8,
            MemoryProtectionLevel.SENSITIVE: 0.6,
            MemoryProtectionLevel.PROTECTED: 0.4,
            MemoryProtectionLevel.TRAUMATIC: 0.2,
        }
        return thresholds.get(protection_level, 1.0)

    def _get_access_threshold_for_protection(
        self, protection_level: MemoryProtectionLevel
    ) -> float:
        """Get trust threshold for protection level."""
        thresholds = {
            MemoryProtectionLevel.PUBLIC: 0.0,
            MemoryProtectionLevel.PRIVATE: 0.3,
            MemoryProtectionLevel.SENSITIVE: 0.6,
            MemoryProtectionLevel.PROTECTED: 0.8,
            MemoryProtectionLevel.TRAUMATIC: 0.9,
        }
        return thresholds.get(protection_level, 0.3)

    def _create_retrieval_cache_key(
        self,
        npc_id: uuid.UUID,
        query: EmotionalRetrievalQuery,
    ) -> str:
        """Create cache key for retrieval query."""
        key_components = [
            str(npc_id),
            query.target_mood.value if query.target_mood else "none",
            "_".join(sorted([et.value for et in query.emotional_types])),
            str(query.significance_threshold),
            query.protection_level_max.value,
            str(query.max_results),
        ]
        return "_".join(key_components)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        total_operations = (
            self._performance_stats["preservations"]
            + self._performance_stats["retrievals"]
            + self._performance_stats["cache_hits"]
        )
        cache_hit_rate = (
            (self._performance_stats["cache_hits"] / total_operations * 100)
            if total_operations > 0
            else 0.0
        )

        return {
            "total_preservations": self._performance_stats["preservations"],
            "total_retrievals": self._performance_stats["retrievals"],
            "cache_hits": self._performance_stats["cache_hits"],
            "cache_hit_rate_percent": round(cache_hit_rate, 1),
            "avg_preservation_time_ms": round(
                self._performance_stats["avg_preservation_time_ms"], 2
            ),
            "avg_retrieval_time_ms": round(
                self._performance_stats["avg_retrieval_time_ms"], 2
            ),
            "cached_records": len(self._emotional_records_cache),
            "cached_retrievals": len(self._retrieval_cache),
        }

    def clear_caches(self) -> None:
        """Clear all caches."""
        self._emotional_records_cache.clear()
        self._retrieval_cache.clear()
        self._performance_stats = {
            "preservations": 0,
            "retrievals": 0,
            "cache_hits": 0,
            "avg_preservation_time_ms": 0.0,
            "avg_retrieval_time_ms": 0.0,
        }
