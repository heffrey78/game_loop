"""Main memory algorithm service orchestrating all memory operations."""

import time
import uuid
from dataclasses import dataclass
from typing import Any

from .clustering import ClusterResult, MemoryClusterData, MemoryClusteringEngine
from .confidence import MemoryConfidenceCalculator
from .config import PERSONALITY_CONFIGS, MemoryAlgorithmConfig
from .emotional_analysis import EmotionalAnalysisResult, EmotionalWeightingAnalyzer


@dataclass
class MemoryProcessingResult:
    """Result of memory processing operation."""

    memories_processed: int
    confidence_updates: dict[uuid.UUID, float]
    emotional_analysis: dict[uuid.UUID, EmotionalAnalysisResult]
    clustering_result: ClusterResult | None
    processing_time_ms: float
    performance_metrics: dict[str, Any]


class MemoryAlgorithmService:
    """
    Main service orchestrating memory confidence calculation, emotional analysis,
    and clustering operations for NPC semantic memory.

    This service coordinates between:
    - Memory confidence calculations with personality modifiers
    - Emotional weighting analysis for memory significance
    - K-means clustering of related memories
    - Performance monitoring and optimization
    """

    def __init__(
        self,
        config: MemoryAlgorithmConfig | None = None,
        personality_type: str | None = None,
    ):
        # Use personality-specific config if provided, otherwise default
        if personality_type and personality_type in PERSONALITY_CONFIGS:
            self.config = PERSONALITY_CONFIGS[personality_type]
            # Override with any provided config values
            if config:
                for field_name in config.__dataclass_fields__:
                    if hasattr(config, field_name):
                        setattr(self.config, field_name, getattr(config, field_name))
        else:
            self.config = config or MemoryAlgorithmConfig()

        # Validate configuration
        self.config.validate()

        # Initialize algorithm components
        self.confidence_calculator = MemoryConfidenceCalculator(self.config)
        self.emotional_analyzer = EmotionalWeightingAnalyzer(self.config)
        self.clustering_engine = MemoryClusteringEngine(self.config)

        # Performance tracking
        self._service_stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "avg_operation_time_ms": 0.0,
        }

    async def process_memory_batch(
        self,
        memory_data: list[dict[str, Any]],
        npc_personality_config: dict[str, float] | None = None,
        include_clustering: bool = True,
    ) -> MemoryProcessingResult:
        """
        Process a batch of memories with confidence calculation, emotional analysis,
        and optional clustering.

        Args:
            memory_data: List of memory dictionaries with required fields
            npc_personality_config: NPC-specific personality modifiers
            include_clustering: Whether to perform clustering analysis

        Returns:
            MemoryProcessingResult with all processing outcomes
        """
        start_time = time.perf_counter()
        self._service_stats["total_operations"] += 1

        try:
            # Extract personality modifiers
            personality_modifiers = self._extract_personality_modifiers(
                npc_personality_config
            )

            # Process confidence calculations
            confidence_updates = await self._process_confidence_batch(
                memory_data, personality_modifiers
            )

            # Process emotional analysis
            emotional_analysis = await self._process_emotional_analysis_batch(
                memory_data
            )

            # Perform clustering if requested
            clustering_result = None
            if include_clustering and len(memory_data) >= 2:
                clustering_result = await self._perform_memory_clustering(
                    memory_data, emotional_analysis, confidence_updates
                )

            # Calculate performance metrics
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            performance_metrics = self._calculate_performance_metrics()

            # Update service statistics
            self._update_service_stats(processing_time_ms, success=True)

            return MemoryProcessingResult(
                memories_processed=len(memory_data),
                confidence_updates=confidence_updates,
                emotional_analysis=emotional_analysis,
                clustering_result=clustering_result,
                processing_time_ms=processing_time_ms,
                performance_metrics=performance_metrics,
            )

        except Exception as e:
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            self._update_service_stats(processing_time_ms, success=False)

            print(f"Memory processing failed: {e}")
            # Return empty result rather than raising
            return MemoryProcessingResult(
                memories_processed=0,
                confidence_updates={},
                emotional_analysis={},
                clustering_result=None,
                processing_time_ms=processing_time_ms,
                performance_metrics=self._calculate_performance_metrics(),
            )

    async def calculate_single_memory_confidence(
        self,
        memory_id: uuid.UUID,
        base_confidence: float,
        memory_age_days: float,
        emotional_weight: float,
        access_count: int = 0,
        npc_personality_config: dict[str, float] | None = None,
    ) -> float:
        """
        Calculate confidence for a single memory with personality modifiers.

        Args:
            memory_id: Unique memory identifier
            base_confidence: Initial confidence score
            memory_age_days: Age of memory in days
            emotional_weight: Emotional significance
            access_count: Number of times accessed
            npc_personality_config: NPC personality parameters

        Returns:
            Current confidence score
        """
        personality_modifiers = self._extract_personality_modifiers(
            npc_personality_config
        )

        return self.confidence_calculator.calculate_confidence(
            base_confidence=base_confidence,
            memory_age_days=memory_age_days,
            emotional_weight=emotional_weight,
            access_count=access_count,
            personality_decay_modifier=personality_modifiers["decay_modifier"],
            personality_emotional_sensitivity=personality_modifiers[
                "emotional_sensitivity"
            ],
        )

    async def analyze_memory_emotional_weight(
        self,
        memory_content: str,
        conversation_context: dict | None = None,
        participant_info: dict | None = None,
    ) -> EmotionalAnalysisResult:
        """
        Analyze emotional significance of memory content.

        Args:
            memory_content: The conversation text to analyze
            conversation_context: Context about the conversation
            participant_info: Information about participants

        Returns:
            EmotionalAnalysisResult with emotional metrics
        """
        return self.emotional_analyzer.analyze_emotional_weight(
            memory_content, conversation_context, participant_info
        )

    async def cluster_npc_memories(
        self,
        npc_id: uuid.UUID,
        memory_data: list[dict[str, Any]],
        target_clusters: int | None = None,
    ) -> ClusterResult | None:
        """
        Cluster memories for a specific NPC.

        Args:
            npc_id: NPC identifier
            memory_data: Memory data including embeddings
            target_clusters: Desired number of clusters

        Returns:
            ClusterResult if clustering successful, None otherwise
        """
        if len(memory_data) < 2:
            return None

        # Prepare clustering data
        cluster_data = []
        for memory in memory_data:
            if "embedding" not in memory or memory["embedding"] is None:
                continue

            cluster_data.append(
                MemoryClusterData(
                    memory_id=memory["memory_id"],
                    embedding=memory["embedding"],
                    emotional_weight=memory.get("emotional_weight", 0.0),
                    confidence_score=memory.get("confidence_score", 1.0),
                    age_days=memory.get("age_days", 0.0),
                    content_preview=memory.get("content", "")[:50],
                )
            )

        if len(cluster_data) < 2:
            return None

        return self.clustering_engine.cluster_memories(
            cluster_data, target_clusters, include_low_confidence=False
        )

    def should_memory_express_uncertainty(
        self,
        confidence: float,
        npc_personality_config: dict[str, float] | None = None,
    ) -> bool:
        """
        Determine if NPC should express uncertainty about this memory.

        Args:
            confidence: Memory confidence score
            npc_personality_config: NPC personality parameters

        Returns:
            True if uncertainty should be expressed
        """
        # Adjust uncertainty threshold based on personality
        threshold = self.config.uncertainty_threshold
        if npc_personality_config and "uncertainty_threshold" in npc_personality_config:
            threshold = npc_personality_config["uncertainty_threshold"]

        return confidence < threshold

    def get_memory_uncertainty_expression(
        self, confidence: float, npc_personality_type: str | None = None
    ) -> str:
        """
        Get appropriate uncertainty expression for confidence level.

        Args:
            confidence: Memory confidence score
            npc_personality_type: Personality type for custom expressions

        Returns:
            Uncertainty phrase for dialogue
        """
        base_expression = self.confidence_calculator.get_uncertainty_expression(
            confidence
        )

        # Customize based on personality type
        if npc_personality_type == "analytical":
            return base_expression.replace("I think", "My analysis suggests")
        elif npc_personality_type == "forgetful":
            return base_expression.replace("I clearly remember", "I think I remember")
        elif npc_personality_type == "emotional":
            return base_expression.replace(
                "That rings a bell", "Something stirs in my memory"
            )

        return base_expression

    async def _process_confidence_batch(
        self,
        memory_data: list[dict[str, Any]],
        personality_modifiers: dict[str, float],
    ) -> dict[uuid.UUID, float]:
        """Process confidence calculations for memory batch."""

        confidence_updates = {}

        # Prepare batch data
        batch_data = []
        memory_ids = []

        for memory in memory_data:
            memory_ids.append(memory["memory_id"])
            batch_data.append(
                (
                    memory.get("base_confidence", 1.0),
                    memory.get("age_days", 0.0),
                    memory.get("emotional_weight", 0.0),
                    memory.get("access_count", 0),
                    personality_modifiers["decay_modifier"],
                    personality_modifiers["emotional_sensitivity"],
                )
            )

        # Calculate confidence scores in batch
        confidence_scores = self.confidence_calculator.batch_calculate_confidence(
            batch_data
        )

        # Create result mapping
        for memory_id, confidence in zip(memory_ids, confidence_scores, strict=False):
            confidence_updates[memory_id] = confidence

        return confidence_updates

    async def _process_emotional_analysis_batch(
        self, memory_data: list[dict[str, Any]]
    ) -> dict[uuid.UUID, EmotionalAnalysisResult]:
        """Process emotional analysis for memory batch."""

        emotional_analysis = {}

        for memory in memory_data:
            memory_id = memory["memory_id"]
            content = memory.get("content", "")
            context = memory.get("conversation_context")
            participants = memory.get("participant_info")

            analysis = await self.analyze_memory_emotional_weight(
                content, context, participants
            )
            emotional_analysis[memory_id] = analysis

        return emotional_analysis

    async def _perform_memory_clustering(
        self,
        memory_data: list[dict[str, Any]],
        emotional_analysis: dict[uuid.UUID, EmotionalAnalysisResult],
        confidence_updates: dict[uuid.UUID, float],
    ) -> ClusterResult | None:
        """Perform clustering on memory data."""

        # Prepare clustering data with updated emotional weights and confidence
        cluster_data = []
        for memory in memory_data:
            memory_id = memory["memory_id"]

            if "embedding" not in memory or memory["embedding"] is None:
                continue

            # Use updated emotional weight and confidence
            emotional_weight = emotional_analysis.get(memory_id)
            if emotional_weight:
                emotional_weight = emotional_weight.emotional_weight
            else:
                emotional_weight = memory.get("emotional_weight", 0.0)

            confidence = confidence_updates.get(
                memory_id, memory.get("confidence_score", 1.0)
            )

            cluster_data.append(
                MemoryClusterData(
                    memory_id=memory_id,
                    embedding=memory["embedding"],
                    emotional_weight=emotional_weight,
                    confidence_score=confidence,
                    age_days=memory.get("age_days", 0.0),
                    content_preview=memory.get("content", "")[:50],
                )
            )

        if len(cluster_data) < 2:
            return None

        return self.clustering_engine.cluster_memories(cluster_data)

    def _extract_personality_modifiers(
        self, npc_personality_config: dict[str, float] | None
    ) -> dict[str, float]:
        """Extract personality modifiers with defaults."""

        return {
            "decay_modifier": (
                npc_personality_config.get("decay_rate_modifier", 1.0)
                if npc_personality_config
                else 1.0
            ),
            "emotional_sensitivity": (
                npc_personality_config.get("emotional_sensitivity", 1.0)
                if npc_personality_config
                else 1.0
            ),
        }

    def _calculate_performance_metrics(self) -> dict[str, Any]:
        """Calculate overall performance metrics."""

        return {
            "confidence_calculator": self.confidence_calculator.get_performance_stats(),
            "emotional_analyzer": self.emotional_analyzer.get_performance_stats(),
            "clustering_engine": self.clustering_engine.get_clustering_stats(),
            "service_stats": self._service_stats.copy(),
        }

    def _update_service_stats(self, processing_time_ms: float, success: bool) -> None:
        """Update service-level statistics."""

        if success:
            self._service_stats["successful_operations"] += 1

        # Update average processing time
        total = self._service_stats["total_operations"]
        current_avg = self._service_stats["avg_operation_time_ms"]
        new_avg = ((current_avg * (total - 1)) + processing_time_ms) / total
        self._service_stats["avg_operation_time_ms"] = new_avg

    def get_service_performance_stats(self) -> dict[str, Any]:
        """Get comprehensive service performance statistics."""

        success_rate = 0.0
        if self._service_stats["total_operations"] > 0:
            success_rate = (
                self._service_stats["successful_operations"]
                / self._service_stats["total_operations"]
                * 100
            )

        return {
            "service_stats": {
                **self._service_stats,
                "success_rate_percent": round(success_rate, 1),
            },
            "component_stats": self._calculate_performance_metrics(),
        }

    def reset_performance_stats(self) -> None:
        """Reset all performance statistics."""

        self._service_stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "avg_operation_time_ms": 0.0,
        }

        self.confidence_calculator.reset_performance_stats()
        self.emotional_analyzer.clear_cache()
        self.clustering_engine.reset_stats()
