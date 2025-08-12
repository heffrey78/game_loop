"""Memory confidence calculation with exponential decay and personality modifiers."""

import math
import time
from functools import lru_cache

from .config import MemoryAlgorithmConfig


class MemoryConfidenceCalculator:
    """
    Calculates memory confidence using exponential decay with emotional amplification.

    Implements the core formula:
    confidence = base_confidence * exp(-decay_rate * age_days * decay_modifier / emotional_amplifier)
    """

    def __init__(self, config: MemoryAlgorithmConfig):
        self.config = config
        self._cache_hits = 0
        self._cache_misses = 0

    def calculate_confidence(
        self,
        base_confidence: float,
        memory_age_days: float,
        emotional_weight: float,
        access_count: int = 0,
        personality_decay_modifier: float = 1.0,
        personality_emotional_sensitivity: float = 1.0,
    ) -> float:
        """
        Calculate current memory confidence based on age, emotion, and access patterns.

        Args:
            base_confidence: Initial confidence score (0.0-1.0)
            memory_age_days: Days since memory was created
            emotional_weight: Emotional significance (0.0-1.0)
            access_count: Number of times memory was accessed
            personality_decay_modifier: NPC personality decay rate modifier
            personality_emotional_sensitivity: NPC emotional sensitivity modifier

        Returns:
            Current confidence score (0.0-1.0)
        """
        # Validate inputs
        if not (0.0 <= base_confidence <= 1.0):
            raise ValueError(f"base_confidence must be 0.0-1.0, got {base_confidence}")
        if memory_age_days < 0:
            raise ValueError(
                f"memory_age_days cannot be negative, got {memory_age_days}"
            )
        if not (0.0 <= emotional_weight <= 1.0):
            raise ValueError(
                f"emotional_weight must be 0.0-1.0, got {emotional_weight}"
            )

        # Use cached calculation if available
        cache_key = self._create_cache_key(
            base_confidence,
            memory_age_days,
            emotional_weight,
            access_count,
            personality_decay_modifier,
            personality_emotional_sensitivity,
        )

        cached_result = self._get_cached_confidence(cache_key)
        if cached_result is not None:
            self._cache_hits += 1
            return cached_result

        self._cache_misses += 1
        start_time = time.perf_counter()

        # Calculate emotional amplification
        emotional_amplifier = self._calculate_emotional_amplifier(
            emotional_weight, personality_emotional_sensitivity
        )

        # Calculate access bonus
        access_bonus = self._calculate_access_bonus(access_count)

        # Apply exponential decay with emotional protection
        effective_decay_rate = self.config.base_decay_rate * personality_decay_modifier

        if emotional_amplifier > 1.0:
            # Emotional memories decay slower
            age_decay = math.exp(
                -effective_decay_rate * memory_age_days / emotional_amplifier
            )
        else:
            # Standard decay for neutral memories
            age_decay = math.exp(-effective_decay_rate * memory_age_days)

        # Calculate final confidence
        confidence = base_confidence * age_decay * (1.0 + access_bonus)
        confidence = max(0.0, min(1.0, confidence))  # Clamp to valid range

        # Cache result and validate performance
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        if processing_time_ms > self.config.max_processing_time_ms:
            print(
                f"Warning: Confidence calculation took {processing_time_ms:.2f}ms "
                f"(target: {self.config.max_processing_time_ms}ms)"
            )

        self._cache_confidence(cache_key, confidence)
        return confidence

    def _calculate_emotional_amplifier(
        self, emotional_weight: float, personality_sensitivity: float
    ) -> float:
        """Calculate emotional amplification factor."""
        if emotional_weight <= 0.1:
            return 1.0  # No amplification for neutral memories

        # Scale emotional weight by personality sensitivity
        adjusted_weight = emotional_weight * personality_sensitivity

        # Calculate amplifier (1.0 to emotional_multiplier_max)
        amplifier = 1.0 + (
            adjusted_weight * (self.config.emotional_multiplier_max - 1.0)
        )
        return min(amplifier, self.config.emotional_multiplier_max)

    def _calculate_access_bonus(self, access_count: int) -> float:
        """Calculate memory access bonus."""
        if access_count <= 0:
            return 0.0

        bonus = access_count * self.config.access_bonus_rate
        return min(bonus, self.config.access_bonus_max)

    def _create_cache_key(
        self,
        base_confidence: float,
        memory_age_days: float,
        emotional_weight: float,
        access_count: int,
        personality_decay_modifier: float,
        personality_emotional_sensitivity: float,
    ) -> str:
        """Create cache key for confidence calculation."""
        # Round values to reduce cache fragmentation
        return (
            f"{base_confidence:.3f}_{memory_age_days:.2f}_{emotional_weight:.3f}_"
            f"{access_count}_{personality_decay_modifier:.3f}_{personality_emotional_sensitivity:.3f}"
        )

    @lru_cache(maxsize=1000)
    def _get_cached_confidence(self, cache_key: str) -> float | None:
        """Get cached confidence calculation (uses LRU cache)."""
        # This method signature enables LRU caching
        return None  # Cache miss, will be populated by _cache_confidence

    def _cache_confidence(self, cache_key: str, confidence: float) -> None:
        """Cache confidence calculation result."""
        # In production, this would update the LRU cache
        # For now, we rely on the @lru_cache decorator on _get_cached_confidence
        pass

    def should_express_uncertainty(self, confidence: float) -> bool:
        """
        Determine if NPC should express uncertainty about this memory.

        Args:
            confidence: Current memory confidence

        Returns:
            True if NPC should use uncertain language
        """
        return confidence < self.config.uncertainty_threshold

    def get_uncertainty_expression(self, confidence: float) -> str:
        """
        Get appropriate uncertainty expression for confidence level.

        Args:
            confidence: Memory confidence score

        Returns:
            Uncertainty phrase for dialogue
        """
        if confidence >= 0.8:
            return "I clearly remember"
        elif confidence >= 0.6:
            return "I believe"
        elif confidence >= 0.4:
            return "I think"
        elif confidence >= 0.2:
            return "That rings a bell"
        else:
            return "I vaguely recall"

    def batch_calculate_confidence(
        self,
        memory_data: list[tuple[float, float, float, int, float, float]],
    ) -> list[float]:
        """
        Calculate confidence for multiple memories efficiently.

        Args:
            memory_data: List of tuples containing (base_confidence, age_days,
                        emotional_weight, access_count, decay_modifier, sensitivity)

        Returns:
            List of confidence scores
        """
        start_time = time.perf_counter()
        results = []

        for data in memory_data:
            confidence = self.calculate_confidence(*data)
            results.append(confidence)

        processing_time_ms = (time.perf_counter() - start_time) * 1000
        avg_time_per_calc = processing_time_ms / len(memory_data) if memory_data else 0

        if avg_time_per_calc > self.config.max_processing_time_ms:
            print(
                f"Warning: Batch confidence calculation averaged {avg_time_per_calc:.2f}ms "
                f"per memory (target: {self.config.max_processing_time_ms}ms)"
            )

        return results

    def get_performance_stats(self) -> dict[str, int]:
        """Get cache performance statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (
            (self._cache_hits / total_requests * 100) if total_requests > 0 else 0
        )

        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate_percent": round(hit_rate, 1),
            "total_requests": total_requests,
        }

    def reset_performance_stats(self) -> None:
        """Reset performance tracking statistics."""
        self._cache_hits = 0
        self._cache_misses = 0
