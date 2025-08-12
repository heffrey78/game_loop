"""Configuration for memory algorithms."""

from dataclasses import dataclass
from typing import Any


@dataclass
class MemoryAlgorithmConfig:
    """Configuration parameters for memory algorithms."""

    # Memory confidence calculation parameters
    base_decay_rate: float = 0.1  # 10% decay per day baseline
    emotional_multiplier_max: float = 3.0  # 3x retention for high emotion
    access_bonus_rate: float = 0.01  # 1% bonus per access
    access_bonus_max: float = 0.2  # Maximum 20% access bonus
    uncertainty_threshold: float = 0.3  # Below 30% = express uncertainty

    # K-means clustering parameters
    max_clusters: int = 10  # Maximum clusters per NPC
    min_cluster_size: int = 3  # Minimum memories per cluster
    similarity_threshold: float = 0.7  # Cosine similarity threshold
    convergence_threshold: float = 0.001  # K-means convergence
    max_iterations: int = 100  # Maximum K-means iterations

    # Emotional weighting parameters
    sentiment_weight: float = 0.4  # Weight for sentiment analysis
    relationship_weight: float = 0.3  # Weight for relationship impact
    keyword_weight: float = 0.3  # Weight for emotional keywords
    emotional_keyword_bonus: float = 0.2  # Bonus for emotional indicators

    # Performance parameters
    cache_size: int = 1000  # LRU cache size for calculations
    batch_size: int = 100  # Batch size for bulk operations
    max_processing_time_ms: int = 10  # Max time per calculation
    clustering_interval_hours: int = 24  # How often to re-cluster

    # Memory filtering parameters
    min_confidence_for_clustering: float = 0.3  # Filter low-confidence memories
    min_emotional_weight: float = 0.1  # Filter emotionally neutral memories
    max_memory_age_days: int = 90  # Maximum age for clustering

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "MemoryAlgorithmConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.base_decay_rate <= 0 or self.base_decay_rate > 1:
            raise ValueError("base_decay_rate must be between 0 and 1")

        if self.emotional_multiplier_max < 1 or self.emotional_multiplier_max > 10:
            raise ValueError("emotional_multiplier_max must be between 1 and 10")

        if self.uncertainty_threshold < 0 or self.uncertainty_threshold > 1:
            raise ValueError("uncertainty_threshold must be between 0 and 1")

        if self.max_clusters < 2 or self.max_clusters > 50:
            raise ValueError("max_clusters must be between 2 and 50")

        if self.similarity_threshold < 0 or self.similarity_threshold > 1:
            raise ValueError("similarity_threshold must be between 0 and 1")


# Default personality-based configuration overrides
PERSONALITY_CONFIGS = {
    "detail_oriented": MemoryAlgorithmConfig(
        base_decay_rate=0.05,  # Slower decay for details
        uncertainty_threshold=0.2,  # More confident expressions
        emotional_multiplier_max=2.0,  # Less emotional amplification
    ),
    "name_focused": MemoryAlgorithmConfig(
        base_decay_rate=0.15,  # Faster decay for general details
        uncertainty_threshold=0.4,  # More uncertainty for details
        relationship_weight=0.5,  # Higher relationship weighting
    ),
    "emotional": MemoryAlgorithmConfig(
        emotional_multiplier_max=4.0,  # Strong emotional amplification
        sentiment_weight=0.6,  # Higher sentiment weighting
        base_decay_rate=0.08,  # Slower decay for emotional memories
    ),
    "analytical": MemoryAlgorithmConfig(
        uncertainty_threshold=0.15,  # Very confident expressions
        convergence_threshold=0.0001,  # Precise clustering
        max_clusters=15,  # More detailed clustering
    ),
    "forgetful": MemoryAlgorithmConfig(
        base_decay_rate=0.3,  # Fast memory decay
        uncertainty_threshold=0.6,  # High uncertainty
        access_bonus_rate=0.005,  # Reduced access bonus
    ),
    "trauma_sensitive": MemoryAlgorithmConfig(
        emotional_multiplier_max=5.0,  # Very strong emotional retention
        uncertainty_threshold=0.25,  # Confident about emotional memories
        min_confidence_for_clustering=0.5,  # Only cluster confident memories
    ),
    "social_memory": MemoryAlgorithmConfig(
        relationship_weight=0.5,  # Strong relationship focus
        max_clusters=12,  # Good relationship clustering
        similarity_threshold=0.6,  # Looser clustering for connections
    ),
}
