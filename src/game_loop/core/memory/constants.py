"""Constants and thresholds for the emotional memory system."""

from dataclasses import dataclass
from typing import Dict, Any


class EmotionalThresholds:
    """Emotional analysis thresholds and constants."""
    
    # Memory significance thresholds
    TRAUMA_WEIGHT_THRESHOLD = 0.7
    HIGH_SIGNIFICANCE_THRESHOLD = 0.8
    FORMATIVE_WEIGHT_THRESHOLD = 0.6
    PEAK_EXPERIENCE_THRESHOLD = 0.8
    HIGH_RELATIONSHIP_IMPACT = 0.7
    MODERATE_SIGNIFICANCE = 0.5
    LOW_SIGNIFICANCE = 0.3
    
    # Emotional intensity thresholds
    HIGH_INTENSITY = 0.8
    MODERATE_INTENSITY = 0.6
    LOW_INTENSITY = 0.3
    
    # Protection level thresholds
    MIN_TRUST_FOR_TRAUMA = 0.8
    MIN_TRUST_FOR_PROTECTED = 0.6
    MIN_TRUST_FOR_SENSITIVE = 0.4
    MIN_TRUST_FOR_PRIVATE = 0.2
    
    # Mood accessibility thresholds
    HIGH_MOOD_ACCESSIBILITY = 0.7
    MODERATE_MOOD_ACCESSIBILITY = 0.5
    LOW_MOOD_ACCESSIBILITY = 0.3
    
    # Clustering thresholds
    MIN_CLUSTER_SIZE = 3
    MAX_CLUSTERS = 20
    ASSOCIATION_THRESHOLD = 0.6
    SIMILARITY_THRESHOLD = 0.7
    
    # Temporal thresholds (in hours)
    RECENT_MEMORY_HOURS = 24
    OLD_MEMORY_HOURS = 168  # 1 week
    VERY_OLD_MEMORY_HOURS = 720  # 30 days
    
    # Performance limits
    MAX_MEMORY_CAPACITY = 10000
    MAX_DAILY_TRAUMA_EXPOSURE = 3
    MIN_TIME_BETWEEN_TRAUMA_HOURS = 2.0
    
    # Cache limits
    DEFAULT_CACHE_SIZE = 1000
    LARGE_CACHE_SIZE = 5000
    SMALL_CACHE_SIZE = 500
    
    # Time limits (in seconds)
    DEFAULT_CACHE_TTL = 3600  # 1 hour
    SHORT_CACHE_TTL = 1800   # 30 minutes
    LONG_CACHE_TTL = 7200    # 2 hours
    
    # Processing limits
    MAX_CLUSTERING_MEMORIES = 100
    CLUSTERING_TIMEOUT_SECONDS = 30
    ANALYSIS_TIMEOUT_SECONDS = 10
    
    # Safety thresholds
    CRISIS_SENSITIVITY_THRESHOLD = 0.9
    EMERGENCY_RESPONSE_THRESHOLD = 0.95
    TRIGGER_DETECTION_THRESHOLD = 0.6


class EmotionalTypeThresholds:
    """Thresholds specific to different emotional memory types."""
    
    TRAUMATIC_INDICATORS = {
        'min_emotional_weight': 0.7,
        'min_negative_sentiment': -0.6,
        'required_keywords': {'trauma', 'traumatic', 'nightmare', 'terrified', 'horrified'}
    }
    
    PEAK_POSITIVE_INDICATORS = {
        'min_emotional_weight': 0.8,
        'min_positive_sentiment': 0.6,
        'required_keywords': {'best day', 'happiest', 'amazing', 'incredible', 'unforgettable'}
    }
    
    CORE_ATTACHMENT_INDICATORS = {
        'min_relationship_impact': 0.7,
        'required_keywords': {'love', 'loved', 'care', 'caring', 'devoted', 'cherish'}
    }
    
    FORMATIVE_INDICATORS = {
        'min_emotional_weight': 0.6,
        'required_keywords': {'first time', 'learned', 'realized', 'understood', 'changed me'}
    }


class MoodTransitionPatterns:
    """Patterns for mood transitions and their probabilities."""
    
    NATURAL_TRANSITIONS = {
        'joyful': {'content': 0.3, 'excited': 0.2, 'neutral': 0.1},
        'content': {'joyful': 0.2, 'neutral': 0.3, 'nostalgic': 0.1},
        'neutral': {'content': 0.2, 'melancholy': 0.1, 'curious': 0.2},
        'melancholy': {'neutral': 0.3, 'hopeful': 0.1, 'nostalgic': 0.2},
        'anxious': {'neutral': 0.2, 'fearful': 0.3, 'melancholy': 0.2},
        'angry': {'neutral': 0.3, 'melancholy': 0.2, 'anxious': 0.1},
        'fearful': {'anxious': 0.4, 'neutral': 0.2, 'melancholy': 0.1},
    }
    
    MOOD_STABILITY_FACTORS = {
        'joyful': 0.6,      # Moderately stable
        'content': 0.8,     # Very stable
        'neutral': 0.9,     # Most stable
        'melancholy': 0.5,  # Less stable
        'anxious': 0.3,     # Unstable
        'angry': 0.2,       # Very unstable
        'fearful': 0.2,     # Very unstable
        'excited': 0.3,     # Unstable
        'nostalgic': 0.6,   # Moderately stable
        'hopeful': 0.7,     # Stable
    }


class ProtectionMechanismConfig:
    """Configuration for trauma protection mechanisms."""
    
    PROTECTION_EFFECTIVENESS = {
        'access_control': 0.8,
        'gradual_exposure': 0.7,
        'safety_checks': 0.9,
        'therapeutic_support': 0.6,
        'emergency_containment': 0.9,
        'trust_gating': 0.7,
    }
    
    TRUST_THRESHOLDS = {
        'public': 0.0,
        'private': 0.3,
        'sensitive': 0.6,
        'protected': 0.8,
        'traumatic': 0.9,
    }
    
    DAILY_EXPOSURE_LIMITS = {
        'traumatic': 2,
        'significant_loss': 3,
        'conflict': 5,
        'general': 10,
    }


@dataclass
class EmotionalMemorySystemConfig:
    """Comprehensive configuration for the emotional memory system."""
    
    # Analysis configuration
    significance_thresholds: Dict[str, float] = None
    emotional_type_thresholds: Dict[str, Dict[str, Any]] = None
    
    # Cache configuration
    cache_sizes: Dict[str, int] = None
    cache_ttls: Dict[str, int] = None
    
    # Performance configuration
    processing_limits: Dict[str, int] = None
    timeout_limits: Dict[str, int] = None
    
    # Safety configuration
    protection_config: Dict[str, Any] = None
    trauma_limits: Dict[str, int] = None
    
    # Feature toggles
    enable_clustering: bool = True
    enable_trauma_protection: bool = True
    enable_mood_tracking: bool = True
    enable_therapeutic_mode: bool = False
    strict_protection_mode: bool = True
    
    # Debug settings
    debug_mode: bool = False
    verbose_logging: bool = False
    performance_tracking: bool = True
    
    def __post_init__(self):
        """Initialize default configurations."""
        if self.significance_thresholds is None:
            self.significance_thresholds = {
                'trauma': EmotionalThresholds.TRAUMA_WEIGHT_THRESHOLD,
                'high_significance': EmotionalThresholds.HIGH_SIGNIFICANCE_THRESHOLD,
                'formative': EmotionalThresholds.FORMATIVE_WEIGHT_THRESHOLD,
                'moderate': EmotionalThresholds.MODERATE_SIGNIFICANCE,
                'low': EmotionalThresholds.LOW_SIGNIFICANCE,
            }
        
        if self.cache_sizes is None:
            self.cache_sizes = {
                'significance_cache': EmotionalThresholds.DEFAULT_CACHE_SIZE,
                'mood_cache': EmotionalThresholds.SMALL_CACHE_SIZE,
                'clustering_cache': EmotionalThresholds.SMALL_CACHE_SIZE,
                'trauma_cache': EmotionalThresholds.SMALL_CACHE_SIZE,
                'dialogue_cache': EmotionalThresholds.DEFAULT_CACHE_SIZE,
            }
        
        if self.cache_ttls is None:
            self.cache_ttls = {
                'significance_cache': EmotionalThresholds.DEFAULT_CACHE_TTL,
                'mood_cache': EmotionalThresholds.SHORT_CACHE_TTL,
                'clustering_cache': EmotionalThresholds.LONG_CACHE_TTL,
                'trauma_cache': EmotionalThresholds.SHORT_CACHE_TTL,
                'dialogue_cache': EmotionalThresholds.SHORT_CACHE_TTL,
            }
        
        if self.processing_limits is None:
            self.processing_limits = {
                'max_memories_for_clustering': EmotionalThresholds.MAX_CLUSTERING_MEMORIES,
                'max_memory_capacity': EmotionalThresholds.MAX_MEMORY_CAPACITY,
                'max_associations_per_memory': 10,
                'max_clusters_per_npc': EmotionalThresholds.MAX_CLUSTERS,
            }
        
        if self.timeout_limits is None:
            self.timeout_limits = {
                'analysis_timeout': EmotionalThresholds.ANALYSIS_TIMEOUT_SECONDS,
                'clustering_timeout': EmotionalThresholds.CLUSTERING_TIMEOUT_SECONDS,
                'database_timeout': 30,
                'llm_timeout': 60,
            }
        
        if self.protection_config is None:
            self.protection_config = {
                'min_trust_for_trauma': EmotionalThresholds.MIN_TRUST_FOR_TRAUMA,
                'crisis_threshold': EmotionalThresholds.CRISIS_SENSITIVITY_THRESHOLD,
                'emergency_threshold': EmotionalThresholds.EMERGENCY_RESPONSE_THRESHOLD,
                'trigger_threshold': EmotionalThresholds.TRIGGER_DETECTION_THRESHOLD,
            }
        
        if self.trauma_limits is None:
            self.trauma_limits = {
                'max_daily_exposure': EmotionalThresholds.MAX_DAILY_TRAUMA_EXPOSURE,
                'min_hours_between_exposure': EmotionalThresholds.MIN_TIME_BETWEEN_TRAUMA_HOURS,
                'max_therapeutic_sessions_per_day': 2,
                'crisis_cooldown_hours': 24,
            }
    
    def get_threshold(self, category: str, threshold_name: str, default: float = 0.5) -> float:
        """Get a threshold value safely."""
        category_dict = getattr(self, f"{category}_thresholds", {})
        return category_dict.get(threshold_name, default)
    
    def get_cache_config(self, cache_name: str) -> Dict[str, int]:
        """Get cache configuration for a specific cache."""
        return {
            'size': self.cache_sizes.get(cache_name, EmotionalThresholds.DEFAULT_CACHE_SIZE),
            'ttl': self.cache_ttls.get(cache_name, EmotionalThresholds.DEFAULT_CACHE_TTL),
        }
    
    def get_processing_limit(self, limit_name: str, default: int = 100) -> int:
        """Get a processing limit safely."""
        return self.processing_limits.get(limit_name, default)
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled."""
        return getattr(self, f"enable_{feature_name}", False)