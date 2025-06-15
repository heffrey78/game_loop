"""
Data models for location generation system.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from .navigation_models import ExpansionPoint


@dataclass
class LocationTheme:
    """Location theme definition."""

    name: str
    description: str
    visual_elements: list[str]
    atmosphere: str
    typical_objects: list[str]
    typical_npcs: list[str]
    generation_parameters: dict[str, Any]
    theme_id: UUID | None = None
    parent_theme_id: UUID | None = None
    created_at: datetime | None = None


@dataclass
class PlayerLocationPreferences:
    """Player preferences for location generation."""

    environments: list[str]
    interaction_style: str
    complexity_level: str
    preferred_themes: list[str] = field(default_factory=list)
    avoided_themes: list[str] = field(default_factory=list)
    exploration_preference: str = "balanced"
    social_preference: str = "balanced"


@dataclass
class AdjacentLocationContext:
    """Context from locations adjacent to expansion point."""

    location_id: UUID
    direction: str
    name: str
    description: str
    theme: str
    short_description: str
    objects: list[str] = field(default_factory=list)
    npcs: list[str] = field(default_factory=list)


@dataclass
class NarrativeContext:
    """Narrative context for location generation."""

    current_quests: list[str] = field(default_factory=list)
    story_themes: list[str] = field(default_factory=list)
    player_actions: list[str] = field(default_factory=list)
    world_events: list[str] = field(default_factory=list)
    narrative_tension: str = "neutral"


@dataclass
class LocationGenerationContext:
    """Context for generating new locations."""

    expansion_point: ExpansionPoint
    adjacent_locations: list[AdjacentLocationContext]
    player_preferences: PlayerLocationPreferences
    world_themes: list[LocationTheme]
    narrative_context: NarrativeContext | None = None
    generation_constraints: dict[str, Any] = field(default_factory=dict)
    world_rules: list[str] = field(default_factory=list)


@dataclass
class GeneratedLocation:
    """Result of location generation."""

    name: str
    description: str
    theme: LocationTheme
    location_type: str
    objects: list[str]
    npcs: list[str]
    connections: dict[str, str]
    metadata: dict[str, Any]
    short_description: str = ""
    atmosphere: str = ""
    special_features: list[str] = field(default_factory=list)
    generation_context: LocationGenerationContext | None = None


@dataclass
class ValidationResult:
    """Location validation result."""

    is_valid: bool
    issues: list[str]
    suggestions: list[str]
    confidence_score: float
    overall_score: float = 0.0
    thematic_score: float = 0.0
    logical_score: float = 0.0
    complexity_score: float = 0.0
    uniqueness_score: float = 0.0
    approval: bool = False


@dataclass
class ThemeTransitionRules:
    """Rules for transitioning between themes."""

    from_theme: str
    to_theme: str
    compatibility_score: float
    transition_requirements: list[str]
    forbidden_elements: list[str] = field(default_factory=list)
    required_elements: list[str] = field(default_factory=list)
    transition_description: str = ""


@dataclass
class ThemeContent:
    """Theme-specific content elements."""

    theme_name: str
    objects: list[str]
    npcs: list[str]
    descriptions: list[str]
    atmospheric_elements: list[str]
    special_features: list[str] = field(default_factory=list)


@dataclass
class StorageResult:
    """Result of location storage operation."""

    success: bool
    location_id: UUID | None = None
    error_message: str | None = None
    storage_time_ms: int | None = None
    embedding_generated: bool = False


@dataclass
class EmbeddingUpdateResult:
    """Result of embedding update operation."""

    success: bool
    location_id: UUID
    embedding_vector: list[float] | None = None
    update_time_ms: int | None = None
    error_message: str | None = None


@dataclass
class EnrichedContext:
    """Enhanced context with additional analysis."""

    base_context: LocationGenerationContext
    historical_patterns: dict[str, Any]
    player_behavior_analysis: dict[str, Any]
    world_consistency_factors: dict[str, Any]
    generation_hints: list[str] = field(default_factory=list)
    priority_elements: list[str] = field(default_factory=list)


@dataclass
class LocationConnection:
    """Connection between locations."""

    from_location_id: UUID
    to_location_id: UUID
    direction: str
    connection_type: str
    description: str
    is_bidirectional: bool = True
    travel_time: int | None = None
    requirements: list[str] = field(default_factory=list)


@dataclass
class CachedGeneration:
    """Cached location generation result."""

    context_hash: str
    generated_location: GeneratedLocation
    cache_expires_at: datetime
    usage_count: int = 0
    created_at: datetime | None = None

    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return datetime.now() > self.cache_expires_at

    @property
    def time_until_expiry(self) -> timedelta:
        """Get time until cache expires."""
        return self.cache_expires_at - datetime.now()


@dataclass
class GenerationMetrics:
    """Metrics for location generation performance."""

    generation_time_ms: int
    context_collection_time_ms: int
    llm_response_time_ms: int
    validation_time_ms: int
    storage_time_ms: int
    cache_hit: bool = False
    retry_count: int = 0
    total_time_ms: int | None = None

    def __post_init__(self) -> None:
        """Calculate total time if not provided."""
        if self.total_time_ms is None:
            self.total_time_ms = (
                self.generation_time_ms
                + self.context_collection_time_ms
                + self.llm_response_time_ms
                + self.validation_time_ms
                + self.storage_time_ms
            )
