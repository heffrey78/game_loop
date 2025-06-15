"""
Object models for dynamic object generation and management.

This module defines the data structures for object generation, properties,
interactions, and management within the game world.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from game_loop.state.models import Location, WorldObject


@dataclass
class ObjectProperties:
    """Properties defining the physical and conceptual characteristics of an object."""

    name: str
    object_type: str
    material: str = "unknown"
    size: str = "medium"
    weight: str = "normal"
    durability: str = "sturdy"
    value: int = 0
    special_properties: list[str] = field(default_factory=list)
    cultural_significance: str = "common"
    description: str = ""

    def __post_init__(self) -> None:
        """Validate object properties after initialization."""
        if not self.name:
            raise ValueError("Object name cannot be empty")
        if not self.object_type:
            raise ValueError("Object type cannot be empty")


@dataclass
class ObjectInteractions:
    """Interaction capabilities and behaviors for an object."""

    available_actions: list[str] = field(default_factory=list)
    use_requirements: dict[str, str] = field(default_factory=dict)
    interaction_results: dict[str, str] = field(default_factory=dict)
    state_changes: dict[str, str] = field(default_factory=dict)
    consumable: bool = False
    portable: bool = True
    examination_text: str = ""
    hidden_properties: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set default actions if none provided."""
        if not self.available_actions:
            self.available_actions = ["examine"]
            if self.portable:
                self.available_actions.extend(["take", "drop"])


@dataclass
class ObjectGenerationContext:
    """Context information for intelligent object generation."""

    location: Location
    location_theme: LocationTheme
    generation_purpose: str
    existing_objects: list[WorldObject]
    player_level: int = 3
    constraints: dict[str, Any] = field(default_factory=dict)
    world_state_snapshot: dict[str, Any] = field(default_factory=dict)
    generation_timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Validate generation context."""
        if self.player_level < 1:
            raise ValueError("Player level must be at least 1")

        valid_purposes = [
            "populate_location",
            "quest_related",
            "random_encounter",
            "player_request",
            "narrative_enhancement",
            "environmental_storytelling",
        ]
        if self.generation_purpose not in valid_purposes:
            raise ValueError(f"Invalid generation purpose: {self.generation_purpose}")


@dataclass
class ObjectPlacement:
    """Information about where and how an object is placed in a location."""

    object_id: UUID
    location_id: UUID
    placement_type: str  # "floor", "table", "shelf", "hidden", "embedded"
    visibility: str = "visible"  # "visible", "hidden", "partially_hidden"
    accessibility: str = "accessible"  # "accessible", "blocked", "requires_tool"
    spatial_description: str = ""
    discovery_difficulty: int = 1  # 1-10 scale
    placement_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate placement information."""
        valid_types = [
            "floor",
            "table",
            "shelf",
            "wall",
            "ceiling",
            "hidden",
            "embedded",
            "container",
        ]
        if self.placement_type not in valid_types:
            raise ValueError(f"Invalid placement type: {self.placement_type}")

        if not (1 <= self.discovery_difficulty <= 10):
            raise ValueError("Discovery difficulty must be between 1 and 10")


@dataclass
class GeneratedObject:
    """Complete generated object with all associated data."""

    base_object: WorldObject
    properties: ObjectProperties
    interactions: ObjectInteractions
    generation_metadata: dict[str, Any] = field(default_factory=dict)
    embedding_vector: list[float] = field(default_factory=list)
    placement_info: ObjectPlacement | None = None

    def __post_init__(self) -> None:
        """Ensure generated object consistency."""
        if self.base_object.name != self.properties.name:
            self.base_object.name = self.properties.name

        if not self.base_object.description and self.properties.description:
            self.base_object.description = self.properties.description


@dataclass
class ObjectArchetype:
    """Definition of an object archetype with location affinities and templates."""

    name: str
    description: str
    typical_properties: ObjectProperties
    location_affinities: dict[str, float] = field(default_factory=dict)
    interaction_templates: dict[str, list[str]] = field(default_factory=dict)
    cultural_variations: dict[str, dict[str, Any]] = field(default_factory=dict)
    rarity: str = "common"  # "common", "uncommon", "rare", "legendary"

    def __post_init__(self) -> None:
        """Validate archetype definition."""
        if not self.name:
            raise ValueError("Archetype name cannot be empty")

        valid_rarities = ["common", "uncommon", "rare", "legendary"]
        if self.rarity not in valid_rarities:
            raise ValueError(f"Invalid rarity: {self.rarity}")


@dataclass
class ObjectSearchCriteria:
    """Criteria for searching objects using semantic search."""

    query_text: str = ""
    object_types: list[str] = field(default_factory=list)
    location_themes: list[str] = field(default_factory=list)
    properties_filter: dict[str, Any] = field(default_factory=dict)
    interactions_required: list[str] = field(default_factory=list)
    cultural_significance: list[str] = field(default_factory=list)
    rarity_levels: list[str] = field(default_factory=list)
    max_results: int = 10
    similarity_threshold: float = 0.7

    def __post_init__(self) -> None:
        """Validate search criteria."""
        if self.max_results <= 0:
            raise ValueError("Max results must be positive")

        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")


@dataclass
class ObjectValidationResult:
    """Result of object validation including consistency checks."""

    is_valid: bool
    validation_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    consistency_score: float = 1.0
    theme_alignment: float = 1.0
    placement_suitability: float = 1.0

    def __post_init__(self) -> None:
        """Set validity based on presence of errors."""
        if self.validation_errors:
            self.is_valid = False


@dataclass
class ObjectStorageResult:
    """Result of object storage operation."""

    success: bool
    object_id: UUID | None = None
    storage_errors: list[str] = field(default_factory=list)
    embedding_generated: bool = False
    placement_stored: bool = False
    storage_duration_ms: float = 0.0

    def __post_init__(self) -> None:
        """Validate storage result."""
        if self.success and not self.object_id:
            raise ValueError("Successful storage must include object_id")


@dataclass
class GenerationMetrics:
    """Metrics for tracking object generation performance."""

    generation_start_time: datetime = field(default_factory=datetime.now)
    generation_end_time: datetime | None = None
    context_collection_time_ms: float = 0.0
    archetype_selection_time_ms: float = 0.0
    llm_generation_time_ms: float = 0.0
    embedding_generation_time_ms: float = 0.0
    validation_time_ms: float = 0.0
    storage_time_ms: float = 0.0
    total_time_ms: float = 0.0
    cache_hit: bool = False
    tokens_used: int = 0

    def mark_complete(self):
        """Mark generation as complete and calculate total time."""
        self.generation_end_time = datetime.now()
        if self.total_time_ms == 0.0:  # Only calculate if not manually set
            duration = self.generation_end_time - self.generation_start_time
            self.total_time_ms = duration.total_seconds() * 1000


@dataclass
class ObjectTheme:
    """Theme definition for object generation consistency."""

    name: str
    description: str
    typical_materials: list[str] = field(default_factory=list)
    common_object_types: list[str] = field(default_factory=list)
    cultural_elements: dict[str, str] = field(default_factory=dict)
    style_descriptors: list[str] = field(default_factory=list)
    forbidden_elements: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate theme definition."""
        if not self.name:
            raise ValueError("Theme name cannot be empty")


@dataclass
class ObjectGenerationRequest:
    """Request for object generation with specific parameters."""

    location_id: UUID
    purpose: str
    requested_type: str | None = None
    quantity: int = 1
    constraints: dict[str, Any] = field(default_factory=dict)
    priority: str = "normal"  # "low", "normal", "high", "urgent"
    requester: str = "system"  # "system", "player", "npc", "quest"

    def __post_init__(self) -> None:
        """Validate generation request."""
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")

        valid_priorities = ["low", "normal", "high", "urgent"]
        if self.priority not in valid_priorities:
            raise ValueError(f"Invalid priority: {self.priority}")
