"""
NPC data models for generation, storage, and interaction.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from ...state.models import Location, NonPlayerCharacter
from .location_models import LocationTheme


@dataclass
class NPCPersonality:
    """Defines NPC personality traits and characteristics."""

    name: str
    archetype: str  # e.g., "merchant", "guard", "scholar", "hermit"
    traits: list[str] = field(
        default_factory=list
    )  # e.g., ["friendly", "knowledgeable", "cautious"]
    motivations: list[str] = field(default_factory=list)
    fears: list[str] = field(default_factory=list)
    speech_patterns: dict[str, str] = field(
        default_factory=dict
    )  # e.g., {"formality": "casual", "verbosity": "concise"}
    relationship_tendencies: dict[str, float] = field(
        default_factory=dict
    )  # tendency scores for different relationship types


@dataclass
class NPCKnowledge:
    """Represents what an NPC knows about the world."""

    world_knowledge: dict[str, Any] = field(
        default_factory=dict
    )  # General world information
    local_knowledge: dict[str, Any] = field(
        default_factory=dict
    )  # Location-specific information
    personal_history: list[str] = field(
        default_factory=list
    )  # Personal experiences and memories
    relationships: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )  # Known people and relationships
    secrets: list[str] = field(
        default_factory=list
    )  # Information they might share under certain conditions
    expertise_areas: list[str] = field(
        default_factory=list
    )  # Areas of specialized knowledge


@dataclass
class NPCDialogueState:
    """Manages NPC dialogue and interaction state."""

    current_mood: str = "neutral"
    relationship_level: float = 0.0  # -1.0 to 1.0
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    active_topics: list[str] = field(default_factory=list)
    available_quests: list[str] = field(default_factory=list)
    interaction_count: int = 0
    last_interaction: datetime | None = None


@dataclass
class NPCGenerationContext:
    """Context for generating contextually appropriate NPCs."""

    location: Location
    location_theme: LocationTheme
    nearby_npcs: list[NonPlayerCharacter] = field(default_factory=list)
    world_state_snapshot: dict[str, Any] = field(default_factory=dict)
    player_level: int = 1
    generation_purpose: str = (
        "populate_location"  # e.g., "populate_location", "quest_related", "random_encounter"
    )
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedNPC:
    """Complete generated NPC with all associated data."""

    base_npc: NonPlayerCharacter
    personality: NPCPersonality
    knowledge: NPCKnowledge
    dialogue_state: NPCDialogueState
    generation_metadata: dict[str, Any] = field(default_factory=dict)
    embedding_vector: list[float] | None = None


@dataclass
class NPCStorageResult:
    """Result of storing an NPC to the database."""

    success: bool
    npc_id: UUID | None = None
    storage_time_ms: int = 0
    embedding_generated: bool = False
    error_message: str | None = None


@dataclass
class NPCValidationResult:
    """Result of validating a generated NPC."""

    is_valid: bool
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    confidence_score: float = 0.0
    personality_score: float = 0.0
    knowledge_score: float = 0.0
    consistency_score: float = 0.0
    approval: bool = False


@dataclass
class NPCGenerationMetrics:
    """Performance metrics for NPC generation."""

    generation_time_ms: int = 0
    context_collection_time_ms: int = 0
    llm_response_time_ms: int = 0
    validation_time_ms: int = 0
    storage_time_ms: int = 0
    total_time_ms: int = 0
    cache_hit: bool = False

    def __post_init__(self) -> None:
        """Calculate total time if not provided."""
        if self.total_time_ms == 0:
            self.total_time_ms = (
                self.generation_time_ms
                + self.context_collection_time_ms
                + self.llm_response_time_ms
                + self.validation_time_ms
                + self.storage_time_ms
            )


@dataclass
class NPCArchetype:
    """Defines an NPC archetype with its characteristics."""

    name: str
    description: str
    typical_traits: list[str] = field(default_factory=list)
    typical_motivations: list[str] = field(default_factory=list)
    speech_patterns: dict[str, str] = field(default_factory=dict)
    location_affinities: dict[str, float] = field(
        default_factory=dict
    )  # theme -> affinity score
    archetype_id: UUID | None = None


@dataclass
class DialogueContext:
    """Context for generating NPC dialogue."""

    npc: GeneratedNPC
    player_input: str
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    current_location: Location | None = None
    world_context: dict[str, Any] = field(default_factory=dict)
    interaction_type: str = "casual"  # casual, quest, trade, etc.


@dataclass
class DialogueResponse:
    """Generated dialogue response from an NPC."""

    response_text: str
    mood_change: str | None = None
    relationship_change: float = 0.0
    new_topics: list[str] = field(default_factory=list)
    quest_offered: str | None = None
    knowledge_shared: dict[str, Any] = field(default_factory=dict)
    response_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NPCSearchCriteria:
    """Criteria for searching NPCs semantically."""

    query_text: str | None = None
    archetype: str | None = None
    location_id: UUID | None = None
    personality_traits: list[str] = field(default_factory=list)
    knowledge_areas: list[str] = field(default_factory=list)
    relationship_level_min: float | None = None
    relationship_level_max: float | None = None
    max_results: int = 10
