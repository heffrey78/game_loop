"""Pydantic models for game state representation and persistence."""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

# --- Player Related Models ---


class InventoryItem(BaseModel):
    """Represents an item held by the player."""

    item_id: UUID = Field(default_factory=uuid4)
    name: str
    description: str
    quantity: int = 1
    attributes: dict[str, Any] = Field(default_factory=dict)


class PlayerKnowledge(BaseModel):
    """Represents a piece of knowledge the player has acquired."""

    knowledge_id: UUID = Field(default_factory=uuid4)
    topic: str
    content: str
    discovered_at: datetime = Field(default_factory=datetime.now)
    source: str | None = None


class PlayerStats(BaseModel):
    """Represents the player's numerical statistics."""

    health: int = 100
    max_health: int = 100
    mana: int = 50
    max_mana: int = 50
    strength: int = 10
    dexterity: int = 10
    intelligence: int = 10


class PlayerProgress(BaseModel):
    """Tracks player progress, like quests or achievements."""

    active_quests: dict[UUID, dict[str, Any]] = Field(
        default_factory=dict
    )  # Quest ID -> Quest State
    completed_quests: list[UUID] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    flags: dict[str, bool] = Field(default_factory=dict)


class PlayerState(BaseModel):
    """Complete state of the player character."""

    player_id: UUID = Field(default_factory=uuid4)
    name: str = "Player"
    current_location_id: UUID | None = None
    inventory: list[InventoryItem] = Field(default_factory=list)
    knowledge: list[PlayerKnowledge] = Field(default_factory=list)
    stats: PlayerStats = Field(default_factory=PlayerStats)
    progress: PlayerProgress = Field(default_factory=PlayerProgress)
    visited_locations: list[UUID] = Field(default_factory=list)
    state_data_json: str | None = None


# --- World Related Models ---
class WorldObject(BaseModel):
    """Represents an object within a location."""

    object_id: UUID = Field(default_factory=uuid4)
    name: str
    description: str
    is_takeable: bool = False
    is_container: bool = False
    is_hidden: bool = False
    state: dict[str, Any] = Field(
        default_factory=dict
    )  # e.g., {"locked": True, "open": False}
    contained_items: list[UUID] = Field(
        default_factory=list
    )  # IDs of items inside, if container


class NonPlayerCharacter(BaseModel):
    """Represents an NPC within a location."""

    npc_id: UUID = Field(default_factory=uuid4)
    name: str
    description: str
    dialogue_state: str = "neutral"
    current_behavior: str = "idle"
    inventory: list[UUID] = Field(default_factory=list)
    knowledge: list[PlayerKnowledge] = Field(default_factory=list)
    state: dict[str, Any] = Field(default_factory=dict)


class Location(BaseModel):
    """Represents a single location in the game world."""

    location_id: UUID = Field(default_factory=uuid4)
    name: str
    description: str
    objects: dict[UUID, WorldObject] = Field(default_factory=dict)
    npcs: dict[UUID, NonPlayerCharacter] = Field(default_factory=dict)
    connections: dict[str, UUID] = Field(default_factory=dict)
    state_flags: dict[str, Any] = Field(default_factory=dict)
    first_visited: datetime | None = None
    last_visited: datetime | None = None


class EvolutionEvent(BaseModel):
    """Represents an event in the world evolution queue."""

    event_id: UUID = Field(default_factory=uuid4)
    trigger: str  # e.g., "npc_spawn", "weather_change"
    data: dict[str, Any] = Field(default_factory=dict)  # Specific data for the event
    priority: int = 0  # Higher priority events are processed first
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )  # Timestamp of queuing
    processed: bool = False
    processed_at: datetime | None = None


class WorldState(BaseModel):
    """Represents the entire state of the game world."""

    world_id: UUID = Field(default_factory=uuid4)
    locations: dict[UUID, Location] = Field(default_factory=dict)
    global_flags: dict[str, Any] = Field(default_factory=dict)
    current_time: datetime = Field(default_factory=datetime.now)
    evolution_queue: list["EvolutionEvent"] = Field(
        default_factory=list
    )  # Changed to EvolutionEvent
    state_data_json: str | None = None


# --- Session and Action Models ---


class GameSession(BaseModel):
    """Metadata for a saved game session."""

    session_id: UUID = Field(default_factory=uuid4)
    player_state_id: UUID
    world_state_id: UUID
    save_name: str = "New Save"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    game_version: str = "0.1.0"


class ActionResult(BaseModel):
    """Result of executing a player action."""

    success: bool = True
    feedback_message: str = ""

    # State Change Flags/Data
    location_change: bool = False
    new_location_id: UUID | None = None

    inventory_changes: list[dict[str, Any]] | None = None
    knowledge_updates: list[PlayerKnowledge] | None = None
    stat_changes: dict[str, int | float] | None = None
    progress_updates: dict[str, Any] | None = None

    object_changes: list[dict[str, Any]] | None = None
    npc_changes: list[dict[str, Any]] | None = None
    location_state_changes: dict[str, Any] | None = None
    global_flag_changes: dict[str, Any] | None = None  # Added

    triggers_evolution: bool = False
    evolution_trigger: str | None = (
        None  # Type of event, maps to EvolutionEvent.trigger
    )
    evolution_data: dict[str, Any] | None = None  # Data for the event
    priority: int = 5  # Priority for the evolution event
    timestamp: datetime = Field(
        default_factory=datetime.now
    )  # Timestamp for action result, can be used for event

    # Optionally add the originating command/intent
    command: str | None = None
    processed_input: Any | None = None


# --- Dynamic World Generation Models ---


class GenerationTrigger(BaseModel):
    """Represents a trigger for dynamic world generation."""
    
    trigger_id: UUID = Field(default_factory=uuid4)
    player_id: UUID
    session_id: UUID
    trigger_type: str  # e.g., "location_boundary", "exploration", "quest_need"
    trigger_context: dict[str, Any] = Field(default_factory=dict)
    location_id: UUID | None = None
    action_that_triggered: str | None = None
    priority_score: float = 0.5
    triggered_at: datetime = Field(default_factory=datetime.now)
    processed_at: datetime | None = None
    generation_result: dict[str, Any] | None = None


class GenerationOpportunity(BaseModel):
    """Represents an opportunity for content generation."""
    
    opportunity_id: UUID = Field(default_factory=uuid4)
    content_type: str  # "location", "npc", "object", "connection"
    opportunity_score: float
    generation_context: dict[str, Any] = Field(default_factory=dict)
    prerequisites: list[str] = Field(default_factory=list)
    estimated_generation_time: float = 0.0


class GenerationContext(BaseModel):
    """Context information for generation decisions."""
    
    player_state: PlayerState
    current_location: Location | None = None
    recent_actions: list[str] = Field(default_factory=list)
    world_gaps: list[str] = Field(default_factory=list)
    player_preferences: dict[str, Any] = Field(default_factory=dict)
    generation_history: list[str] = Field(default_factory=list)


class WorldGenerationResponse(BaseModel):
    """Response from world generation system."""
    
    has_new_content: bool = False
    generated_content: list[dict[str, Any]] = Field(default_factory=list)
    generation_time: float = 0.0
    quality_scores: dict[str, float] = Field(default_factory=dict)
    integration_required: bool = False


class PlayerPreferences(BaseModel):
    """Learned player preferences for content generation."""
    
    content_type_preferences: dict[str, float] = Field(default_factory=dict)
    theme_preferences: dict[str, float] = Field(default_factory=dict)
    difficulty_preference: float = 0.5
    exploration_style: str = "balanced"  # "thorough", "direct", "random"
    interaction_style: str = "balanced"  # "social", "combat", "puzzle"
    confidence_scores: dict[str, float] = Field(default_factory=dict)


class ExplorationPatterns(BaseModel):
    """Player exploration behavior patterns."""
    
    average_time_per_location: float = 0.0
    preferred_connection_types: list[str] = Field(default_factory=list)
    backtracking_frequency: float = 0.0
    discovery_thoroughness: float = 0.0
    risk_tolerance: float = 0.5


class EngagementMetrics(BaseModel):
    """Player engagement metrics."""
    
    session_duration: float = 0.0
    actions_per_minute: float = 0.0
    content_interaction_rate: float = 0.0
    exploration_depth: float = 0.0
    quest_completion_rate: float = 0.0
    satisfaction_indicators: dict[str, float] = Field(default_factory=dict)


class InterestPrediction(BaseModel):
    """Prediction of player interest in content."""
    
    content_type: str
    interest_score: float
    confidence: float
    reasoning: list[str] = Field(default_factory=list)


class ContentInteraction(BaseModel):
    """Record of player interaction with content."""
    
    interaction_id: UUID = Field(default_factory=uuid4)
    content_id: UUID
    content_type: str
    interaction_type: str  # "discovered", "examined", "used", "ignored"
    interaction_duration: float = 0.0
    satisfaction_score: int | None = None  # 1-5 rating
    interaction_context: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class DiscoveryEvent(BaseModel):
    """Event when player discovers generated content."""
    
    discovery_id: UUID = Field(default_factory=uuid4)
    player_id: UUID
    session_id: UUID
    content_id: UUID
    content_type: str
    discovery_method: str  # "exploration", "quest", "hint", "accident"
    location_id: UUID | None = None
    discovery_context: dict[str, Any] = Field(default_factory=dict)
    time_to_discovery_seconds: int | None = None
    discovered_at: datetime = Field(default_factory=datetime.now)


class InteractionEvent(BaseModel):
    """Event when player interacts with content."""
    
    interaction_id: UUID = Field(default_factory=uuid4)
    player_id: UUID
    content_id: UUID
    interaction_type: str
    interaction_duration_seconds: int | None = None
    interaction_outcome: str | None = None
    satisfaction_score: int | None = None
    interaction_data: dict[str, Any] = Field(default_factory=dict)
    interacted_at: datetime = Field(default_factory=datetime.now)


class DiscoveryPatterns(BaseModel):
    """Patterns in how content is discovered."""
    
    average_discovery_time: float = 0.0
    common_discovery_methods: list[tuple[str, float]] = Field(default_factory=list)
    discovery_success_rate: float = 0.0
    player_guidance_needed: float = 0.0


class ContentEffectiveness(BaseModel):
    """Effectiveness metrics for generated content."""
    
    discovery_rate: float = 0.0
    interaction_rate: float = 0.0
    average_satisfaction: float = 0.0
    completion_rate: float = 0.0
    replay_value: float = 0.0


class UndiscoveredContent(BaseModel):
    """Content that exists but hasn't been discovered."""
    
    content_id: UUID
    content_type: str
    difficulty_to_discover: float
    hints_available: list[str] = Field(default_factory=list)
    time_since_generation: float = 0.0


class DiscoveryAnalytics(BaseModel):
    """Comprehensive analytics on content discovery."""
    
    total_content_generated: int = 0
    total_content_discovered: int = 0
    discovery_rate_by_type: dict[str, float] = Field(default_factory=dict)
    average_time_to_discovery: dict[str, float] = Field(default_factory=dict)
    player_satisfaction_trends: list[float] = Field(default_factory=list)


class QualityAssessment(BaseModel):
    """Assessment of generated content quality."""
    
    content_id: UUID
    overall_quality_score: float
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    improvement_suggestions: list[str] = Field(default_factory=list)
    assessment_method: str
    assessed_at: datetime = Field(default_factory=datetime.now)


class SatisfactionData(BaseModel):
    """Player satisfaction data for content."""
    
    rating: int  # 1-5 scale
    feedback_text: str | None = None
    interaction_duration: float = 0.0
    completion_status: str = "incomplete"  # "completed", "abandoned", "incomplete"


class QualityIssue(BaseModel):
    """Identified quality issue in generated content."""
    
    issue_id: UUID = Field(default_factory=uuid4)
    content_id: UUID
    issue_type: str  # "consistency", "quality", "performance", "engagement"
    severity: str  # "low", "medium", "high", "critical"
    description: str
    suggested_fix: str | None = None
    detected_at: datetime = Field(default_factory=datetime.now)


class QualityTrends(BaseModel):
    """Trends in content quality over time."""
    
    quality_scores_over_time: list[tuple[datetime, float]] = Field(default_factory=list)
    improvement_rate: float = 0.0
    trend_direction: str = "stable"  # "improving", "declining", "stable"
    key_factors: list[str] = Field(default_factory=list)


class QualityReport(BaseModel):
    """Comprehensive quality report."""
    
    report_period: tuple[datetime, datetime]
    overall_quality_score: float
    quality_by_content_type: dict[str, float] = Field(default_factory=dict)
    major_issues: list[QualityIssue] = Field(default_factory=list)
    improvements_made: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class QualityImprovement(BaseModel):
    """Suggested improvement for quality issues."""
    
    improvement_id: UUID = Field(default_factory=uuid4)
    target_issue_types: list[str] = Field(default_factory=list)
    improvement_type: str  # "parameter_adjustment", "template_update", "algorithm_change"
    description: str
    expected_impact: float = 0.0
    implementation_difficulty: str = "medium"  # "easy", "medium", "hard"


class GenerationPipelineResult(BaseModel):
    """Result from coordinated generation pipeline."""
    
    success: bool = True
    generated_content: list[dict[str, Any]] = Field(default_factory=list)
    pipeline_time: float = 0.0
    coordination_quality: float = 0.0
    consistency_score: float = 0.0
    error_messages: list[str] = Field(default_factory=list)


class WorldGenerationStatus(BaseModel):
    """Current status of world generation systems."""
    
    active_generators: list[str] = Field(default_factory=list)
    generation_queue_size: int = 0
    average_generation_time: float = 0.0
    recent_quality_scores: list[float] = Field(default_factory=list)
    system_health: str = "healthy"  # "healthy", "degraded", "error"
    last_maintenance: datetime | None = None


class ContentGap(BaseModel):
    """Identified gap in world content."""
    
    gap_id: UUID = Field(default_factory=uuid4)
    gap_type: str  # "missing_connection", "empty_location", "no_npcs", "no_objects"
    location_id: UUID | None = None
    severity: str = "medium"  # "low", "medium", "high"
    suggested_content: list[str] = Field(default_factory=list)
    player_impact: float = 0.0


class GenerationPlan(BaseModel):
    """Plan for coordinated content generation."""
    
    plan_id: UUID = Field(default_factory=uuid4)
    generation_requests: list[dict[str, Any]] = Field(default_factory=list)
    coordination_strategy: str = "sequential"  # "sequential", "parallel", "mixed"
    estimated_time: float = 0.0
    quality_targets: dict[str, float] = Field(default_factory=dict)


class LocationWithContent(BaseModel):
    """Location generated with associated content."""
    
    location: Location
    generated_npcs: list[NonPlayerCharacter] = Field(default_factory=list)
    generated_objects: list[WorldObject] = Field(default_factory=list)
    generated_connections: list[dict[str, Any]] = Field(default_factory=list)
    generation_metadata: dict[str, Any] = Field(default_factory=dict)


class LocationExpansion(BaseModel):
    """Expansion of existing location with new content."""
    
    location_id: UUID
    expansion_type: str  # "npc_addition", "object_addition", "connection_addition"
    added_content: list[dict[str, Any]] = Field(default_factory=list)
    integration_quality: float = 0.0


class ContentCluster(BaseModel):
    """Thematically connected group of content."""
    
    cluster_id: UUID = Field(default_factory=uuid4)
    theme: str
    anchor_location_id: UUID
    cluster_content: list[dict[str, Any]] = Field(default_factory=list)
    coherence_score: float = 0.0


class ConsistencyReport(BaseModel):
    """Report on content consistency validation."""
    
    overall_consistency: float = 0.0
    theme_consistency: float = 0.0
    narrative_consistency: float = 0.0
    mechanical_consistency: float = 0.0
    inconsistencies_found: list[str] = Field(default_factory=list)


class GenerationRequest(BaseModel):
    """Request for content generation."""
    
    request_id: UUID = Field(default_factory=uuid4)
    content_type: str
    generation_context: dict[str, Any] = Field(default_factory=dict)
    priority: float = 0.5
    quality_requirements: dict[str, float] = Field(default_factory=dict)
    dependencies: list[UUID] = Field(default_factory=list)


class GenerationRecovery(BaseModel):
    """Recovery strategy for failed generation."""
    
    recovery_type: str  # "retry", "fallback", "skip", "manual"
    fallback_content: dict[str, Any] | None = None
    retry_parameters: dict[str, Any] = Field(default_factory=dict)
    recovery_success: bool = False


class GeneratedContent(BaseModel):
    """Generic container for any generated content."""
    
    content_id: UUID = Field(default_factory=uuid4)
    content_type: str
    content_data: dict[str, Any] = Field(default_factory=dict)
    generation_metadata: dict[str, Any] = Field(default_factory=dict)
    quality_score: float = 0.0
    generated_at: datetime = Field(default_factory=datetime.now)


class PlayerFeedback(BaseModel):
    """Player feedback for preference learning."""
    
    feedback_id: UUID = Field(default_factory=uuid4)
    player_id: UUID
    content_id: UUID | None = None
    feedback_type: str  # "rating", "behavior", "explicit"
    feedback_data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class PerformanceMetrics(BaseModel):
    """Performance metrics for generation systems."""
    
    generation_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    database_queries: int = 0
    cache_hit_rate: float = 0.0


class QualityFeedback(BaseModel):
    """Quality feedback for content generation systems."""
    
    feedback_id: UUID = Field(default_factory=uuid4)
    player_id: UUID | None = None
    content_id: UUID | None = None
    rating: int | None = None  # 1-5 scale
    feedback_text: str | None = None
    feedback_category: str = "general"  # "quality", "engagement", "relevance", etc.
    timestamp: datetime = Field(default_factory=datetime.now)
