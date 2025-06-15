"""
Connection Data Models for World Connection Management System.

This module defines comprehensive data structures for world connections,
connection metadata, generation context, and validation results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from game_loop.state.models import Location


@dataclass
class ConnectionProperties:
    """Properties defining a connection between locations."""

    connection_type: str  # passage, portal, bridge, tunnel, etc.
    difficulty: int  # 1-10 traversal difficulty
    travel_time: int  # time in seconds to traverse
    description: str
    visibility: str  # visible, hidden, secret
    requirements: list[str]  # conditions to use connection
    reversible: bool = True
    condition_flags: dict[str, Any] = field(default_factory=dict)
    special_features: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate connection properties after initialization."""
        if not (1 <= self.difficulty <= 10):
            raise ValueError("Connection difficulty must be between 1 and 10")

        if self.travel_time < 0:
            raise ValueError("Travel time cannot be negative")

        valid_visibility = ["visible", "hidden", "secret", "partially_hidden"]
        if self.visibility not in valid_visibility:
            raise ValueError(f"Invalid visibility: {self.visibility}")

        valid_types = [
            "passage",
            "portal",
            "bridge",
            "tunnel",
            "path",
            "road",
            "stairway",
            "gateway",
            "door",
            "teleporter",
        ]
        if self.connection_type not in valid_types:
            raise ValueError(f"Invalid connection type: {self.connection_type}")


@dataclass
class ConnectionGenerationContext:
    """Context for generating connections between locations."""

    source_location: Location
    target_location: Location
    generation_purpose: str  # expand_world, quest_path, exploration, etc.
    distance_preference: str  # short, medium, long
    terrain_constraints: dict[str, Any] = field(default_factory=dict)
    narrative_context: dict[str, Any] = field(default_factory=dict)
    existing_connections: list[str] = field(default_factory=list)
    player_level: int = 1
    world_state_snapshot: dict[str, Any] = field(default_factory=dict)
    generation_timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Validate generation context after initialization."""
        valid_purposes = [
            "expand_world",
            "quest_path",
            "exploration",
            "narrative_enhancement",
            "player_request",
        ]
        if self.generation_purpose not in valid_purposes:
            raise ValueError(f"Invalid generation purpose: {self.generation_purpose}")

        valid_distances = ["short", "medium", "long", "variable"]
        if self.distance_preference not in valid_distances:
            raise ValueError(f"Invalid distance preference: {self.distance_preference}")


@dataclass
class ConnectionValidationResult:
    """Result of connection validation checks."""

    is_valid: bool
    validation_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    consistency_score: float = 1.0
    logical_soundness: float = 1.0
    terrain_compatibility: float = 1.0

    def __post_init__(self) -> None:
        """Validate result scores after initialization."""
        if self.validation_errors:
            self.is_valid = False

        # Ensure scores are in valid range
        for score_name, score in [
            ("consistency_score", self.consistency_score),
            ("logical_soundness", self.logical_soundness),
            ("terrain_compatibility", self.terrain_compatibility),
        ]:
            if not (0.0 <= score <= 1.0):
                raise ValueError(f"{score_name} must be between 0.0 and 1.0")


@dataclass
class GeneratedConnection:
    """Complete generated connection with all metadata."""

    source_location_id: UUID
    target_location_id: UUID
    properties: ConnectionProperties
    metadata: dict[str, Any] = field(default_factory=dict)
    generation_timestamp: datetime = field(default_factory=datetime.now)
    embedding_vector: list[float] | None = None
    connection_id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        """Validate generated connection after initialization."""
        if self.source_location_id == self.target_location_id:
            raise ValueError("Source and target locations cannot be the same")


@dataclass
class ConnectionArchetype:
    """Archetype definition for connection types."""

    name: str
    description: str
    typical_properties: ConnectionProperties
    terrain_affinities: dict[str, float]  # terrain -> affinity score
    theme_compatibility: dict[str, float]  # theme -> compatibility score
    generation_templates: dict[str, str]  # template type -> template
    rarity: str = "common"  # common, uncommon, rare

    def __post_init__(self) -> None:
        """Validate archetype after initialization."""
        valid_rarities = ["common", "uncommon", "rare", "legendary"]
        if self.rarity not in valid_rarities:
            raise ValueError(f"Invalid rarity: {self.rarity}")


@dataclass
class ConnectionSearchCriteria:
    """Criteria for searching connections."""

    connection_types: list[str] = field(default_factory=list)
    source_location_themes: list[str] = field(default_factory=list)
    target_location_themes: list[str] = field(default_factory=list)
    min_difficulty: int = 1
    max_difficulty: int = 10
    visibility_types: list[str] = field(default_factory=list)
    requires_features: list[str] = field(default_factory=list)
    exclude_features: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate search criteria after initialization."""
        if not (1 <= self.min_difficulty <= 10):
            raise ValueError("Min difficulty must be between 1 and 10")
        if not (1 <= self.max_difficulty <= 10):
            raise ValueError("Max difficulty must be between 1 and 10")
        if self.min_difficulty > self.max_difficulty:
            raise ValueError("Min difficulty cannot exceed max difficulty")


@dataclass
class ConnectionStorageResult:
    """Result of connection storage operation."""

    success: bool
    connection_id: UUID | None = None
    error_message: str = ""
    storage_time_ms: int = 0
    validation_warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate storage result after initialization."""
        if self.success and not self.connection_id:
            raise ValueError("Successful storage must include connection_id")
        if not self.success and not self.error_message:
            raise ValueError("Failed storage must include error_message")


@dataclass
class ConnectionMetrics:
    """Metrics for connection generation and quality."""

    generation_time_ms: int
    validation_score: float
    consistency_metrics: dict[str, float]
    generation_model: str = "unknown"
    template_version: str = "1.0"
    context_complexity: int = 1

    def __post_init__(self) -> None:
        """Validate metrics after initialization."""
        if self.generation_time_ms < 0:
            raise ValueError("Generation time cannot be negative")
        if not (0.0 <= self.validation_score <= 1.0):
            raise ValueError("Validation score must be between 0.0 and 1.0")


@dataclass
class WorldConnectivityGraph:
    """Represents the world's connectivity structure."""

    nodes: dict[UUID, dict[str, Any]]  # location_id -> location data
    edges: dict[tuple[UUID, UUID], GeneratedConnection]  # connection mapping
    adjacency_list: dict[UUID, list[UUID]] = field(default_factory=dict)
    path_cache: dict[tuple[UUID, UUID], list[UUID]] = field(default_factory=dict)

    def add_connection(self, connection: GeneratedConnection) -> None:
        """Add a connection to the graph."""
        source_id = connection.source_location_id
        target_id = connection.target_location_id

        # Add to edges
        self.edges[(source_id, target_id)] = connection

        # Update adjacency list
        if source_id not in self.adjacency_list:
            self.adjacency_list[source_id] = []
        if target_id not in self.adjacency_list[source_id]:
            self.adjacency_list[source_id].append(target_id)

        # Add reverse connection if reversible
        if connection.properties.reversible:
            self.edges[(target_id, source_id)] = connection
            if target_id not in self.adjacency_list:
                self.adjacency_list[target_id] = []
            if source_id not in self.adjacency_list[target_id]:
                self.adjacency_list[target_id].append(source_id)

        # Clear path cache as it's now invalid
        self.path_cache.clear()

    def get_connections_from(self, location_id: UUID) -> list[GeneratedConnection]:
        """Get all connections from a location."""
        connections = []
        if location_id in self.adjacency_list:
            for target_id in self.adjacency_list[location_id]:
                if (location_id, target_id) in self.edges:
                    connections.append(self.edges[(location_id, target_id)])
        return connections


@dataclass
class ConnectionGenerationRequest:
    """Request for generating a new connection."""

    source_location_id: UUID
    target_location_id: UUID
    purpose: str
    distance_preference: str = "medium"
    override_properties: dict[str, Any] = field(default_factory=dict)
    generation_constraints: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate generation request after initialization."""
        if self.source_location_id == self.target_location_id:
            raise ValueError("Source and target locations cannot be the same")
