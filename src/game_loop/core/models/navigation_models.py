"""
Navigation models for the Game Loop system.

This module defines all data structures used for world boundaries, navigation paths,
and spatial relationships between locations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from uuid import UUID


class ConnectionType(Enum):
    """Types of connections between locations."""

    NORMAL = "normal"
    DOOR = "door"
    PORTAL = "portal"
    HIDDEN = "hidden"
    ONE_WAY = "one_way"
    CONDITIONAL = "conditional"


class BoundaryType(Enum):
    """Types of world boundaries."""

    EDGE = "edge"
    FRONTIER = "frontier"
    INTERNAL = "internal"
    ISOLATED = "isolated"


class NavigationError(Enum):
    """Types of navigation errors."""

    NO_CONNECTION = "no_connection"
    BLOCKED = "blocked"
    MISSING_REQUIREMENT = "missing_requirement"
    INSUFFICIENT_SKILL = "insufficient_skill"
    INVALID_STATE = "invalid_state"
    PATH_NOT_FOUND = "path_not_found"


class PathfindingCriteria(Enum):
    """Criteria for pathfinding optimization."""

    SHORTEST = "shortest"
    SAFEST = "safest"
    SCENIC = "scenic"
    FASTEST = "fastest"


@dataclass
class NavigationResult:
    """Result of a navigation validation."""

    success: bool
    message: str
    error: NavigationError | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class NavigationPath:
    """Represents a path between locations."""

    start_location_id: UUID
    end_location_id: UUID
    path_nodes: list[UUID]
    directions: list[str]
    total_cost: float
    is_valid: bool
    estimated_time: int | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class ExpansionPoint:
    """Point where the world can expand."""

    location_id: UUID
    direction: str
    priority: float
    context: dict[str, Any]
    suggested_theme: str | None = None


@dataclass
class NavigationContext:
    """Context for navigation operations."""

    player_id: UUID
    current_location_id: UUID
    destination_id: UUID | None = None
    preferred_criteria: PathfindingCriteria = PathfindingCriteria.SHORTEST
    avoid_locations: list[UUID] = field(default_factory=list)
    time_constraints: int | None = None


@dataclass
class PathNode:
    """Node in a pathfinding path."""

    location_id: UUID
    g_score: float  # Cost from start
    f_score: float  # Estimated total cost
    parent: Optional["PathNode"] = None
    direction_from_parent: str | None = None
