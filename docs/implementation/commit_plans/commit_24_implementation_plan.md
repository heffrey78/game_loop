# Commit 24: World Boundaries and Navigation Implementation Plan

## Overview
This commit implements the world boundary detection and navigation systems that enable dynamic world expansion while maintaining consistent spatial relationships between locations.

## Scope

### Core Components to Implement

1. **WorldBoundaryManager** (`src/game_loop/core/world/boundary_manager.py`)
   - Boundary detection algorithm
   - Edge location identification
   - Boundary type classification
   - Expansion point determination

2. **LocationConnectionGraph** (`src/game_loop/core/world/connection_graph.py`)
   - Graph data structure for locations
   - Connection management
   - Graph traversal algorithms
   - Connection validation

3. **NavigationValidator** (`src/game_loop/core/navigation/validator.py`)
   - Movement validation
   - Connection checking
   - Path validity verification
   - Navigation rules enforcement

4. **PathfindingService** (`src/game_loop/core/navigation/pathfinder.py`)
   - A* pathfinding implementation
   - Multi-criteria path optimization
   - Path caching system
   - Alternative route generation

5. **NavigationModels** (`src/game_loop/core/models/navigation_models.py`)
   - NavigationPath model
   - ConnectionType enumeration
   - PathfindingResult model
   - NavigationContext model

## Detailed Implementation

### 1. World Boundary Manager

```python
# src/game_loop/core/world/boundary_manager.py

from typing import List, Set, Dict, Optional, Tuple
from uuid import UUID
from enum import Enum
import asyncio

from ...database.models import Location
from ...state.models import WorldState
from ..models.navigation_models import BoundaryType, ExpansionPoint

class BoundaryType(Enum):
    EDGE = "edge"  # Location at world boundary
    FRONTIER = "frontier"  # Location adjacent to unexplored area
    INTERNAL = "internal"  # Location fully surrounded
    ISOLATED = "isolated"  # Location with no connections

class WorldBoundaryManager:
    """Manages world boundaries and identifies expansion points."""
    
    def __init__(self, world_state: WorldState):
        self.world_state = world_state
        self._boundary_cache: Dict[UUID, BoundaryType] = {}
        
    async def detect_boundaries(self) -> Dict[UUID, BoundaryType]:
        """Detect and classify all world boundaries."""
        boundaries = {}
        
        for location_id, location in self.world_state.locations.items():
            boundary_type = await self._classify_location_boundary(location)
            boundaries[location_id] = boundary_type
            self._boundary_cache[location_id] = boundary_type
            
        return boundaries
    
    async def _classify_location_boundary(self, location: Location) -> BoundaryType:
        """Classify a single location's boundary type."""
        if not location.connections:
            return BoundaryType.ISOLATED
            
        # Check connection counts in each direction
        connected_directions = set(location.connections.keys())
        all_directions = {"north", "south", "east", "west", "up", "down"}
        missing_directions = all_directions - connected_directions
        
        if len(missing_directions) >= 4:
            return BoundaryType.EDGE
        elif len(missing_directions) >= 2:
            return BoundaryType.FRONTIER
        else:
            return BoundaryType.INTERNAL
    
    async def find_expansion_points(self) -> List[ExpansionPoint]:
        """Find suitable points for world expansion."""
        expansion_points = []
        boundaries = await self.detect_boundaries()
        
        for location_id, boundary_type in boundaries.items():
            if boundary_type in [BoundaryType.EDGE, BoundaryType.FRONTIER]:
                location = self.world_state.locations[location_id]
                missing_connections = self._get_missing_connections(location)
                
                for direction in missing_connections:
                    expansion_point = ExpansionPoint(
                        location_id=location_id,
                        direction=direction,
                        priority=self._calculate_expansion_priority(location, direction),
                        context=self._gather_expansion_context(location)
                    )
                    expansion_points.append(expansion_point)
        
        return sorted(expansion_points, key=lambda x: x.priority, reverse=True)
    
    def _get_missing_connections(self, location: Location) -> List[str]:
        """Get directions without connections."""
        all_directions = {"north", "south", "east", "west"}
        return list(all_directions - set(location.connections.keys()))
    
    def _calculate_expansion_priority(self, location: Location, direction: str) -> float:
        """Calculate priority for expanding in a given direction."""
        # Higher priority for locations with more player visits
        visit_score = location.state.get("visit_count", 0) * 0.3
        
        # Higher priority for locations with fewer existing connections
        connection_score = (4 - len(location.connections)) * 0.2
        
        # Higher priority for cardinal directions
        direction_score = 0.5 if direction in ["north", "south", "east", "west"] else 0.3
        
        return visit_score + connection_score + direction_score
    
    def _gather_expansion_context(self, location: Location) -> Dict[str, Any]:
        """Gather context for expansion generation."""
        return {
            "location_name": location.name,
            "location_type": location.state.get("type", "generic"),
            "themes": location.state.get("themes", []),
            "description": location.description,
            "existing_connections": list(location.connections.keys())
        }
```

### 2. Location Connection Graph

```python
# src/game_loop/core/world/connection_graph.py

import networkx as nx
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID
from dataclasses import dataclass

from ...state.models import Location
from ..models.navigation_models import ConnectionType

@dataclass
class ConnectionInfo:
    """Information about a connection between locations."""
    from_location: UUID
    to_location: UUID
    direction: str
    connection_type: ConnectionType
    description: Optional[str] = None
    requirements: Optional[Dict[str, Any]] = None

class LocationConnectionGraph:
    """Graph representation of location connections."""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self._connection_cache: Dict[Tuple[UUID, UUID], ConnectionInfo] = {}
        
    def add_location(self, location_id: UUID, location_data: Dict[str, Any]) -> None:
        """Add a location node to the graph."""
        self.graph.add_node(location_id, **location_data)
    
    def add_connection(
        self,
        from_location: UUID,
        to_location: UUID,
        direction: str,
        connection_type: ConnectionType = ConnectionType.NORMAL,
        bidirectional: bool = True,
        **kwargs
    ) -> None:
        """Add a connection between locations."""
        connection_info = ConnectionInfo(
            from_location=from_location,
            to_location=to_location,
            direction=direction,
            connection_type=connection_type,
            **kwargs
        )
        
        # Add edge with connection info
        self.graph.add_edge(
            from_location,
            to_location,
            direction=direction,
            connection_type=connection_type,
            **kwargs
        )
        
        # Cache connection info
        self._connection_cache[(from_location, to_location)] = connection_info
        
        # Add reverse connection if bidirectional
        if bidirectional:
            reverse_direction = self._get_reverse_direction(direction)
            if reverse_direction:
                self.add_connection(
                    to_location,
                    from_location,
                    reverse_direction,
                    connection_type,
                    bidirectional=False,
                    **kwargs
                )
    
    def _get_reverse_direction(self, direction: str) -> Optional[str]:
        """Get the reverse of a direction."""
        reverse_map = {
            "north": "south",
            "south": "north",
            "east": "west",
            "west": "east",
            "up": "down",
            "down": "up",
            "in": "out",
            "out": "in"
        }
        return reverse_map.get(direction)
    
    def get_neighbors(self, location_id: UUID) -> List[Tuple[UUID, str]]:
        """Get all neighboring locations with directions."""
        neighbors = []
        for neighbor_id in self.graph.neighbors(location_id):
            edge_data = self.graph.edges[location_id, neighbor_id]
            neighbors.append((neighbor_id, edge_data["direction"]))
        return neighbors
    
    def has_connection(self, from_location: UUID, to_location: UUID) -> bool:
        """Check if a direct connection exists."""
        return self.graph.has_edge(from_location, to_location)
    
    def get_connection_info(
        self, from_location: UUID, to_location: UUID
    ) -> Optional[ConnectionInfo]:
        """Get detailed connection information."""
        return self._connection_cache.get((from_location, to_location))
    
    def find_connected_components(self) -> List[Set[UUID]]:
        """Find all connected components in the graph."""
        undirected = self.graph.to_undirected()
        return [set(comp) for comp in nx.connected_components(undirected)]
    
    def get_subgraph(self, location_ids: Set[UUID]) -> nx.DiGraph:
        """Get a subgraph containing only specified locations."""
        return self.graph.subgraph(location_ids)
```

### 3. Navigation Validator

```python
# src/game_loop/core/navigation/validator.py

from typing import Optional, Dict, Any
from uuid import UUID

from ...state.models import Location, PlayerState
from ..world.connection_graph import LocationConnectionGraph
from ..models.navigation_models import NavigationResult, NavigationError

class NavigationValidator:
    """Validates navigation actions and movements."""
    
    def __init__(self, connection_graph: LocationConnectionGraph):
        self.connection_graph = connection_graph
        
    async def validate_movement(
        self,
        player_state: PlayerState,
        from_location: Location,
        to_location_id: UUID,
        direction: str
    ) -> NavigationResult:
        """Validate if a movement is allowed."""
        # Check if connection exists
        if not self.connection_graph.has_connection(
            from_location.location_id, to_location_id
        ):
            return NavigationResult(
                success=False,
                error=NavigationError.NO_CONNECTION,
                message=f"No connection exists to the {direction}."
            )
        
        # Get connection info
        connection_info = self.connection_graph.get_connection_info(
            from_location.location_id, to_location_id
        )
        
        # Check requirements
        if connection_info and connection_info.requirements:
            validation_result = await self._check_requirements(
                player_state, connection_info.requirements
            )
            if not validation_result.success:
                return validation_result
        
        # Check if the connection is blocked
        if from_location.state.get(f"blocked_{direction}", False):
            return NavigationResult(
                success=False,
                error=NavigationError.BLOCKED,
                message=f"The way {direction} is blocked."
            )
        
        return NavigationResult(
            success=True,
            message=f"You can go {direction}."
        )
    
    async def _check_requirements(
        self,
        player_state: PlayerState,
        requirements: Dict[str, Any]
    ) -> NavigationResult:
        """Check if player meets movement requirements."""
        # Check item requirements
        if "required_items" in requirements:
            for item_name in requirements["required_items"]:
                if not any(item.name == item_name for item in player_state.inventory):
                    return NavigationResult(
                        success=False,
                        error=NavigationError.MISSING_REQUIREMENT,
                        message=f"You need a {item_name} to go this way."
                    )
        
        # Check skill requirements
        if "required_skills" in requirements:
            player_skills = player_state.stats.skills if player_state.stats else {}
            for skill, min_level in requirements["required_skills"].items():
                if player_skills.get(skill, 0) < min_level:
                    return NavigationResult(
                        success=False,
                        error=NavigationError.INSUFFICIENT_SKILL,
                        message=f"Your {skill} skill is too low."
                    )
        
        # Check state requirements
        if "required_state" in requirements:
            for key, value in requirements["required_state"].items():
                if player_state.state.get(key) != value:
                    return NavigationResult(
                        success=False,
                        error=NavigationError.INVALID_STATE,
                        message="You're not in the right state for this action."
                    )
        
        return NavigationResult(success=True)
    
    def get_valid_directions(
        self, location: Location, player_state: Optional[PlayerState] = None
    ) -> Dict[str, bool]:
        """Get all directions and their validity from a location."""
        valid_directions = {}
        
        for direction, destination_id in location.connections.items():
            if player_state:
                # Full validation with player state
                result = asyncio.run(self.validate_movement(
                    player_state, location, destination_id, direction
                ))
                valid_directions[direction] = result.success
            else:
                # Simple check for connection existence
                valid_directions[direction] = not location.state.get(
                    f"blocked_{direction}", False
                )
        
        return valid_directions
```

### 4. Pathfinding Service

```python
# src/game_loop/core/navigation/pathfinder.py

from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID
import heapq
from dataclasses import dataclass
from collections import defaultdict

from ...state.models import WorldState, PlayerState
from ..world.connection_graph import LocationConnectionGraph
from ..models.navigation_models import NavigationPath, PathNode, PathfindingCriteria

@dataclass
class PathNode:
    location_id: UUID
    g_score: float  # Cost from start
    f_score: float  # Estimated total cost
    parent: Optional['PathNode'] = None
    direction_from_parent: Optional[str] = None

class PathfindingService:
    """A* pathfinding implementation for location navigation."""
    
    def __init__(
        self,
        world_state: WorldState,
        connection_graph: LocationConnectionGraph
    ):
        self.world_state = world_state
        self.connection_graph = connection_graph
        self._path_cache: Dict[Tuple[UUID, UUID], NavigationPath] = {}
        
    async def find_path(
        self,
        start_location_id: UUID,
        end_location_id: UUID,
        player_state: Optional[PlayerState] = None,
        criteria: PathfindingCriteria = PathfindingCriteria.SHORTEST
    ) -> Optional[NavigationPath]:
        """Find optimal path between two locations."""
        # Check cache first
        cache_key = (start_location_id, end_location_id)
        if cache_key in self._path_cache:
            cached_path = self._path_cache[cache_key]
            # Validate cached path is still valid
            if await self._is_path_valid(cached_path, player_state):
                return cached_path
        
        # Run A* algorithm
        path = await self._astar_search(
            start_location_id,
            end_location_id,
            player_state,
            criteria
        )
        
        if path:
            self._path_cache[cache_key] = path
            
        return path
    
    async def _astar_search(
        self,
        start_id: UUID,
        goal_id: UUID,
        player_state: Optional[PlayerState],
        criteria: PathfindingCriteria
    ) -> Optional[NavigationPath]:
        """A* pathfinding algorithm implementation."""
        # Initialize open set with start node
        start_node = PathNode(
            location_id=start_id,
            g_score=0,
            f_score=self._heuristic(start_id, goal_id)
        )
        
        open_set = [(start_node.f_score, id(start_node), start_node)]
        closed_set = set()
        g_scores = {start_id: 0}
        
        while open_set:
            _, _, current = heapq.heappop(open_set)
            
            if current.location_id == goal_id:
                # Reconstruct path
                return self._reconstruct_path(current)
            
            if current.location_id in closed_set:
                continue
                
            closed_set.add(current.location_id)
            
            # Explore neighbors
            for neighbor_id, direction in self.connection_graph.get_neighbors(
                current.location_id
            ):
                if neighbor_id in closed_set:
                    continue
                
                # Calculate tentative g_score
                edge_cost = await self._calculate_edge_cost(
                    current.location_id,
                    neighbor_id,
                    direction,
                    player_state,
                    criteria
                )
                
                if edge_cost is None:  # Connection not traversable
                    continue
                    
                tentative_g = current.g_score + edge_cost
                
                if neighbor_id not in g_scores or tentative_g < g_scores[neighbor_id]:
                    g_scores[neighbor_id] = tentative_g
                    
                    neighbor_node = PathNode(
                        location_id=neighbor_id,
                        g_score=tentative_g,
                        f_score=tentative_g + self._heuristic(neighbor_id, goal_id),
                        parent=current,
                        direction_from_parent=direction
                    )
                    
                    heapq.heappush(
                        open_set,
                        (neighbor_node.f_score, id(neighbor_node), neighbor_node)
                    )
        
        return None  # No path found
    
    def _heuristic(self, from_id: UUID, to_id: UUID) -> float:
        """Heuristic function for A* (Manhattan distance in grid)."""
        # For now, use connection count as simple heuristic
        # In a full implementation, this would use spatial coordinates
        return len(nx.shortest_path(self.connection_graph.graph, from_id, to_id)) - 1
    
    async def _calculate_edge_cost(
        self,
        from_id: UUID,
        to_id: UUID,
        direction: str,
        player_state: Optional[PlayerState],
        criteria: PathfindingCriteria
    ) -> Optional[float]:
        """Calculate cost of traversing an edge."""
        connection_info = self.connection_graph.get_connection_info(from_id, to_id)
        if not connection_info:
            return None
            
        base_cost = 1.0
        
        # Adjust cost based on criteria
        if criteria == PathfindingCriteria.SHORTEST:
            # Default base cost
            pass
        elif criteria == PathfindingCriteria.SAFEST:
            # Increase cost for dangerous areas
            location = self.world_state.locations[to_id]
            danger_level = location.state.get("danger_level", 0)
            base_cost += danger_level * 2
        elif criteria == PathfindingCriteria.SCENIC:
            # Decrease cost for interesting areas
            location = self.world_state.locations[to_id]
            interest_level = location.state.get("interest_level", 0)
            base_cost -= interest_level * 0.5
            
        # Check if player can traverse this connection
        if player_state and connection_info.requirements:
            # Simplified check - in full implementation would use NavigationValidator
            if not self._meets_requirements(player_state, connection_info.requirements):
                return None
                
        return max(0.1, base_cost)  # Ensure positive cost
    
    def _meets_requirements(
        self,
        player_state: PlayerState,
        requirements: Dict[str, Any]
    ) -> bool:
        """Simple requirement check for pathfinding."""
        # This is a simplified version - full implementation would use NavigationValidator
        if "required_items" in requirements:
            for item_name in requirements["required_items"]:
                if not any(item.name == item_name for item in player_state.inventory):
                    return False
        return True
    
    def _reconstruct_path(self, end_node: PathNode) -> NavigationPath:
        """Reconstruct path from end node."""
        path_nodes = []
        directions = []
        current = end_node
        
        while current:
            path_nodes.append(current.location_id)
            if current.direction_from_parent:
                directions.append(current.direction_from_parent)
            current = current.parent
        
        path_nodes.reverse()
        directions.reverse()
        
        return NavigationPath(
            start_location_id=path_nodes[0],
            end_location_id=path_nodes[-1],
            path_nodes=path_nodes,
            directions=directions,
            total_cost=end_node.g_score,
            is_valid=True
        )
    
    async def _is_path_valid(
        self,
        path: NavigationPath,
        player_state: Optional[PlayerState]
    ) -> bool:
        """Check if a cached path is still valid."""
        for i in range(len(path.path_nodes) - 1):
            from_id = path.path_nodes[i]
            to_id = path.path_nodes[i + 1]
            
            if not self.connection_graph.has_connection(from_id, to_id):
                return False
                
            # Check if connection is still traversable
            if player_state:
                cost = await self._calculate_edge_cost(
                    from_id, to_id, path.directions[i],
                    player_state, PathfindingCriteria.SHORTEST
                )
                if cost is None:
                    return False
                    
        return True
    
    async def find_alternative_paths(
        self,
        start_location_id: UUID,
        end_location_id: UUID,
        player_state: Optional[PlayerState] = None,
        max_alternatives: int = 3
    ) -> List[NavigationPath]:
        """Find alternative paths between locations."""
        alternatives = []
        used_edges = set()
        
        for _ in range(max_alternatives):
            # Temporarily increase cost of used edges
            for edge in used_edges:
                # Modify graph weights temporarily
                pass
                
            path = await self.find_path(
                start_location_id,
                end_location_id,
                player_state,
                PathfindingCriteria.SHORTEST
            )
            
            if path and path not in alternatives:
                alternatives.append(path)
                # Add path edges to used set
                for i in range(len(path.path_nodes) - 1):
                    used_edges.add((path.path_nodes[i], path.path_nodes[i + 1]))
            
            # Restore original weights
            for edge in used_edges:
                # Reset graph weights
                pass
                
        return alternatives
```

### 5. Navigation Models

```python
# src/game_loop/core/models/navigation_models.py

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
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
    error: Optional[NavigationError] = None
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NavigationPath:
    """Represents a path between locations."""
    start_location_id: UUID
    end_location_id: UUID
    path_nodes: List[UUID]
    directions: List[str]
    total_cost: float
    is_valid: bool
    estimated_time: Optional[int] = None
    warnings: List[str] = field(default_factory=list)

@dataclass
class ExpansionPoint:
    """Point where the world can expand."""
    location_id: UUID
    direction: str
    priority: float
    context: Dict[str, Any]
    suggested_theme: Optional[str] = None

@dataclass
class NavigationContext:
    """Context for navigation operations."""
    player_id: UUID
    current_location_id: UUID
    destination_id: Optional[UUID] = None
    preferred_criteria: PathfindingCriteria = PathfindingCriteria.SHORTEST
    avoid_locations: List[UUID] = field(default_factory=list)
    time_constraints: Optional[int] = None
```

### 6. Database Migration

```sql
-- src/game_loop/database/migrations/026_navigation_system.sql

-- Add navigation-related columns to locations table
ALTER TABLE locations
ADD COLUMN IF NOT EXISTS boundary_type VARCHAR(20) DEFAULT 'internal',
ADD COLUMN IF NOT EXISTS expansion_priority FLOAT DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS navigation_metadata JSONB DEFAULT '{}';

-- Create navigation paths table for caching
CREATE TABLE IF NOT EXISTS navigation_paths (
    path_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    start_location_id UUID NOT NULL REFERENCES locations(location_id),
    end_location_id UUID NOT NULL REFERENCES locations(location_id),
    path_data JSONB NOT NULL,
    total_cost FLOAT NOT NULL,
    criteria VARCHAR(20) NOT NULL,
    is_valid BOOLEAN DEFAULT true,
    last_validated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT unique_path UNIQUE(start_location_id, end_location_id, criteria)
);

-- Create connection requirements table
CREATE TABLE IF NOT EXISTS connection_requirements (
    requirement_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    from_location_id UUID NOT NULL REFERENCES locations(location_id),
    to_location_id UUID NOT NULL REFERENCES locations(location_id),
    direction VARCHAR(20) NOT NULL,
    requirements JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT unique_connection_requirement 
        UNIQUE(from_location_id, to_location_id)
);

-- Create indexes for efficient queries
CREATE INDEX idx_navigation_paths_locations 
    ON navigation_paths(start_location_id, end_location_id);
CREATE INDEX idx_connection_requirements_from 
    ON connection_requirements(from_location_id);
CREATE INDEX idx_locations_boundary_type 
    ON locations(boundary_type);
```

### 7. Integration with Game Loop

```python
# Updates to src/game_loop/core/game_loop.py

# Add to __init__
from .world.boundary_manager import WorldBoundaryManager
from .world.connection_graph import LocationConnectionGraph
from .navigation.validator import NavigationValidator
from .navigation.pathfinder import PathfindingService

# In GameLoop.__init__
self.connection_graph = LocationConnectionGraph()
self.boundary_manager = WorldBoundaryManager(self.state_manager.world_tracker.get_state())
self.navigation_validator = NavigationValidator(self.connection_graph)
self.pathfinding_service = PathfindingService(
    self.state_manager.world_tracker.get_state(),
    self.connection_graph
)

# Add method to initialize navigation graph
async def _initialize_navigation_graph(self):
    """Initialize the navigation graph from world state."""
    world_state = self.state_manager.world_tracker.get_state()
    
    # Add all locations to graph
    for location_id, location in world_state.locations.items():
        self.connection_graph.add_location(
            location_id,
            {
                "name": location.name,
                "type": location.state.get("type", "generic")
            }
        )
    
    # Add all connections
    for location_id, location in world_state.locations.items():
        for direction, destination_id in location.connections.items():
            self.connection_graph.add_connection(
                location_id,
                destination_id,
                direction
            )

# Update movement validation to use NavigationValidator
async def _handle_movement(self, direction: str, player_state, current_location):
    # ... existing code ...
    
    # Use navigation validator
    validation_result = await self.navigation_validator.validate_movement(
        player_state,
        current_location,
        destination_id,
        direction
    )
    
    if not validation_result.success:
        return ActionResult(
            success=False,
            feedback_message=validation_result.message
        )
    
    # ... rest of movement handling ...
```

### 8. Testing

```python
# tests/unit/core/world/test_boundary_manager.py

import pytest
from uuid import uuid4
from game_loop.core.world.boundary_manager import WorldBoundaryManager, BoundaryType
from game_loop.state.models import WorldState, Location

@pytest.mark.asyncio
async def test_boundary_detection():
    """Test boundary type detection."""
    # Create test world
    loc1_id = uuid4()
    loc2_id = uuid4()
    loc3_id = uuid4()
    
    locations = {
        loc1_id: Location(
            location_id=loc1_id,
            name="Edge Location",
            connections={"north": loc2_id}  # Only one connection
        ),
        loc2_id: Location(
            location_id=loc2_id,
            name="Frontier Location",
            connections={"south": loc1_id, "north": loc3_id}  # Two connections
        ),
        loc3_id: Location(
            location_id=loc3_id,
            name="Internal Location",
            connections={
                "south": loc2_id,
                "north": uuid4(),
                "east": uuid4(),
                "west": uuid4()
            }  # Four connections
        )
    }
    
    world_state = WorldState(locations=locations)
    manager = WorldBoundaryManager(world_state)
    
    boundaries = await manager.detect_boundaries()
    
    assert boundaries[loc1_id] == BoundaryType.EDGE
    assert boundaries[loc2_id] == BoundaryType.FRONTIER
    assert boundaries[loc3_id] == BoundaryType.INTERNAL

@pytest.mark.asyncio
async def test_expansion_points():
    """Test finding expansion points."""
    # ... test implementation ...

# tests/unit/core/navigation/test_pathfinder.py

@pytest.mark.asyncio
async def test_pathfinding():
    """Test A* pathfinding."""
    # ... test implementation ...

@pytest.mark.asyncio
async def test_path_validation():
    """Test path validation with requirements."""
    # ... test implementation ...
```

## Implementation Order

1. **Navigation Models** - Define all data structures
2. **Connection Graph** - Implement graph structure
3. **Boundary Manager** - Implement boundary detection
4. **Navigation Validator** - Implement movement validation
5. **Pathfinding Service** - Implement A* algorithm
6. **Database Migration** - Create schema updates
7. **Game Loop Integration** - Wire everything together
8. **Testing** - Comprehensive test coverage

## Verification Steps

1. **Boundary Detection**
   - Create a test world with various connection patterns
   - Verify correct boundary classification
   - Test expansion point generation

2. **Graph Operations**
   - Test adding/removing connections
   - Verify bidirectional connections work
   - Test connected component detection

3. **Movement Validation**
   - Test valid and invalid movements
   - Verify requirement checking
   - Test blocked connections

4. **Pathfinding**
   - Test paths between various locations
   - Verify optimal path selection
   - Test with movement restrictions
   - Verify alternative path generation

5. **Integration**
   - Test movement commands in game
   - Verify pathfinding commands work
   - Test boundary detection updates

## Success Criteria

- [ ] Boundary detection correctly identifies edge/frontier/internal locations
- [ ] Connection graph maintains accurate spatial relationships
- [ ] Movement validation enforces all navigation rules
- [ ] Pathfinding returns optimal paths based on criteria
- [ ] Path caching improves performance
- [ ] All navigation components integrate seamlessly with game loop
- [ ] Comprehensive test coverage (>90%) for navigation components
- [ ] Performance benchmarks show <100ms for pathfinding in 1000-location world