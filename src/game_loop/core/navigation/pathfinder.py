"""
Pathfinding service with A* algorithm implementation.

This module provides pathfinding capabilities for navigation between locations.
"""

import heapq
import logging
from typing import Any
from uuid import UUID

import networkx as nx

from ...state.models import PlayerState, WorldState
from ..models.navigation_models import NavigationPath, PathfindingCriteria, PathNode
from ..world.connection_graph import LocationConnectionGraph

logger = logging.getLogger(__name__)


class PathfindingService:
    """A* pathfinding implementation for location navigation."""

    def __init__(
        self, world_state: WorldState, connection_graph: LocationConnectionGraph
    ):
        self.world_state = world_state
        self.connection_graph = connection_graph
        self._path_cache: dict[tuple[UUID, UUID], NavigationPath] = {}

    async def find_path(
        self,
        start_location_id: UUID,
        end_location_id: UUID,
        player_state: PlayerState | None = None,
        criteria: PathfindingCriteria = PathfindingCriteria.SHORTEST,
    ) -> NavigationPath | None:
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
            start_location_id, end_location_id, player_state, criteria
        )

        if path:
            self._path_cache[cache_key] = path

        return path

    async def _astar_search(
        self,
        start_id: UUID,
        goal_id: UUID,
        player_state: PlayerState | None,
        criteria: PathfindingCriteria,
    ) -> NavigationPath | None:
        """A* pathfinding algorithm implementation."""
        # Initialize open set with start node
        start_node = PathNode(
            location_id=start_id, g_score=0, f_score=self._heuristic(start_id, goal_id)
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
                    current.location_id, neighbor_id, direction, player_state, criteria
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
                        direction_from_parent=direction,
                    )

                    heapq.heappush(
                        open_set,
                        (neighbor_node.f_score, id(neighbor_node), neighbor_node),
                    )

        return None  # No path found

    def _heuristic(self, from_id: UUID, to_id: UUID) -> float:
        """Heuristic function for A* (estimated distance)."""
        # Simple heuristic: try to get shortest path length as estimate
        try:
            if self.connection_graph.is_connected(from_id, to_id):
                path_length = self.connection_graph.shortest_path_length(from_id, to_id)
                return float(path_length) if path_length is not None else 0.0
            else:
                return float("inf")  # No connection possible
        except Exception:
            # Fallback to simple heuristic
            return 1.0

    async def _calculate_edge_cost(
        self,
        from_id: UUID,
        to_id: UUID,
        direction: str,
        player_state: PlayerState | None,
        criteria: PathfindingCriteria,
    ) -> float | None:
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
            if to_id in self.world_state.locations:
                location = self.world_state.locations[to_id]
                danger_level = location.state_flags.get("danger_level", 0)
                base_cost += danger_level * 2
        elif criteria == PathfindingCriteria.SCENIC:
            # Decrease cost for interesting areas
            if to_id in self.world_state.locations:
                location = self.world_state.locations[to_id]
                interest_level = location.state_flags.get("interest_level", 0)
                base_cost -= interest_level * 0.5
        elif criteria == PathfindingCriteria.FASTEST:
            # Consider movement speed modifiers
            if to_id in self.world_state.locations:
                location = self.world_state.locations[to_id]
                speed_modifier = location.state_flags.get("speed_modifier", 1.0)
                base_cost /= speed_modifier

        # Check if player can traverse this connection
        if player_state and connection_info.requirements:
            # Simplified check - in full implementation would use NavigationValidator
            if not self._meets_requirements(player_state, connection_info.requirements):
                return None

        return max(0.1, base_cost)  # Ensure positive cost

    def _meets_requirements(
        self, player_state: PlayerState, requirements: dict[str, Any]
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
            is_valid=True,
        )

    async def _is_path_valid(
        self, path: NavigationPath, player_state: PlayerState | None
    ) -> bool:
        """Check if a cached path is still valid."""
        for i in range(len(path.path_nodes) - 1):
            from_id = path.path_nodes[i]
            to_id = path.path_nodes[i + 1]

            if not self.connection_graph.has_connection(from_id, to_id):
                return False

            # Check if connection is still traversable
            if player_state and i < len(path.directions):
                cost = await self._calculate_edge_cost(
                    from_id,
                    to_id,
                    path.directions[i],
                    player_state,
                    PathfindingCriteria.SHORTEST,
                )
                if cost is None:
                    return False

        return True

    async def find_alternative_paths(
        self,
        start_location_id: UUID,
        end_location_id: UUID,
        player_state: PlayerState | None = None,
        max_alternatives: int = 3,
    ) -> list[NavigationPath]:
        """Find alternative paths between locations."""
        alternatives = []

        # For now, just return the primary path as the only alternative
        # A full implementation would temporarily modify edge weights to find different paths
        primary_path = await self.find_path(
            start_location_id,
            end_location_id,
            player_state,
            PathfindingCriteria.SHORTEST,
        )

        if primary_path:
            alternatives.append(primary_path)

        # Try with different criteria for variety
        if len(alternatives) < max_alternatives:
            for criteria in [PathfindingCriteria.SAFEST, PathfindingCriteria.SCENIC]:
                alt_path = await self.find_path(
                    start_location_id, end_location_id, player_state, criteria
                )
                if alt_path and alt_path not in alternatives:
                    alternatives.append(alt_path)
                    if len(alternatives) >= max_alternatives:
                        break

        return alternatives

    def clear_cache(self) -> None:
        """Clear the path cache."""
        self._path_cache.clear()

    def get_cache_size(self) -> int:
        """Get the current cache size."""
        return len(self._path_cache)

    def find_nearest_locations(
        self, center_location_id: UUID, max_distance: int = 3, max_results: int = 10
    ) -> list[tuple[UUID, int]]:
        """Find locations within a certain distance of a center location."""
        try:
            # Use NetworkX to find all paths within max_distance
            graph = self.connection_graph.graph.to_undirected()
            if center_location_id not in graph:
                return []

            nearby = []
            for location_id in graph.nodes():
                if location_id != center_location_id:
                    try:
                        distance = nx.shortest_path_length(
                            graph, center_location_id, location_id
                        )
                        if distance <= max_distance:
                            nearby.append((location_id, distance))
                    except nx.NetworkXNoPath:
                        continue

            # Sort by distance and limit results
            nearby.sort(key=lambda x: x[1])
            return nearby[:max_results]

        except Exception as e:
            logger.error(f"Error finding nearby locations: {e}")
            return []
