"""
Location connection graph for spatial relationships.

This module provides graph-based management of location connections using NetworkX.
"""

import logging
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import networkx as nx

from ..models.navigation_models import ConnectionType

logger = logging.getLogger(__name__)


@dataclass
class ConnectionInfo:
    """Information about a connection between locations."""

    from_location: UUID
    to_location: UUID
    direction: str
    connection_type: ConnectionType
    description: str | None = None
    requirements: dict[str, Any] | None = None


class LocationConnectionGraph:
    """Graph representation of location connections."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self._connection_cache: dict[tuple[UUID, UUID], ConnectionInfo] = {}

    def add_location(self, location_id: UUID, location_data: dict[str, Any]) -> None:
        """Add a location node to the graph."""
        self.graph.add_node(location_id, **location_data)

    def add_connection(
        self,
        from_location: UUID,
        to_location: UUID,
        direction: str,
        connection_type: ConnectionType = ConnectionType.NORMAL,
        bidirectional: bool = True,
        **kwargs,
    ) -> None:
        """Add a connection between locations."""
        connection_info = ConnectionInfo(
            from_location=from_location,
            to_location=to_location,
            direction=direction,
            connection_type=connection_type,
            **kwargs,
        )

        # Add edge with connection info
        self.graph.add_edge(
            from_location,
            to_location,
            direction=direction,
            connection_type=connection_type,
            **kwargs,
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
                    **kwargs,
                )

    def _get_reverse_direction(self, direction: str) -> str | None:
        """Get the reverse of a direction."""
        reverse_map = {
            "north": "south",
            "south": "north",
            "east": "west",
            "west": "east",
            "up": "down",
            "down": "up",
            "in": "out",
            "out": "in",
        }
        return reverse_map.get(direction)

    def get_neighbors(self, location_id: UUID) -> list[tuple[UUID, str]]:
        """Get all neighboring locations with directions."""
        neighbors = []
        try:
            for neighbor_id in self.graph.neighbors(location_id):
                edge_data = self.graph.edges[location_id, neighbor_id]
                neighbors.append((neighbor_id, edge_data["direction"]))
        except KeyError as e:
            logger.warning(f"Error getting neighbors for {location_id}: {e}")
        return neighbors

    def has_connection(self, from_location: UUID, to_location: UUID) -> bool:
        """Check if a direct connection exists."""
        return self.graph.has_edge(from_location, to_location)

    def get_connection_info(
        self, from_location: UUID, to_location: UUID
    ) -> ConnectionInfo | None:
        """Get detailed connection information."""
        return self._connection_cache.get((from_location, to_location))

    def find_connected_components(self) -> list[set[UUID]]:
        """Find all connected components in the graph."""
        undirected = self.graph.to_undirected()
        return [set(comp) for comp in nx.connected_components(undirected)]

    def get_subgraph(self, location_ids: set[UUID]) -> nx.DiGraph:
        """Get a subgraph containing only specified locations."""
        return self.graph.subgraph(location_ids)

    def remove_connection(self, from_location: UUID, to_location: UUID) -> bool:
        """Remove a connection between locations."""
        try:
            if self.graph.has_edge(from_location, to_location):
                self.graph.remove_edge(from_location, to_location)
                # Remove from cache
                cache_key = (from_location, to_location)
                if cache_key in self._connection_cache:
                    del self._connection_cache[cache_key]
                return True
        except Exception as e:
            logger.error(f"Error removing connection: {e}")
        return False

    def get_all_connections(self) -> list[ConnectionInfo]:
        """Get all connection information."""
        return list(self._connection_cache.values())

    def clear(self) -> None:
        """Clear all locations and connections."""
        self.graph.clear()
        self._connection_cache.clear()

    def get_location_count(self) -> int:
        """Get the number of locations in the graph."""
        return self.graph.number_of_nodes()

    def get_connection_count(self) -> int:
        """Get the number of connections in the graph."""
        return self.graph.number_of_edges()

    def is_connected(self, location_a: UUID, location_b: UUID) -> bool:
        """Check if two locations are connected by any path."""
        try:
            return nx.has_path(self.graph.to_undirected(), location_a, location_b)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return False

    def shortest_path_length(self, location_a: UUID, location_b: UUID) -> int | None:
        """Get the shortest path length between two locations."""
        try:
            return nx.shortest_path_length(
                self.graph.to_undirected(), location_a, location_b
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
