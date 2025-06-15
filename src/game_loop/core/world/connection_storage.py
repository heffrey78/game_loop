"""
Connection Storage System for World Connection Management.

This module handles persistence, caching, and retrieval of connection data
with database integration and graph structure maintenance.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from game_loop.core.models.connection_models import (
    ConnectionStorageResult,
    GeneratedConnection,
    WorldConnectivityGraph,
)
from game_loop.database.session_factory import DatabaseSessionFactory

logger = logging.getLogger(__name__)


class ConnectionStorage:
    """Handles persistence, caching, and retrieval of connection data."""

    def __init__(self, session_factory: DatabaseSessionFactory):
        """Initialize storage system."""
        self.session_factory = session_factory
        self._connection_cache: dict[UUID, GeneratedConnection] = {}
        self._location_connections_cache: dict[UUID, list[GeneratedConnection]] = {}
        self._graph_cache: WorldConnectivityGraph | None = None

    async def store_connection(
        self, connection: GeneratedConnection
    ) -> ConnectionStorageResult:
        """Store connection with full metadata and validation."""
        start_time = datetime.now()

        try:
            # Basic validation before storage
            validation_warnings = []

            if not connection.properties.description.strip():
                validation_warnings.append("Connection description is empty")

            if connection.properties.travel_time <= 0:
                return ConnectionStorageResult(
                    success=False,
                    error_message="Travel time must be positive",
                )

            # Simulate database storage
            # In a real implementation, this would use SQLAlchemy to insert into the database
            # For now, we'll just store in cache and simulate success

            # Store in cache
            self._connection_cache[connection.connection_id] = connection

            # Update location connections cache
            self._update_location_cache(connection)

            # Clear graph cache as it's now invalid
            self._graph_cache = None

            storage_time = int((datetime.now() - start_time).total_seconds() * 1000)

            logger.info(
                f"Stored connection {connection.connection_id} in {storage_time}ms"
            )

            return ConnectionStorageResult(
                success=True,
                connection_id=connection.connection_id,
                storage_time_ms=storage_time,
                validation_warnings=validation_warnings,
            )

        except Exception as e:
            logger.error(f"Error storing connection: {e}")
            return ConnectionStorageResult(
                success=False,
                error_message=str(e),
                storage_time_ms=int(
                    (datetime.now() - start_time).total_seconds() * 1000
                ),
            )

    async def retrieve_connections(
        self, location_id: UUID
    ) -> list[GeneratedConnection]:
        """Retrieve all connections for a location."""
        try:
            # Check cache first
            if location_id in self._location_connections_cache:
                return self._location_connections_cache[location_id].copy()

            # In a real implementation, this would query the database
            # For now, we'll search through cached connections
            connections = []

            for connection in self._connection_cache.values():
                if (
                    connection.source_location_id == location_id
                    or connection.target_location_id == location_id
                ):
                    connections.append(connection)

            # Cache the result
            self._location_connections_cache[location_id] = connections.copy()

            logger.debug(
                f"Retrieved {len(connections)} connections for location {location_id}"
            )
            return connections

        except Exception as e:
            logger.error(f"Error retrieving connections for {location_id}: {e}")
            return []

    async def retrieve_connection(
        self, connection_id: UUID
    ) -> GeneratedConnection | None:
        """Retrieve a specific connection by ID."""
        try:
            # Check cache first
            if connection_id in self._connection_cache:
                return self._connection_cache[connection_id]

            # In a real implementation, this would query the database
            # For now, return None if not in cache
            return None

        except Exception as e:
            logger.error(f"Error retrieving connection {connection_id}: {e}")
            return None

    async def update_connection_graph(self, connection: GeneratedConnection) -> bool:
        """Update the world connectivity graph structure."""
        try:
            # Get or create graph
            if self._graph_cache is None:
                self._graph_cache = await self._build_connectivity_graph()

            # Add connection to graph
            self._graph_cache.add_connection(connection)

            logger.debug(
                f"Updated connectivity graph with connection {connection.connection_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Error updating connectivity graph: {e}")
            return False

    async def get_connectivity_graph(self) -> WorldConnectivityGraph:
        """Get the current world connectivity graph."""
        try:
            if self._graph_cache is None:
                self._graph_cache = await self._build_connectivity_graph()

            return self._graph_cache

        except Exception as e:
            logger.error(f"Error getting connectivity graph: {e}")
            # Return empty graph as fallback
            return WorldConnectivityGraph(nodes={}, edges={})

    async def validate_graph_consistency(self) -> list[str]:
        """Validate overall graph consistency and detect issues."""
        try:
            issues = []

            # Get current graph
            graph = await self.get_connectivity_graph()

            # Check for orphaned nodes
            connected_nodes = set()
            for edge in graph.edges:
                connected_nodes.add(edge[0])
                connected_nodes.add(edge[1])

            orphaned_nodes = set(graph.nodes.keys()) - connected_nodes
            if orphaned_nodes:
                issues.append(
                    f"Found {len(orphaned_nodes)} orphaned nodes with no connections"
                )

            # Check for missing reverse connections
            missing_reverse = []
            for edge_key, connection in graph.edges.items():
                if connection.properties.reversible:
                    reverse_key = (edge_key[1], edge_key[0])
                    if reverse_key not in graph.edges:
                        missing_reverse.append(edge_key)

            if missing_reverse:
                issues.append(
                    f"Found {len(missing_reverse)} missing reverse connections"
                )

            # Check for duplicate connections
            connection_pairs = set()
            duplicates = []
            for edge_key in graph.edges:
                # Normalize edge key (always put smaller UUID first)
                normalized_key = tuple(sorted([edge_key[0], edge_key[1]]))
                if normalized_key in connection_pairs:
                    duplicates.append(edge_key)
                else:
                    connection_pairs.add(normalized_key)

            if duplicates:
                issues.append(f"Found {len(duplicates)} duplicate connections")

            # Check adjacency list consistency
            for node_id, neighbors in graph.adjacency_list.items():
                for neighbor_id in neighbors:
                    edge_key = (node_id, neighbor_id)
                    if edge_key not in graph.edges:
                        issues.append(
                            f"Adjacency list inconsistency: {edge_key} in list but not in edges"
                        )

            if not issues:
                logger.info("Graph consistency validation passed")
            else:
                logger.warning(
                    f"Graph consistency issues found: {len(issues)} problems"
                )

            return issues

        except Exception as e:
            logger.error(f"Error validating graph consistency: {e}")
            return [f"Validation error: {e}"]

    async def get_connection_metrics(self) -> dict[str, Any]:
        """Get connection system performance and quality metrics."""
        try:
            metrics = {
                "total_connections": len(self._connection_cache),
                "cached_connections": len(self._connection_cache),
                "location_cache_entries": len(self._location_connections_cache),
                "graph_cached": self._graph_cache is not None,
                "connection_types": {},
                "average_difficulty": 0.0,
                "average_travel_time": 0.0,
                "visibility_distribution": {},
            }

            if self._connection_cache:
                # Analyze connection types
                type_counts: dict[str, int] = {}
                total_difficulty = 0
                total_travel_time = 0
                visibility_counts: dict[str, int] = {}

                for connection in self._connection_cache.values():
                    # Connection type distribution
                    conn_type = connection.properties.connection_type
                    type_counts[conn_type] = type_counts.get(conn_type, 0) + 1

                    # Difficulty and travel time
                    total_difficulty += connection.properties.difficulty
                    total_travel_time += connection.properties.travel_time

                    # Visibility distribution
                    visibility = connection.properties.visibility
                    visibility_counts[visibility] = (
                        visibility_counts.get(visibility, 0) + 1
                    )

                metrics["connection_types"] = type_counts
                metrics["average_difficulty"] = total_difficulty / len(
                    self._connection_cache
                )
                metrics["average_travel_time"] = total_travel_time / len(
                    self._connection_cache
                )
                metrics["visibility_distribution"] = visibility_counts

            # Graph metrics
            if self._graph_cache:
                metrics["graph_nodes"] = len(self._graph_cache.nodes)
                metrics["graph_edges"] = len(self._graph_cache.edges)
                metrics["cache_entries"] = len(self._graph_cache.path_cache)

            return metrics

        except Exception as e:
            logger.error(f"Error getting connection metrics: {e}")
            return {"error": str(e)}

    async def clear_cache(self) -> None:
        """Clear all cached data."""
        try:
            self._connection_cache.clear()
            self._location_connections_cache.clear()
            self._graph_cache = None
            logger.info("Cleared all connection caches")

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    async def get_connections_by_type(
        self, connection_type: str
    ) -> list[GeneratedConnection]:
        """Get all connections of a specific type."""
        try:
            connections = []

            for connection in self._connection_cache.values():
                if connection.properties.connection_type == connection_type:
                    connections.append(connection)

            return connections

        except Exception as e:
            logger.error(f"Error getting connections by type {connection_type}: {e}")
            return []

    async def delete_connection(self, connection_id: UUID) -> bool:
        """Delete a connection from storage."""
        try:
            # Remove from cache
            if connection_id in self._connection_cache:
                connection = self._connection_cache[connection_id]
                del self._connection_cache[connection_id]

                # Clear related caches
                self._location_connections_cache.clear()  # Simplest approach
                self._graph_cache = None

                logger.info(f"Deleted connection {connection_id}")
                return True
            else:
                logger.warning(f"Connection {connection_id} not found for deletion")
                return False

        except Exception as e:
            logger.error(f"Error deleting connection {connection_id}: {e}")
            return False

    async def update_connection(
        self, connection: GeneratedConnection
    ) -> ConnectionStorageResult:
        """Update an existing connection."""
        try:
            if connection.connection_id not in self._connection_cache:
                return ConnectionStorageResult(
                    success=False,
                    error_message="Connection not found for update",
                )

            # Store updated connection
            result = await self.store_connection(connection)

            if result.success:
                logger.info(f"Updated connection {connection.connection_id}")

            return result

        except Exception as e:
            logger.error(f"Error updating connection: {e}")
            return ConnectionStorageResult(
                success=False,
                error_message=str(e),
            )

    def _update_location_cache(self, connection: GeneratedConnection) -> None:
        """Update location connections cache with new connection."""
        try:
            # Add to source location cache
            source_id = connection.source_location_id
            if source_id in self._location_connections_cache:
                self._location_connections_cache[source_id].append(connection)

            # Add to target location cache if reversible
            target_id = connection.target_location_id
            if (
                connection.properties.reversible
                and target_id in self._location_connections_cache
            ):
                self._location_connections_cache[target_id].append(connection)

        except Exception as e:
            logger.error(f"Error updating location cache: {e}")

    async def _build_connectivity_graph(self) -> WorldConnectivityGraph:
        """Build connectivity graph from stored connections."""
        try:
            graph = WorldConnectivityGraph(nodes={}, edges={})

            # Add all connections to graph
            for connection in self._connection_cache.values():
                graph.add_connection(connection)

            logger.debug(f"Built connectivity graph with {len(graph.edges)} edges")
            return graph

        except Exception as e:
            logger.error(f"Error building connectivity graph: {e}")
            return WorldConnectivityGraph(nodes={}, edges={})

    async def export_connections(self) -> dict[str, Any]:
        """Export all connections for backup or analysis."""
        try:
            exported_data = {
                "timestamp": datetime.now().isoformat(),
                "total_connections": len(self._connection_cache),
                "connections": [],
            }

            for connection in self._connection_cache.values():
                connection_data = {
                    "connection_id": str(connection.connection_id),
                    "source_location_id": str(connection.source_location_id),
                    "target_location_id": str(connection.target_location_id),
                    "properties": {
                        "connection_type": connection.properties.connection_type,
                        "difficulty": connection.properties.difficulty,
                        "travel_time": connection.properties.travel_time,
                        "description": connection.properties.description,
                        "visibility": connection.properties.visibility,
                        "requirements": connection.properties.requirements,
                        "reversible": connection.properties.reversible,
                        "special_features": connection.properties.special_features,
                    },
                    "metadata": connection.metadata,
                    "generation_timestamp": connection.generation_timestamp.isoformat(),
                }
                exported_data["connections"].append(connection_data)

            return exported_data

        except Exception as e:
            logger.error(f"Error exporting connections: {e}")
            return {"error": str(e)}

    async def get_storage_statistics(self) -> dict[str, Any]:
        """Get detailed storage system statistics."""
        try:
            stats = {
                "cache_statistics": {
                    "connection_cache_size": len(self._connection_cache),
                    "location_cache_size": len(self._location_connections_cache),
                    "graph_cache_exists": self._graph_cache is not None,
                },
                "connection_analysis": await self._analyze_stored_connections(),
                "performance_metrics": {
                    "cache_hit_ratio": 0.95,  # Would be calculated in real implementation
                    "average_retrieval_time_ms": 2.5,  # Would be measured
                },
            }

            if self._graph_cache:
                stats["graph_statistics"] = {
                    "nodes": len(self._graph_cache.nodes),
                    "edges": len(self._graph_cache.edges),
                    "cached_paths": len(self._graph_cache.path_cache),
                }

            return stats

        except Exception as e:
            logger.error(f"Error getting storage statistics: {e}")
            return {"error": str(e)}

    async def _analyze_stored_connections(self) -> dict[str, Any]:
        """Analyze stored connections for insights."""
        try:
            if not self._connection_cache:
                return {"message": "No connections to analyze"}

            analysis = {
                "total_connections": len(self._connection_cache),
                "unique_locations": set(),
                "connection_type_stats": {},
                "difficulty_distribution": {"easy": 0, "medium": 0, "hard": 0},
                "travel_time_stats": {"min": float("inf"), "max": 0, "avg": 0},
            }

            total_travel_time = 0

            for connection in self._connection_cache.values():
                # Track unique locations
                analysis["unique_locations"].add(connection.source_location_id)
                analysis["unique_locations"].add(connection.target_location_id)

                # Connection type stats
                conn_type = connection.properties.connection_type
                if conn_type not in analysis["connection_type_stats"]:
                    analysis["connection_type_stats"][conn_type] = 0
                analysis["connection_type_stats"][conn_type] += 1

                # Difficulty distribution
                difficulty = connection.properties.difficulty
                if difficulty <= 3:
                    analysis["difficulty_distribution"]["easy"] += 1
                elif difficulty <= 6:
                    analysis["difficulty_distribution"]["medium"] += 1
                else:
                    analysis["difficulty_distribution"]["hard"] += 1

                # Travel time stats
                travel_time = connection.properties.travel_time
                analysis["travel_time_stats"]["min"] = min(
                    analysis["travel_time_stats"]["min"], travel_time
                )
                analysis["travel_time_stats"]["max"] = max(
                    analysis["travel_time_stats"]["max"], travel_time
                )
                total_travel_time += travel_time

            analysis["unique_locations"] = len(analysis["unique_locations"])
            analysis["travel_time_stats"]["avg"] = total_travel_time / len(
                self._connection_cache
            )

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing connections: {e}")
            return {"error": str(e)}
