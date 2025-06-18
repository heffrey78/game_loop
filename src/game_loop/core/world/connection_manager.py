"""
World connection manager for maintaining bidirectional connections and preventing isolation.

This module manages location connections ensuring all areas remain connected and navigable.
"""

import logging
from typing import Any
from uuid import UUID

import networkx as nx

from ...database.models.world import LocationConnection

logger = logging.getLogger(__name__)


class WorldConnectionManager:
    """Manage bidirectional world connections and prevent isolation."""

    def __init__(self, state_manager: Any) -> None:
        self.state_manager = state_manager
        self.connection_graph = nx.Graph()
        self._direction_reverses = {
            "north": "south",
            "south": "north",
            "east": "west",
            "west": "east",
            "up": "down",
            "down": "up",
            "northeast": "southwest",
            "southwest": "northeast",
            "northwest": "southeast",
            "southeast": "northwest",
        }

    async def create_bidirectional_connection(
        self,
        from_location_id: UUID,
        to_location_id: UUID,
        direction: str,
        description: str | None = None,
    ) -> bool:
        """Create connection and automatic reverse connection."""
        try:
            reverse_direction = self._get_reverse_direction(direction)

            # Create primary connection
            await self._create_connection(
                from_location_id, to_location_id, direction, description
            )

            # Create reverse connection
            reverse_description = self._generate_reverse_description(
                description, reverse_direction
            )
            await self._create_connection(
                to_location_id, from_location_id, reverse_direction, reverse_description
            )

            # Update graph
            self.connection_graph.add_edge(
                str(from_location_id),
                str(to_location_id),
                direction=direction,
                reverse_direction=reverse_direction,
            )

            return True

        except Exception as e:
            logger.error(f"Error creating bidirectional connection: {e}")
            return False

    async def _create_connection(
        self,
        from_id: UUID,
        to_id: UUID,
        direction: str,
        description: str | None = None,
    ) -> None:
        """Create a single directional connection in database."""
        async with self.state_manager.get_database_session() as session:
            connection = LocationConnection(
                from_location_id=from_id,
                to_location_id=to_id,
                direction=direction,
                description=description or f"Path {direction}",
                is_visible=True,
            )
            session.add(connection)
            await session.commit()

    def _get_reverse_direction(self, direction: str) -> str:
        """Get the opposite direction for bidirectional connections."""
        return str(self._direction_reverses.get(direction.lower(), direction))

    def _generate_reverse_description(
        self, original_desc: str | None, reverse_dir: str
    ) -> str:
        """Generate description for reverse connection."""
        if original_desc:
            return f"Path {reverse_dir} (return route)"
        return f"Path {reverse_dir}"

    async def validate_world_connectivity(self) -> dict[str, Any]:
        """Ensure all locations are reachable from starting point."""
        try:
            # Build current graph from database
            await self._rebuild_connection_graph()

            # Find starting location
            starting_location = await self._get_starting_location()
            if not starting_location:
                return {"status": "error", "message": "No starting location found"}

            start_id = str(starting_location)

            # Check connectivity
            if start_id not in self.connection_graph:
                return {
                    "status": "error",
                    "message": "Starting location not in connection graph",
                }

            reachable = set(nx.bfs_tree(self.connection_graph, start_id))
            all_locations = set(self.connection_graph.nodes())

            isolated_locations = all_locations - reachable

            result = {
                "status": "success",
                "total_locations": len(all_locations),
                "reachable_locations": len(reachable),
                "isolated_locations": list(isolated_locations),
                "connectivity_ratio": (
                    len(reachable) / len(all_locations) if all_locations else 0
                ),
            }

            # Auto-repair if needed
            if isolated_locations:
                repair_result = await self._repair_isolated_locations(
                    isolated_locations
                )
                result["repair_attempted"] = True
                result["repair_result"] = repair_result

            return result

        except Exception as e:
            logger.error(f"Error validating world connectivity: {e}")
            return {"status": "error", "message": str(e)}

    async def _rebuild_connection_graph(self) -> None:
        """Rebuild the connection graph from current database state."""
        self.connection_graph.clear()

        async with self.state_manager.get_database_session() as session:
            # Query all connections
            result = await session.execute(
                "SELECT from_location_id, to_location_id, direction FROM location_connections WHERE is_visible = true"
            )
            connections = result.fetchall()

            for from_id, to_id, direction in connections:
                self.connection_graph.add_edge(
                    str(from_id), str(to_id), direction=direction
                )

    async def _get_starting_location(self) -> UUID | None:
        """Get the designated starting location ID."""
        async with self.state_manager.get_database_session() as session:
            # Look for location marked as starting point
            result = await session.execute(
                """
                SELECT id FROM locations 
                WHERE state_json->>'is_starting_location' = 'true'
                   OR name ILIKE '%reception%'
                   OR name ILIKE '%entrance%'
                   OR name ILIKE '%lobby%'
                ORDER BY created_at ASC
                LIMIT 1
                """
            )
            row = result.fetchone()
            return UUID(row[0]) if row else None

    async def _repair_isolated_locations(
        self, isolated_ids: set[str]
    ) -> dict[str, Any]:
        """Create connections to repair isolated areas."""
        repairs_made = 0
        errors = []

        for isolated_id_str in isolated_ids:
            try:
                isolated_id = UUID(isolated_id_str)

                # Find nearest connected location
                nearest_connected = await self._find_nearest_connected_location(
                    isolated_id
                )
                if nearest_connected:
                    # Create emergency connection
                    direction = await self._determine_logical_direction(
                        isolated_id, nearest_connected
                    )
                    success = await self.create_bidirectional_connection(
                        isolated_id,
                        nearest_connected,
                        direction,
                        "Emergency connection to restore connectivity",
                    )
                    if success:
                        repairs_made += 1
                    else:
                        errors.append(
                            f"Failed to create emergency connection for {isolated_id}"
                        )
                else:
                    errors.append(f"No connected location found near {isolated_id}")

            except Exception as e:
                errors.append(f"Error repairing {isolated_id_str}: {e}")

        return {
            "repairs_made": repairs_made,
            "total_isolated": len(isolated_ids),
            "errors": errors,
        }

    async def _find_nearest_connected_location(
        self, isolated_id: UUID
    ) -> UUID | None:
        """Find the nearest location that is connected to the main graph."""
        try:
            # Get starting location as reference point for "connected"
            starting_location = await self._get_starting_location()
            if not starting_location:
                return None

            start_str = str(starting_location)
            if start_str not in self.connection_graph:
                return None

            # Get all locations reachable from start
            reachable = set(nx.bfs_tree(self.connection_graph, start_str))

            if not reachable:
                return starting_location

            # For now, just return the starting location as the repair target
            # In a more sophisticated system, we'd calculate actual distances
            return starting_location

        except Exception as e:
            logger.error(f"Error finding nearest connected location: {e}")
            return None

    async def _determine_logical_direction(
        self, from_location: UUID, to_location: UUID
    ) -> str:
        """Determine a logical direction for connecting two locations."""
        try:
            # Get location details to determine logical connection
            from_loc = await self.state_manager.get_location_details(from_location)
            to_loc = await self.state_manager.get_location_details(to_location)

            if not from_loc or not to_loc:
                return "south"  # Default direction

            # Simple heuristic: if going to/from upper floors, use up/down
            from_name = getattr(from_loc, "name", "").lower()
            to_name = getattr(to_loc, "name", "").lower()

            if any(
                word in to_name
                for word in ["upper", "second", "third", "floor", "upstairs"]
            ):
                return "up"
            elif any(
                word in from_name
                for word in ["upper", "second", "third", "floor", "upstairs"]
            ):
                return "down"
            elif any(word in to_name for word in ["entrance", "lobby", "reception"]):
                return "south"  # Generally head south to main areas
            else:
                return "north"  # Default exploration direction

        except Exception as e:
            logger.error(f"Error determining logical direction: {e}")
            return "south"

    async def find_path_between_locations(
        self, from_location: UUID, to_location: UUID
    ) -> list[dict[str, Any]] | None:
        """Find shortest path between two locations."""
        try:
            await self._rebuild_connection_graph()

            from_str = str(from_location)
            to_str = str(to_location)

            if (
                from_str not in self.connection_graph
                or to_str not in self.connection_graph
            ):
                return None

            try:
                path_nodes = nx.shortest_path(self.connection_graph, from_str, to_str)

                # Convert to path with directions
                path_with_directions = []
                for i in range(len(path_nodes) - 1):
                    current = path_nodes[i]
                    next_node = path_nodes[i + 1]

                    # Get direction from edge data
                    edge_data = self.connection_graph.get_edge_data(current, next_node)
                    direction = (
                        edge_data.get("direction", "forward")
                        if edge_data
                        else "forward"
                    )

                    path_with_directions.append(
                        {
                            "from_location": UUID(current),
                            "to_location": UUID(next_node),
                            "direction": direction,
                        }
                    )

                return path_with_directions

            except nx.NetworkXNoPath:
                # No path exists - attempt emergency repair
                logger.warning(
                    f"No path found between {from_location} and {to_location}"
                )
                return None

        except Exception as e:
            logger.error(f"Error finding path: {e}")
            return None

    async def get_connection_statistics(self) -> dict[str, Any]:
        """Get statistics about world connectivity."""
        try:
            await self._rebuild_connection_graph()

            total_nodes = self.connection_graph.number_of_nodes()
            total_edges = self.connection_graph.number_of_edges()

            # Check if graph is connected
            is_connected = (
                nx.is_connected(self.connection_graph) if total_nodes > 0 else True
            )

            # Find components
            components = list(nx.connected_components(self.connection_graph))
            largest_component_size = (
                max(len(comp) for comp in components) if components else 0
            )

            return {
                "total_locations": total_nodes,
                "total_connections": total_edges,
                "is_fully_connected": is_connected,
                "number_of_components": len(components),
                "largest_component_size": largest_component_size,
                "average_connections_per_location": (
                    total_edges * 2 / total_nodes if total_nodes > 0 else 0
                ),
            }

        except Exception as e:
            logger.error(f"Error getting connection statistics: {e}")
            return {"error": str(e)}
