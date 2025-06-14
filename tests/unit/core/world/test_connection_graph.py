"""
Unit tests for LocationConnectionGraph.
"""

from uuid import uuid4

from game_loop.core.models.navigation_models import ConnectionType
from game_loop.core.world.connection_graph import (
    ConnectionInfo,
    LocationConnectionGraph,
)


class TestLocationConnectionGraph:
    """Test cases for LocationConnectionGraph."""

    def test_add_location(self):
        """Test adding locations to the graph."""
        graph = LocationConnectionGraph()
        loc_id = uuid4()

        graph.add_location(loc_id, {"name": "Test Location", "type": "forest"})

        assert graph.get_location_count() == 1
        assert loc_id in graph.graph.nodes()

    def test_add_connection_bidirectional(self):
        """Test adding bidirectional connections."""
        graph = LocationConnectionGraph()
        loc1_id = uuid4()
        loc2_id = uuid4()

        # Add locations first
        graph.add_location(loc1_id, {"name": "Location 1"})
        graph.add_location(loc2_id, {"name": "Location 2"})

        # Add bidirectional connection
        graph.add_connection(loc1_id, loc2_id, "north", bidirectional=True)

        # Check both directions exist
        assert graph.has_connection(loc1_id, loc2_id)
        assert graph.has_connection(loc2_id, loc1_id)

        # Check connection count (should be 2 for bidirectional)
        assert graph.get_connection_count() == 2

    def test_add_connection_unidirectional(self):
        """Test adding unidirectional connections."""
        graph = LocationConnectionGraph()
        loc1_id = uuid4()
        loc2_id = uuid4()

        # Add locations first
        graph.add_location(loc1_id, {"name": "Location 1"})
        graph.add_location(loc2_id, {"name": "Location 2"})

        # Add unidirectional connection
        graph.add_connection(loc1_id, loc2_id, "north", bidirectional=False)

        # Check only one direction exists
        assert graph.has_connection(loc1_id, loc2_id)
        assert not graph.has_connection(loc2_id, loc1_id)

        assert graph.get_connection_count() == 1

    def test_get_neighbors(self):
        """Test getting neighboring locations."""
        graph = LocationConnectionGraph()
        loc1_id = uuid4()
        loc2_id = uuid4()
        loc3_id = uuid4()

        # Add locations
        graph.add_location(loc1_id, {"name": "Location 1"})
        graph.add_location(loc2_id, {"name": "Location 2"})
        graph.add_location(loc3_id, {"name": "Location 3"})

        # Add connections
        graph.add_connection(loc1_id, loc2_id, "north", bidirectional=False)
        graph.add_connection(loc1_id, loc3_id, "east", bidirectional=False)

        neighbors = graph.get_neighbors(loc1_id)

        assert len(neighbors) == 2
        neighbor_ids = [neighbor[0] for neighbor in neighbors]
        neighbor_directions = [neighbor[1] for neighbor in neighbors]

        assert loc2_id in neighbor_ids
        assert loc3_id in neighbor_ids
        assert "north" in neighbor_directions
        assert "east" in neighbor_directions

    def test_get_connection_info(self):
        """Test getting connection information."""
        graph = LocationConnectionGraph()
        loc1_id = uuid4()
        loc2_id = uuid4()

        # Add locations
        graph.add_location(loc1_id, {"name": "Location 1"})
        graph.add_location(loc2_id, {"name": "Location 2"})

        # Add connection with additional info
        graph.add_connection(
            loc1_id,
            loc2_id,
            "north",
            connection_type=ConnectionType.DOOR,
            description="A wooden door",
            requirements={"required_items": ["key"]},
        )

        connection_info = graph.get_connection_info(loc1_id, loc2_id)

        assert connection_info is not None
        assert connection_info.direction == "north"
        assert connection_info.connection_type == ConnectionType.DOOR
        assert connection_info.description == "A wooden door"
        assert connection_info.requirements == {"required_items": ["key"]}

    def test_reverse_directions(self):
        """Test reverse direction mapping."""
        graph = LocationConnectionGraph()

        # Test standard directions
        assert graph._get_reverse_direction("north") == "south"
        assert graph._get_reverse_direction("south") == "north"
        assert graph._get_reverse_direction("east") == "west"
        assert graph._get_reverse_direction("west") == "east"
        assert graph._get_reverse_direction("up") == "down"
        assert graph._get_reverse_direction("down") == "up"
        assert graph._get_reverse_direction("in") == "out"
        assert graph._get_reverse_direction("out") == "in"

        # Test unknown direction
        assert graph._get_reverse_direction("northwest") is None

    def test_find_connected_components(self):
        """Test finding connected components."""
        graph = LocationConnectionGraph()

        # Create two separate components
        loc1_id = uuid4()
        loc2_id = uuid4()
        loc3_id = uuid4()
        loc4_id = uuid4()

        # Add all locations
        for loc_id in [loc1_id, loc2_id, loc3_id, loc4_id]:
            graph.add_location(loc_id, {"name": f"Location {loc_id}"})

        # Connect loc1 and loc2
        graph.add_connection(loc1_id, loc2_id, "north")

        # Connect loc3 and loc4 (separate component)
        graph.add_connection(loc3_id, loc4_id, "east")

        components = graph.find_connected_components()

        assert len(components) == 2

        # Check that each component contains the expected locations
        component_sizes = [len(comp) for comp in components]
        assert 2 in component_sizes  # Both components should have 2 locations each

    def test_remove_connection(self):
        """Test removing connections."""
        graph = LocationConnectionGraph()
        loc1_id = uuid4()
        loc2_id = uuid4()

        # Add locations and connection
        graph.add_location(loc1_id, {"name": "Location 1"})
        graph.add_location(loc2_id, {"name": "Location 2"})
        graph.add_connection(loc1_id, loc2_id, "north", bidirectional=False)

        assert graph.has_connection(loc1_id, loc2_id)

        # Remove connection
        success = graph.remove_connection(loc1_id, loc2_id)

        assert success
        assert not graph.has_connection(loc1_id, loc2_id)

        # Try to remove non-existent connection
        success = graph.remove_connection(loc1_id, loc2_id)
        assert not success

    def test_clear_graph(self):
        """Test clearing the entire graph."""
        graph = LocationConnectionGraph()
        loc1_id = uuid4()
        loc2_id = uuid4()

        # Add locations and connections
        graph.add_location(loc1_id, {"name": "Location 1"})
        graph.add_location(loc2_id, {"name": "Location 2"})
        graph.add_connection(loc1_id, loc2_id, "north")

        assert graph.get_location_count() > 0
        assert graph.get_connection_count() > 0

        # Clear graph
        graph.clear()

        assert graph.get_location_count() == 0
        assert graph.get_connection_count() == 0

    def test_is_connected(self):
        """Test checking if locations are connected by any path."""
        graph = LocationConnectionGraph()
        loc1_id = uuid4()
        loc2_id = uuid4()
        loc3_id = uuid4()
        loc4_id = uuid4()

        # Add locations
        for loc_id in [loc1_id, loc2_id, loc3_id, loc4_id]:
            graph.add_location(loc_id, {"name": f"Location {loc_id}"})

        # Create a path: loc1 -> loc2 -> loc3
        graph.add_connection(loc1_id, loc2_id, "north", bidirectional=False)
        graph.add_connection(loc2_id, loc3_id, "east", bidirectional=False)

        # loc4 is isolated

        # Test connectivity
        assert graph.is_connected(loc1_id, loc2_id)
        assert graph.is_connected(loc1_id, loc3_id)  # Indirect connection
        assert not graph.is_connected(loc1_id, loc4_id)  # No connection

    def test_shortest_path_length(self):
        """Test calculating shortest path length."""
        graph = LocationConnectionGraph()
        loc1_id = uuid4()
        loc2_id = uuid4()
        loc3_id = uuid4()

        # Add locations
        for loc_id in [loc1_id, loc2_id, loc3_id]:
            graph.add_location(loc_id, {"name": f"Location {loc_id}"})

        # Create a path: loc1 -> loc2 -> loc3
        graph.add_connection(loc1_id, loc2_id, "north")
        graph.add_connection(loc2_id, loc3_id, "east")

        # Test path lengths
        assert graph.shortest_path_length(loc1_id, loc2_id) == 1
        assert graph.shortest_path_length(loc1_id, loc3_id) == 2

        # Test non-existent path
        loc4_id = uuid4()
        graph.add_location(loc4_id, {"name": "Isolated Location"})
        assert graph.shortest_path_length(loc1_id, loc4_id) is None

    def test_get_all_connections(self):
        """Test getting all connection information."""
        graph = LocationConnectionGraph()
        loc1_id = uuid4()
        loc2_id = uuid4()
        loc3_id = uuid4()

        # Add locations
        for loc_id in [loc1_id, loc2_id, loc3_id]:
            graph.add_location(loc_id, {"name": f"Location {loc_id}"})

        # Add connections
        graph.add_connection(loc1_id, loc2_id, "north", bidirectional=False)
        graph.add_connection(loc2_id, loc3_id, "east", bidirectional=False)

        all_connections = graph.get_all_connections()

        assert len(all_connections) == 2

        # Check that all connections are ConnectionInfo objects
        for conn in all_connections:
            assert isinstance(conn, ConnectionInfo)
            assert conn.from_location in [loc1_id, loc2_id]
            assert conn.to_location in [loc2_id, loc3_id]

    def test_get_subgraph(self):
        """Test getting a subgraph."""
        graph = LocationConnectionGraph()
        loc1_id = uuid4()
        loc2_id = uuid4()
        loc3_id = uuid4()

        # Add locations and connections
        for loc_id in [loc1_id, loc2_id, loc3_id]:
            graph.add_location(loc_id, {"name": f"Location {loc_id}"})

        graph.add_connection(loc1_id, loc2_id, "north")
        graph.add_connection(loc2_id, loc3_id, "east")

        # Get subgraph with only loc1 and loc2
        subgraph = graph.get_subgraph({loc1_id, loc2_id})

        assert loc1_id in subgraph.nodes()
        assert loc2_id in subgraph.nodes()
        assert loc3_id not in subgraph.nodes()
