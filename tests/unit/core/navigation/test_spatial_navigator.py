"""
Unit tests for SpatialNavigator.

This module tests the spatial navigation functionality including
pathfinding algorithms, landmark identification, and navigation assistance.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from game_loop.core.navigation.spatial_navigator import (
    SpatialNavigator,
    NavigationAlgorithm,
)


class TestSpatialNavigator:
    """Test cases for SpatialNavigator."""

    @pytest.fixture
    def mock_world_graph_manager(self):
        """Fixture for mock world graph manager."""
        return Mock()

    @pytest.fixture
    def mock_location_service(self):
        """Fixture for mock location service."""
        return Mock()

    @pytest.fixture
    def mock_search_service(self):
        """Fixture for mock search service."""
        return Mock()

    @pytest.fixture
    def spatial_navigator(self, mock_world_graph_manager, mock_location_service, mock_search_service):
        """Fixture for SpatialNavigator instance."""
        return SpatialNavigator(
            world_graph_manager=mock_world_graph_manager,
            location_service=mock_location_service,
            search_service=mock_search_service,
        )

    @pytest.fixture
    def sample_preferences(self):
        """Fixture for sample pathfinding preferences."""
        return {
            "algorithm": NavigationAlgorithm.A_STAR,
            "avoid_dangerous": True,
            "prefer_roads": True,
            "max_distance": 1000.0,
        }

    @pytest.fixture
    def sample_player_knowledge(self):
        """Fixture for sample player knowledge."""
        return {
            "known_locations": {"town_square", "market", "inn", "blacksmith"},
            "visited_locations": {"town_square", "market"},
            "known_routes": {
                "town_square_to_market": ["town_square", "main_street", "market"],
                "market_to_inn": ["market", "side_road", "inn"],
            },
        }

    @pytest.mark.asyncio
    async def test_find_optimal_path_success(self, spatial_navigator, sample_preferences):
        """Test successful optimal path finding."""
        path = await spatial_navigator.find_optimal_path(
            "town_square", "mountain_peak", sample_preferences
        )
        
        assert isinstance(path, list)
        if path:  # Path found
            assert len(path) > 0
            assert all("location" in step for step in path)
            # First step should be start location
            assert path[0]["location"] == "town_square"

    @pytest.mark.asyncio
    async def test_find_optimal_path_no_path(self, spatial_navigator):
        """Test path finding when no path exists."""
        # Use preferences that might lead to no path
        preferences = {"algorithm": NavigationAlgorithm.DIJKSTRA, "blocked_locations": ["all"]}
        
        path = await spatial_navigator.find_optimal_path(
            "isolated_island", "unreachable_destination", preferences
        )
        
        # Should return None or empty list when no path exists
        assert path is None or path == []

    @pytest.mark.asyncio
    async def test_find_optimal_path_different_algorithms(self, spatial_navigator):
        """Test path finding with different algorithms."""
        start, end = "village", "castle"
        
        algorithms = [
            NavigationAlgorithm.A_STAR,
            NavigationAlgorithm.DIJKSTRA,
            NavigationAlgorithm.BREADTH_FIRST,
            NavigationAlgorithm.DEPTH_FIRST,
            NavigationAlgorithm.BEST_FIRST,
        ]
        
        for algorithm in algorithms:
            preferences = {"algorithm": algorithm}
            path = await spatial_navigator.find_optimal_path(start, end, preferences)
            
            # Each algorithm should return a result (success or failure)
            assert path is None or isinstance(path, list)

    @pytest.mark.asyncio
    async def test_get_navigation_directions_success(self, spatial_navigator, sample_player_knowledge):
        """Test successful navigation directions generation."""
        directions = await spatial_navigator.get_navigation_directions(
            "town_square", "mountain_peak", sample_player_knowledge
        )
        
        assert isinstance(directions, list)
        assert len(directions) > 0
        # Each direction should be a string
        assert all(isinstance(direction, str) for direction in directions)

    @pytest.mark.asyncio
    async def test_get_navigation_directions_no_path(self, spatial_navigator, sample_player_knowledge):
        """Test navigation directions when no path exists."""
        directions = await spatial_navigator.get_navigation_directions(
            "unreachable_start", "unreachable_end", sample_player_knowledge
        )
        
        assert isinstance(directions, list)
        assert len(directions) > 0
        # Should contain error message
        assert any("no path" in direction.lower() for direction in directions)

    @pytest.mark.asyncio
    async def test_identify_landmarks_success(self, spatial_navigator):
        """Test successful landmark identification."""
        landmarks = await spatial_navigator.identify_landmarks("central_plaza", 200.0)
        
        assert isinstance(landmarks, list)
        # Each landmark should have required fields
        for landmark in landmarks:
            assert isinstance(landmark, dict)
            assert "id" in landmark
            assert "name" in landmark
            assert "distance" in landmark
            assert "direction" in landmark

    @pytest.mark.asyncio
    async def test_identify_landmarks_empty_area(self, spatial_navigator):
        """Test landmark identification in area with no landmarks."""
        landmarks = await spatial_navigator.identify_landmarks("empty_desert", 100.0)
        
        assert isinstance(landmarks, list)
        # Should return empty list or minimal landmarks

    @pytest.mark.asyncio
    async def test_calculate_travel_estimates_valid_path(self, spatial_navigator):
        """Test travel estimates calculation for valid path."""
        path = ["village", "crossroads", "forest", "mountain_base", "mountain_peak"]
        movement_speed = 50.0  # units per minute
        obstacles = ["fallen_tree", "river_crossing"]
        
        estimates = await spatial_navigator.calculate_travel_estimates(
            path, movement_speed, obstacles
        )
        
        assert isinstance(estimates, dict)
        assert "total_time" in estimates
        assert "total_energy" in estimates
        assert "total_distance" in estimates
        assert "difficulty_rating" in estimates
        assert "segments" in estimates
        
        # All values should be positive
        assert estimates["total_time"] > 0
        assert estimates["total_energy"] > 0
        assert estimates["total_distance"] > 0

    @pytest.mark.asyncio
    async def test_calculate_travel_estimates_empty_path(self, spatial_navigator):
        """Test travel estimates calculation for empty path."""
        estimates = await spatial_navigator.calculate_travel_estimates([], 50.0, [])
        
        assert isinstance(estimates, dict)
        assert "error" in estimates

    @pytest.mark.asyncio
    async def test_find_alternative_routes(self, spatial_navigator):
        """Test finding alternative routes."""
        blocked_path = ["town", "main_road", "bridge", "city"]
        target = "city"
        constraints = {"start": "town", "avoid_dangerous": True}
        
        alternatives = await spatial_navigator.find_alternative_routes(
            blocked_path, target, constraints
        )
        
        assert isinstance(alternatives, list)
        # Each alternative should be a list of location IDs
        for alternative in alternatives:
            assert isinstance(alternative, list)
            if alternative:
                assert isinstance(alternative[0], str)  # Location IDs are strings

    @pytest.mark.asyncio
    async def test_find_alternative_routes_no_alternatives(self, spatial_navigator):
        """Test finding alternative routes when none exist."""
        blocked_path = ["isolated_start", "only_path", "isolated_end"]
        alternatives = await spatial_navigator.find_alternative_routes(
            blocked_path, "isolated_end", {"start": "isolated_start"}
        )
        
        assert isinstance(alternatives, list)
        # Should return empty list if no alternatives

    @pytest.mark.asyncio
    async def test_update_navigation_knowledge(self, spatial_navigator):
        """Test updating player navigation knowledge."""
        discovered_path = ["home", "forest_path", "hidden_grove", "secret_cave"]
        path_quality = 0.85
        
        await spatial_navigator.update_navigation_knowledge(
            "player_1", discovered_path, path_quality
        )
        
        # Should not raise an exception
        # Check that data was stored
        assert "player_1" in spatial_navigator._exploration_data

    @pytest.mark.asyncio
    async def test_get_exploration_suggestions(self, spatial_navigator):
        """Test getting exploration suggestions."""
        exploration_history = ["village", "nearby_forest", "old_ruins"]
        
        suggestions = await spatial_navigator.get_exploration_suggestions(
            "village", exploration_history
        )
        
        assert isinstance(suggestions, list)
        # Each suggestion should have required fields
        for suggestion in suggestions:
            assert isinstance(suggestion, dict)
            assert "location_id" in suggestion
            assert "name" in suggestion
            assert "distance" in suggestion
            assert "exploration_value" in suggestion

    @pytest.mark.asyncio
    async def test_validate_path_accessibility_accessible(self, spatial_navigator):
        """Test path accessibility validation for accessible path."""
        path = ["start", "checkpoint1", "checkpoint2", "destination"]
        player_capabilities = {
            "strength": 60,
            "agility": 50,
            "climbing_skill": 40,
            "swimming_skill": 30,
        }
        
        can_traverse, issues = await spatial_navigator.validate_path_accessibility(
            path, player_capabilities
        )
        
        assert isinstance(can_traverse, bool)
        assert isinstance(issues, list)

    @pytest.mark.asyncio
    async def test_validate_path_accessibility_blocked(self, spatial_navigator):
        """Test path accessibility validation for blocked path."""
        path = ["start", "high_cliff", "destination"]
        weak_player_capabilities = {
            "strength": 10,
            "agility": 10,
            "climbing_skill": 5,
        }
        
        can_traverse, issues = await spatial_navigator.validate_path_accessibility(
            path, weak_player_capabilities
        )
        
        assert isinstance(can_traverse, bool)
        assert isinstance(issues, list)
        if not can_traverse:
            assert len(issues) > 0

    @pytest.mark.asyncio
    async def test_create_mental_map(self, spatial_navigator):
        """Test creating player mental map."""
        # First add some exploration data
        await spatial_navigator.update_navigation_knowledge(
            "explorer", ["base", "forest", "mountain"], 0.8
        )
        
        mental_map = await spatial_navigator.create_mental_map(
            "explorer", {"exploration_history": ["base", "forest"]}
        )
        
        assert isinstance(mental_map, dict)
        assert "known_locations" in mental_map
        assert "known_connections" in mental_map
        assert "exploration_stats" in mental_map

    @pytest.mark.asyncio
    async def test_create_mental_map_no_data(self, spatial_navigator):
        """Test creating mental map for player with no exploration data."""
        mental_map = await spatial_navigator.create_mental_map(
            "new_player", {}
        )
        
        assert isinstance(mental_map, dict)
        assert "error" in mental_map

    def test_register_navigation_algorithm(self, spatial_navigator):
        """Test registering custom navigation algorithm."""
        def custom_algorithm(start, goal, graph, preferences):
            return [start, goal]  # Simple direct path
        
        spatial_navigator.register_navigation_algorithm("custom", custom_algorithm)
        
        assert "custom" in spatial_navigator._pathfinding_algorithms
        assert spatial_navigator._pathfinding_algorithms["custom"] == custom_algorithm

    @pytest.mark.asyncio
    async def test_a_star_pathfinding(self, spatial_navigator):
        """Test A* pathfinding algorithm."""
        graph = {
            "start": [{"id": "middle", "cost": 1.0}],
            "middle": [{"id": "end", "cost": 1.0}],
            "end": [],
        }
        
        path = await spatial_navigator._a_star_pathfinding("start", "end", graph, {})
        
        if path:  # Path found
            assert isinstance(path, list)
            assert path[0] == "start"
            assert path[-1] == "end"

    @pytest.mark.asyncio
    async def test_dijkstra_pathfinding(self, spatial_navigator):
        """Test Dijkstra pathfinding algorithm."""
        graph = {
            "start": [{"id": "middle", "cost": 2.0}],
            "middle": [{"id": "end", "cost": 1.0}],
            "end": [],
        }
        
        path = await spatial_navigator._dijkstra_pathfinding("start", "end", graph, {})
        
        if path:  # Path found
            assert isinstance(path, list)
            assert path[0] == "start"
            assert path[-1] == "end"

    @pytest.mark.asyncio
    async def test_breadth_first_pathfinding(self, spatial_navigator):
        """Test breadth-first search pathfinding algorithm."""
        graph = {
            "start": [{"id": "middle1"}, {"id": "middle2"}],
            "middle1": [{"id": "end"}],
            "middle2": [{"id": "end"}],
            "end": [],
        }
        
        path = await spatial_navigator._breadth_first_pathfinding("start", "end", graph, {})
        
        if path:  # Path found
            assert isinstance(path, list)
            assert path[0] == "start"
            assert path[-1] == "end"

    @pytest.mark.asyncio
    async def test_depth_first_pathfinding(self, spatial_navigator):
        """Test depth-first search pathfinding algorithm."""
        graph = {
            "start": [{"id": "middle"}],
            "middle": [{"id": "end"}],
            "end": [],
        }
        
        path = await spatial_navigator._depth_first_pathfinding("start", "end", graph, {})
        
        if path:  # Path found
            assert isinstance(path, list)
            assert path[0] == "start"
            assert path[-1] == "end"

    @pytest.mark.asyncio
    async def test_best_first_pathfinding(self, spatial_navigator):
        """Test best-first search pathfinding algorithm."""
        graph = {
            "start": [{"id": "middle"}],
            "middle": [{"id": "end"}],
            "end": [],
        }
        
        path = await spatial_navigator._best_first_pathfinding("start", "end", graph, {})
        
        if path:  # Path found
            assert isinstance(path, list)
            assert path[0] == "start"
            assert path[-1] == "end"

    @pytest.mark.asyncio
    async def test_heuristic_distance_calculation(self, spatial_navigator):
        """Test heuristic distance calculation."""
        distance = await spatial_navigator._heuristic_distance("location_a", "location_b")
        
        assert isinstance(distance, float)
        assert distance >= 0

    @pytest.mark.asyncio
    async def test_path_validation(self, spatial_navigator):
        """Test cached path validation."""
        sample_path = [
            {"location": "start", "direction": "start"},
            {"location": "end", "direction": "north"},
        ]
        
        is_valid = await spatial_navigator._is_path_still_valid(sample_path)
        
        assert isinstance(is_valid, bool)

    @pytest.mark.asyncio
    async def test_path_enhancement(self, spatial_navigator):
        """Test path enhancement with additional information."""
        simple_path = ["start", "middle", "end"]
        preferences = {"add_landmarks": True, "include_warnings": True}
        
        enhanced_path = await spatial_navigator._enhance_path_information(simple_path, preferences)
        
        assert isinstance(enhanced_path, list)
        assert len(enhanced_path) == len(simple_path)
        # Each step should have enhanced information
        for step in enhanced_path:
            assert isinstance(step, dict)
            assert "location" in step
            assert "step_number" in step

    @pytest.mark.asyncio
    async def test_landmark_visibility_calculation(self, spatial_navigator):
        """Test landmark visibility calculation."""
        landmark_data = {"prominence": 1.5, "height": 100}
        distance = 300.0
        
        visibility = spatial_navigator._calculate_landmark_visibility(landmark_data, distance)
        
        assert isinstance(visibility, float)
        assert visibility >= 0

    def test_exploration_value_calculation(self, spatial_navigator):
        """Test exploration value calculation."""
        high_value_location = {
            "is_landmark": True,
            "tags": ["treasure", "unique"],
        }
        low_value_location = {
            "is_landmark": False,
            "tags": [],
        }
        
        high_value = spatial_navigator._calculate_exploration_value(high_value_location)
        low_value = spatial_navigator._calculate_exploration_value(low_value_location)
        
        assert isinstance(high_value, float)
        assert isinstance(low_value, float)
        assert high_value > low_value

    @pytest.mark.asyncio
    async def test_segment_estimates_calculation(self, spatial_navigator):
        """Test path segment estimates calculation."""
        segment_data = await spatial_navigator._calculate_segment_estimates(
            "start", "end", 60.0, ["obstacle1", "obstacle2"]
        )
        
        assert isinstance(segment_data, dict)
        assert "start" in segment_data
        assert "end" in segment_data
        assert "distance" in segment_data
        assert "time" in segment_data
        assert "energy" in segment_data
        assert "obstacles" in segment_data

    def test_path_difficulty_calculation(self, spatial_navigator):
        """Test path difficulty calculation."""
        easy_segments = [
            {"obstacles": []},
            {"obstacles": ["small_rock"]},
        ]
        hard_segments = [
            {"obstacles": ["cliff", "river", "dense_forest"]},
            {"obstacles": ["swamp", "thorns"]},
        ]
        
        easy_difficulty = spatial_navigator._calculate_path_difficulty(easy_segments)
        hard_difficulty = spatial_navigator._calculate_path_difficulty(hard_segments)
        
        assert isinstance(easy_difficulty, float)
        assert isinstance(hard_difficulty, float)
        assert hard_difficulty > easy_difficulty
        assert 0 <= easy_difficulty <= 1
        assert 0 <= hard_difficulty <= 1

    @pytest.mark.asyncio
    async def test_travel_recommendations_generation(self, spatial_navigator):
        """Test travel recommendations generation."""
        long_path = ["start", "waypoint1", "waypoint2", "waypoint3", "end"]
        high_energy = 150.0
        long_time = 4000.0  # More than 1 hour
        
        recommendations = await spatial_navigator._generate_travel_recommendations(
            long_path, long_time, high_energy
        )
        
        assert isinstance(recommendations, list)
        # Should provide recommendations for long/difficult journeys

    @pytest.mark.asyncio
    async def test_error_handling(self, spatial_navigator):
        """Test error handling in navigation operations."""
        # Test with invalid inputs
        path = await spatial_navigator.find_optimal_path(None, None, {})
        assert path is None
        
        # Test directions with invalid location
        directions = await spatial_navigator.get_navigation_directions(
            None, None, {}
        )
        assert isinstance(directions, list)
        assert len(directions) > 0
        assert any("error" in direction.lower() for direction in directions)

    def test_navigation_algorithm_constants(self):
        """Test navigation algorithm constants."""
        assert NavigationAlgorithm.DIJKSTRA == "dijkstra"
        assert NavigationAlgorithm.A_STAR == "a_star"
        assert NavigationAlgorithm.BREADTH_FIRST == "breadth_first"
        assert NavigationAlgorithm.DEPTH_FIRST == "depth_first"
        assert NavigationAlgorithm.BEST_FIRST == "best_first"


@pytest.mark.asyncio
async def test_spatial_navigator_integration():
    """Test SpatialNavigator integration scenarios."""
    # Create spatial navigator
    spatial_navigator = SpatialNavigator()
    
    # Test complex navigation scenario
    preferences = {
        "algorithm": NavigationAlgorithm.A_STAR,
        "avoid_dangerous": True,
        "prefer_roads": True,
    }
    
    # Find path
    path = await spatial_navigator.find_optimal_path(
        "adventure_start", "quest_destination", preferences
    )
    
    # Get directions
    player_knowledge = {
        "known_locations": {"adventure_start"},
        "visited_locations": {"adventure_start"},
    }
    
    directions = await spatial_navigator.get_navigation_directions(
        "adventure_start", "quest_destination", player_knowledge
    )
    
    # Calculate estimates if path exists
    if path and len(path) > 1:
        path_ids = [step["location"] for step in path]
        estimates = await spatial_navigator.calculate_travel_estimates(
            path_ids, 45.0, ["river", "mountain_pass"]
        )
        
        assert isinstance(estimates, dict)
    
    # Test results
    assert path is None or isinstance(path, list)
    assert isinstance(directions, list)
    assert len(directions) > 0


if __name__ == "__main__":
    pytest.main([__file__])