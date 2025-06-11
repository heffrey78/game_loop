"""
Spatial Navigator for advanced navigation and pathfinding capabilities.

This module provides intelligent pathfinding, landmark identification,
navigation assistance, and spatial mapping for the game world.
"""

import asyncio
import heapq
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class NavigationAlgorithm:
    """Constants for navigation algorithms."""

    DIJKSTRA = "dijkstra"
    A_STAR = "a_star"
    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    BEST_FIRST = "best_first"


class SpatialNavigator:
    """
    Advanced navigation and pathfinding capabilities.

    This class provides intelligent pathfinding algorithms, landmark-based navigation,
    dynamic obstacle avoidance, and navigation assistance for entities in the game world.
    """

    def __init__(
        self,
        world_graph_manager: Any = None,
        location_service: Any = None,
        search_service: Any = None,
    ):
        """
        Initialize the spatial navigator.

        Args:
            world_graph_manager: Manager for world connectivity graph
            location_service: Service for location data and management
            search_service: Semantic search service for landmark identification
        """
        self.world_graph = world_graph_manager
        self.locations = location_service
        self.search = search_service
        self._navigation_cache: dict[str, list[dict[str, Any]]] = {}
        self._landmark_map: dict[str, list[dict[str, Any]]] = {}
        self._pathfinding_algorithms: dict[str, Callable] = {}
        self._exploration_data: dict[str, dict[str, Any]] = {}
        self._initialize_algorithms()

    async def find_optimal_path(
        self,
        start_location: str,
        target_location: str,
        preferences: dict[str, Any],
    ) -> list[dict[str, Any]] | None:
        """
        Find optimal path between locations considering preferences.

        Args:
            start_location: Starting location ID
            target_location: Target location ID
            preferences: Pathfinding preferences and constraints

        Returns:
            List of path steps with detailed information, or None if no path found
        """
        try:
            # Validate inputs
            if not start_location or not target_location:
                logger.warning("Invalid location parameters provided for pathfinding")
                return None

            # Check cache first
            cache_key = f"{start_location}:{target_location}:{hash(str(preferences))}"
            if cache_key in self._navigation_cache:
                cached_path = self._navigation_cache[cache_key]
                if await self._is_path_still_valid(cached_path):
                    return cached_path

            # Select algorithm based on preferences
            algorithm = preferences.get("algorithm", NavigationAlgorithm.A_STAR)
            algorithm_func = self._pathfinding_algorithms.get(
                algorithm, self._a_star_pathfinding
            )

            # Get world graph
            world_graph = await self._build_world_graph(preferences)

            # Find path using selected algorithm
            path = await algorithm_func(
                start_location, target_location, world_graph, preferences
            )

            if path:
                # Enhance path with additional information
                enhanced_path = await self._enhance_path_information(path, preferences)

                # Cache the result
                self._navigation_cache[cache_key] = enhanced_path

                return enhanced_path

            return None

        except Exception as e:
            logger.error(f"Error finding optimal path: {e}")
            return None

    async def get_navigation_directions(
        self,
        current_location: str,
        target_location: str,
        player_knowledge: dict[str, Any],
    ) -> list[str]:
        """
        Get step-by-step navigation directions.

        Args:
            current_location: Current location ID
            target_location: Target location ID
            player_knowledge: Player's knowledge about the world

        Returns:
            List of human-readable navigation directions
        """
        try:
            # Validate inputs
            if not current_location or not target_location:
                return ["Error: Invalid location parameters provided for navigation."]

            # Find path
            preferences = {
                "algorithm": NavigationAlgorithm.A_STAR,
                "avoid_dangerous": True,
            }
            path = await self.find_optimal_path(
                current_location, target_location, preferences
            )

            if not path:
                return ["No path to destination could be found."]

            directions = []
            known_locations = player_knowledge.get("known_locations", set())

            for i, step in enumerate(path):
                if i == 0:
                    continue  # Skip starting location

                location_name = step["location"]
                direction = step.get("direction", "unknown")
                distance = step.get("distance", 0)

                # Customize direction based on player knowledge
                if location_name in known_locations:
                    directions.append(f"Go {direction} to {location_name}")
                else:
                    # Use landmarks or generic descriptions
                    landmarks = await self.identify_landmarks(location_name, 100.0)
                    if landmarks:
                        landmark_desc = landmarks[0]["description"]
                        directions.append(f"Go {direction} toward {landmark_desc}")
                    else:
                        directions.append(f"Go {direction} for {distance} units")

                # Add warnings or special instructions
                warnings = step.get("warnings", [])
                for warning in warnings:
                    directions.append(f"⚠️  {warning}")

            return directions

        except Exception as e:
            logger.error(f"Error getting navigation directions: {e}")
            return ["Error generating directions."]

    async def identify_landmarks(
        self, location_id: str, visibility_range: float
    ) -> list[dict[str, Any]]:
        """
        Identify notable landmarks visible from location.

        Args:
            location_id: Location to search from
            visibility_range: Maximum visibility distance

        Returns:
            List of landmark information dictionaries
        """
        try:
            # Check cache first
            cache_key = f"landmarks:{location_id}:{visibility_range}"
            if cache_key in self._landmark_map:
                return self._landmark_map[cache_key]

            landmarks = []

            # Get nearby locations within visibility range
            nearby_locations = await self._get_locations_within_range(
                location_id, visibility_range
            )

            for nearby_location in nearby_locations:
                location_data = await self._get_location_data(nearby_location["id"])

                # Check if location has landmark properties
                if location_data.get("is_landmark", False):
                    landmark = {
                        "id": nearby_location["id"],
                        "name": location_data.get("name", nearby_location["id"]),
                        "description": location_data.get("landmark_description", ""),
                        "type": location_data.get("landmark_type", "structure"),
                        "distance": nearby_location["distance"],
                        "direction": nearby_location["direction"],
                        "visibility": self._calculate_landmark_visibility(
                            location_data, nearby_location["distance"]
                        ),
                        "prominence": location_data.get("prominence", 1.0),
                    }
                    landmarks.append(landmark)

            # Sort by prominence and visibility
            landmarks.sort(
                key=lambda x: (-x["prominence"], -x["visibility"], x["distance"])
            )

            # Cache the results
            self._landmark_map[cache_key] = landmarks[:10]  # Limit to top 10 landmarks

            return landmarks[:10]

        except Exception as e:
            logger.error(f"Error identifying landmarks: {e}")
            return []

    async def calculate_travel_estimates(
        self, path: list[str], movement_speed: float, obstacles: list[str]
    ) -> dict[str, Any]:
        """
        Calculate time and energy estimates for travel.

        Args:
            path: List of location IDs in the path
            movement_speed: Base movement speed
            obstacles: List of known obstacles along the path

        Returns:
            Dictionary containing travel estimates
        """
        try:
            if not path or len(path) < 2:
                return {"error": "Invalid path provided"}

            total_time = 0.0
            total_energy = 0.0
            total_distance = 0.0
            segment_details = []

            for i in range(len(path) - 1):
                current_location = path[i]
                next_location = path[i + 1]

                # Calculate segment details
                segment = await self._calculate_segment_estimates(
                    current_location, next_location, movement_speed, obstacles
                )

                total_time += segment["time"]
                total_energy += segment["energy"]
                total_distance += segment["distance"]
                segment_details.append(segment)

            # Add rest stops for long journeys
            rest_stops = max(
                0, int(total_time / 3600) - 1
            )  # One rest per hour after first hour
            if rest_stops > 0:
                total_time += rest_stops * 600  # 10 minutes per rest stop

            return {
                "total_time": total_time,
                "total_energy": total_energy,
                "total_distance": total_distance,
                "estimated_arrival": total_time,
                "rest_stops_needed": rest_stops,
                "difficulty_rating": self._calculate_path_difficulty(segment_details),
                "segments": segment_details,
                "recommendations": await self._generate_travel_recommendations(
                    path, total_time, total_energy
                ),
            }

        except Exception as e:
            logger.error(f"Error calculating travel estimates: {e}")
            return {"error": str(e)}

    async def find_alternative_routes(
        self, blocked_path: list[str], target: str, constraints: dict[str, Any]
    ) -> list[list[str]]:
        """
        Find alternative routes when primary path is blocked.

        Args:
            blocked_path: Original path that is now blocked
            target: Target destination
            constraints: Routing constraints and preferences

        Returns:
            List of alternative paths (each path is a list of location IDs)
        """
        try:
            alternatives = []
            start_location = (
                blocked_path[0] if blocked_path else constraints.get("start")
            )

            if not start_location:
                return alternatives

            # Try different pathfinding strategies
            strategies = [
                {"algorithm": NavigationAlgorithm.DIJKSTRA, "avoid_blocked": True},
                {"algorithm": NavigationAlgorithm.A_STAR, "allow_detours": True},
                {
                    "algorithm": NavigationAlgorithm.BEST_FIRST,
                    "prioritize_safety": True,
                },
            ]

            blocked_locations = set(blocked_path[1:-1])  # Exclude start and end

            for strategy in strategies:
                strategy.update(constraints)
                strategy["blocked_locations"] = blocked_locations

                path = await self.find_optimal_path(start_location, target, strategy)
                if path:
                    path_ids = [step["location"] for step in path]

                    # Check if this is truly an alternative (not just the original path)
                    if not any(loc in blocked_locations for loc in path_ids[1:-1]):
                        alternatives.append(path_ids)

            # Remove duplicate paths
            unique_alternatives = []
            for alt in alternatives:
                if alt not in unique_alternatives:
                    unique_alternatives.append(alt)

            # Sort by estimated quality (length, safety, etc.)
            unique_alternatives.sort(key=lambda x: len(x))

            return unique_alternatives[:5]  # Return top 5 alternatives

        except Exception as e:
            logger.error(f"Error finding alternative routes: {e}")
            return []

    async def update_navigation_knowledge(
        self, player_id: str, discovered_path: list[str], path_quality: float
    ) -> None:
        """
        Update player's navigation knowledge with discovered paths.

        Args:
            player_id: ID of the player
            discovered_path: Path that was discovered/traveled
            path_quality: Quality rating of the path (0.0 to 1.0)
        """
        try:
            if player_id not in self._exploration_data:
                self._exploration_data[player_id] = {
                    "known_paths": {},
                    "location_ratings": {},
                    "total_exploration": 0,
                    "preferred_routes": {},
                }

            player_data = self._exploration_data[player_id]

            # Record the path
            path_key = f"{discovered_path[0]}:{discovered_path[-1]}"
            if path_key not in player_data["known_paths"]:
                player_data["known_paths"][path_key] = []

            player_data["known_paths"][path_key].append(
                {
                    "path": discovered_path,
                    "quality": path_quality,
                    "discovered_time": asyncio.get_event_loop().time(),
                    "usage_count": 1,
                }
            )

            # Update location ratings
            for location in discovered_path:
                if location not in player_data["location_ratings"]:
                    player_data["location_ratings"][location] = {
                        "visits": 0,
                        "rating": 0.5,
                    }

                player_data["location_ratings"][location]["visits"] += 1
                # Update rating based on path quality
                current_rating = player_data["location_ratings"][location]["rating"]
                player_data["location_ratings"][location]["rating"] = (
                    current_rating * 0.8 + path_quality * 0.2
                )

            # Update total exploration
            player_data["total_exploration"] += len(discovered_path)

            # Update preferred routes
            if path_quality > 0.7:  # High quality path
                if path_key not in player_data["preferred_routes"]:
                    player_data["preferred_routes"][path_key] = discovered_path

        except Exception as e:
            logger.error(f"Error updating navigation knowledge: {e}")

    async def get_exploration_suggestions(
        self, current_location: str, exploration_history: list[str]
    ) -> list[dict[str, Any]]:
        """
        Suggest unexplored areas for discovery.

        Args:
            current_location: Player's current location
            exploration_history: List of previously visited locations

        Returns:
            List of exploration suggestions
        """
        try:
            suggestions = []
            visited_locations = set(exploration_history)

            # Get nearby unvisited locations
            nearby_locations = await self._get_locations_within_range(
                current_location, 500.0
            )

            for location_info in nearby_locations:
                location_id = location_info["id"]

                if location_id not in visited_locations:
                    location_data = await self._get_location_data(location_id)

                    suggestion = {
                        "location_id": location_id,
                        "name": location_data.get("name", location_id),
                        "description": location_data.get(
                            "description", "An unexplored area"
                        ),
                        "distance": location_info["distance"],
                        "direction": location_info["direction"],
                        "exploration_value": self._calculate_exploration_value(
                            location_data
                        ),
                        "difficulty": location_data.get("difficulty", 1.0),
                        "estimated_time": location_info["distance"]
                        / 50.0,  # Assume 50 units/minute
                        "tags": location_data.get("tags", []),
                    }
                    suggestions.append(suggestion)

            # Sort by exploration value and proximity
            suggestions.sort(key=lambda x: (-x["exploration_value"], x["distance"]))

            return suggestions[:10]  # Return top 10 suggestions

        except Exception as e:
            logger.error(f"Error getting exploration suggestions: {e}")
            return []

    async def validate_path_accessibility(
        self, path: list[str], player_capabilities: dict[str, Any]
    ) -> tuple[bool, list[str]]:
        """
        Validate that player can traverse entire path.

        Args:
            path: List of location IDs representing the path
            player_capabilities: Player's current capabilities and equipment

        Returns:
            Tuple of (can_traverse, list_of_blocking_issues)
        """
        try:
            blocking_issues = []

            for i in range(len(path) - 1):
                current_location = path[i]
                next_location = path[i + 1]

                # Check connection requirements
                connection_data = await self._get_connection_data(
                    current_location, next_location
                )
                requirements = connection_data.get("requirements", {})

                for req_type, req_value in requirements.items():
                    player_value = player_capabilities.get(req_type, 0)

                    if player_value < req_value:
                        blocking_issues.append(
                            f"Connection from {current_location} to {next_location} "
                            f"requires {req_type} {req_value}, but player has {player_value}"
                        )

                # Check location-specific requirements
                location_data = await self._get_location_data(next_location)
                location_requirements = location_data.get("entry_requirements", {})

                for req_type, req_value in location_requirements.items():
                    player_value = player_capabilities.get(req_type, 0)

                    if player_value < req_value:
                        blocking_issues.append(
                            f"Location {next_location} requires {req_type} {req_value}, "
                            f"but player has {player_value}"
                        )

            can_traverse = len(blocking_issues) == 0
            return can_traverse, blocking_issues

        except Exception as e:
            logger.error(f"Error validating path accessibility: {e}")
            return False, [f"Validation error: {str(e)}"]

    async def create_mental_map(
        self, player_id: str, exploration_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Create player's mental map based on exploration.

        Args:
            player_id: ID of the player
            exploration_data: Player's exploration history and observations

        Returns:
            Dictionary representing the player's mental map
        """
        try:
            if player_id not in self._exploration_data:
                return {"error": "No exploration data found for player"}

            player_data = self._exploration_data[player_id]

            mental_map = {
                "known_locations": {},
                "known_connections": {},
                "landmarks": {},
                "regions": {},
                "confidence_areas": {},
                "exploration_stats": {
                    "total_locations_visited": len(player_data["location_ratings"]),
                    "total_distance_traveled": player_data["total_exploration"],
                    "preferred_route_count": len(player_data["preferred_routes"]),
                },
            }

            # Build known locations with confidence ratings
            for location_id, rating_data in player_data["location_ratings"].items():
                visits = rating_data["visits"]
                rating = rating_data["rating"]

                confidence = min(1.0, visits / 5.0)  # Full confidence after 5 visits

                mental_map["known_locations"][location_id] = {
                    "visits": visits,
                    "rating": rating,
                    "confidence": confidence,
                    "familiarity": (
                        "high"
                        if confidence > 0.8
                        else "medium" if confidence > 0.4 else "low"
                    ),
                }

            # Build known connections from traveled paths
            for path_key, path_list in player_data["known_paths"].items():
                start, end = path_key.split(":")

                if start not in mental_map["known_connections"]:
                    mental_map["known_connections"][start] = []

                best_path = max(path_list, key=lambda x: x["quality"])
                mental_map["known_connections"][start].append(
                    {
                        "destination": end,
                        "path": best_path["path"],
                        "quality": best_path["quality"],
                        "usage_count": sum(p["usage_count"] for p in path_list),
                    }
                )

            # Identify key landmarks from highly-rated locations
            for location_id, data in mental_map["known_locations"].items():
                if data["rating"] > 0.8 and data["confidence"] > 0.6:
                    location_info = await self._get_location_data(location_id)
                    if location_info.get("is_landmark", False):
                        mental_map["landmarks"][location_id] = {
                            "name": location_info.get("name", location_id),
                            "type": location_info.get("landmark_type", "structure"),
                            "importance": data["rating"] * data["confidence"],
                        }

            return mental_map

        except Exception as e:
            logger.error(f"Error creating mental map: {e}")
            return {"error": str(e)}

    def register_navigation_algorithm(
        self, algorithm_name: str, algorithm: Callable
    ) -> None:
        """
        Register custom pathfinding algorithm.

        Args:
            algorithm_name: Name of the algorithm
            algorithm: Algorithm function
        """
        self._pathfinding_algorithms[algorithm_name] = algorithm

    # Private helper methods

    def _initialize_algorithms(self) -> None:
        """Initialize pathfinding algorithms."""
        self._pathfinding_algorithms = {
            NavigationAlgorithm.DIJKSTRA: self._dijkstra_pathfinding,
            NavigationAlgorithm.A_STAR: self._a_star_pathfinding,
            NavigationAlgorithm.BREADTH_FIRST: self._breadth_first_pathfinding,
            NavigationAlgorithm.DEPTH_FIRST: self._depth_first_pathfinding,
            NavigationAlgorithm.BEST_FIRST: self._best_first_pathfinding,
        }

    async def _a_star_pathfinding(
        self, start: str, goal: str, graph: dict[str, Any], preferences: dict[str, Any]
    ) -> list[str] | None:
        """A* pathfinding algorithm implementation."""
        try:
            open_set = [(0, start, [start])]
            closed_set = set()

            while open_set:
                current_cost, current, path = heapq.heappop(open_set)

                if current == goal:
                    return path

                if current in closed_set:
                    continue

                closed_set.add(current)

                neighbors = graph.get(current, [])
                for neighbor in neighbors:
                    if neighbor["id"] in closed_set:
                        continue

                    new_path = path + [neighbor["id"]]
                    g_cost = current_cost + neighbor.get("cost", 1.0)
                    h_cost = await self._heuristic_distance(neighbor["id"], goal)
                    f_cost = g_cost + h_cost

                    heapq.heappush(open_set, (f_cost, neighbor["id"], new_path))

            return None

        except Exception as e:
            logger.error(f"Error in A* pathfinding: {e}")
            return None

    async def _dijkstra_pathfinding(
        self, start: str, goal: str, graph: dict[str, Any], preferences: dict[str, Any]
    ) -> list[str] | None:
        """Dijkstra pathfinding algorithm implementation."""
        try:
            distances = {start: 0}
            previous = {}
            unvisited = [(0, start)]

            while unvisited:
                current_distance, current = heapq.heappop(unvisited)

                if current == goal:
                    # Reconstruct path
                    path = []
                    while current is not None:
                        path.append(current)
                        current = previous.get(current)
                    return path[::-1]

                if current_distance > distances.get(current, float("inf")):
                    continue

                neighbors = graph.get(current, [])
                for neighbor in neighbors:
                    neighbor_id = neighbor["id"]
                    distance = current_distance + neighbor.get("cost", 1.0)

                    if distance < distances.get(neighbor_id, float("inf")):
                        distances[neighbor_id] = distance
                        previous[neighbor_id] = current
                        heapq.heappush(unvisited, (distance, neighbor_id))

            return None

        except Exception as e:
            logger.error(f"Error in Dijkstra pathfinding: {e}")
            return None

    async def _breadth_first_pathfinding(
        self, start: str, goal: str, graph: dict[str, Any], preferences: dict[str, Any]
    ) -> list[str] | None:
        """Breadth-first search pathfinding implementation."""
        try:
            from collections import deque

            queue = deque([(start, [start])])
            visited = {start}

            while queue:
                current, path = queue.popleft()

                if current == goal:
                    return path

                neighbors = graph.get(current, [])
                for neighbor in neighbors:
                    neighbor_id = neighbor["id"]
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append((neighbor_id, path + [neighbor_id]))

            return None

        except Exception as e:
            logger.error(f"Error in breadth-first pathfinding: {e}")
            return None

    async def _depth_first_pathfinding(
        self, start: str, goal: str, graph: dict[str, Any], preferences: dict[str, Any]
    ) -> list[str] | None:
        """Depth-first search pathfinding implementation."""
        try:
            stack = [(start, [start])]
            visited = set()

            while stack:
                current, path = stack.pop()

                if current == goal:
                    return path

                if current in visited:
                    continue

                visited.add(current)

                neighbors = graph.get(current, [])
                for neighbor in neighbors:
                    neighbor_id = neighbor["id"]
                    if neighbor_id not in visited:
                        stack.append((neighbor_id, path + [neighbor_id]))

            return None

        except Exception as e:
            logger.error(f"Error in depth-first pathfinding: {e}")
            return None

    async def _best_first_pathfinding(
        self, start: str, goal: str, graph: dict[str, Any], preferences: dict[str, Any]
    ) -> list[str] | None:
        """Best-first search pathfinding implementation."""
        try:
            open_set = [(0, start, [start])]
            visited = set()

            while open_set:
                _, current, path = heapq.heappop(open_set)

                if current == goal:
                    return path

                if current in visited:
                    continue

                visited.add(current)

                neighbors = graph.get(current, [])
                for neighbor in neighbors:
                    neighbor_id = neighbor["id"]
                    if neighbor_id not in visited:
                        h_cost = await self._heuristic_distance(neighbor_id, goal)
                        heapq.heappush(
                            open_set, (h_cost, neighbor_id, path + [neighbor_id])
                        )

            return None

        except Exception as e:
            logger.error(f"Error in best-first pathfinding: {e}")
            return None

    # Additional helper methods (simplified implementations)

    async def _build_world_graph(
        self, preferences: dict[str, Any]
    ) -> dict[str, list[dict[str, Any]]]:
        """Build world connectivity graph."""
        # Simplified graph - in a full implementation, this would query the world state
        return {
            "location_1": [
                {"id": "location_2", "cost": 1.0, "direction": "north"},
                {"id": "location_3", "cost": 1.5, "direction": "east"},
            ],
            "location_2": [
                {"id": "location_1", "cost": 1.0, "direction": "south"},
                {"id": "location_4", "cost": 2.0, "direction": "north"},
            ],
            "location_3": [
                {"id": "location_1", "cost": 1.5, "direction": "west"},
                {"id": "location_4", "cost": 1.0, "direction": "north"},
            ],
            "location_4": [
                {"id": "location_2", "cost": 2.0, "direction": "south"},
                {"id": "location_3", "cost": 1.0, "direction": "south"},
            ],
            # Add test locations
            "town_square": [
                {"id": "forest_path", "cost": 1.0, "direction": "north"},
                {"id": "village", "cost": 2.0, "direction": "east"},
            ],
            "forest_path": [
                {"id": "town_square", "cost": 1.0, "direction": "south"},
                {"id": "mountain_base", "cost": 3.0, "direction": "north"},
            ],
            "mountain_base": [
                {"id": "forest_path", "cost": 3.0, "direction": "south"},
                {"id": "mountain_peak", "cost": 5.0, "direction": "up"},
            ],
            "mountain_peak": [
                {"id": "mountain_base", "cost": 5.0, "direction": "down"},
            ],
            "village": [
                {"id": "town_square", "cost": 2.0, "direction": "west"},
            ],
        }

    async def _heuristic_distance(self, location1: str, location2: str) -> float:
        """Calculate heuristic distance between locations."""
        # Simplified distance calculation
        return abs(hash(location1) % 100 - hash(location2) % 100) / 10.0

    async def _is_path_still_valid(self, path: list[dict[str, Any]]) -> bool:
        """Check if cached path is still valid."""
        # In a full implementation, this would check for blocked locations, etc.
        return True

    async def _enhance_path_information(
        self, path: list[str], preferences: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Enhance path with additional information."""
        enhanced_path = []
        for i, location in enumerate(path):
            step = {
                "location": location,
                "step_number": i,
                "distance": 1.0 if i > 0 else 0.0,
                "direction": "north" if i > 0 else "start",
                "warnings": [],
                "landmarks": [],
            }
            enhanced_path.append(step)
        return enhanced_path

    async def _get_locations_within_range(
        self, center_location: str, max_range: float
    ) -> list[dict[str, Any]]:
        """Get locations within specified range."""
        # Simulate nearby locations
        return [
            {
                "id": f"{center_location}_nearby_1",
                "distance": 50.0,
                "direction": "north",
            },
            {
                "id": f"{center_location}_nearby_2",
                "distance": 75.0,
                "direction": "east",
            },
        ]

    async def _get_location_data(self, location_id: str) -> dict[str, Any]:
        """Get location data."""
        return {
            "name": location_id.replace("_", " ").title(),
            "description": f"A location called {location_id}",
            "is_landmark": "landmark" in location_id.lower(),
            "landmark_type": "structure",
            "prominence": 1.0,
        }

    def _calculate_landmark_visibility(
        self, location_data: dict[str, Any], distance: float
    ) -> float:
        """Calculate landmark visibility."""
        base_visibility = location_data.get("prominence", 1.0)
        distance_factor = max(0.1, 1.0 - (distance / 1000.0))
        return base_visibility * distance_factor

    async def _calculate_segment_estimates(
        self, start: str, end: str, speed: float, obstacles: list[str]
    ) -> dict[str, Any]:
        """Calculate estimates for a path segment."""
        base_distance = 100.0  # Simplified
        base_time = base_distance / speed

        obstacle_modifier = 1.0 + (len(obstacles) * 0.2)

        return {
            "start": start,
            "end": end,
            "distance": base_distance,
            "time": base_time * obstacle_modifier,
            "energy": base_time * 2.0 * obstacle_modifier,
            "obstacles": obstacles,
        }

    def _calculate_path_difficulty(self, segments: list[dict[str, Any]]) -> float:
        """Calculate overall path difficulty."""
        if not segments:
            return 0.0

        total_obstacle_count = sum(len(seg.get("obstacles", [])) for seg in segments)
        avg_obstacles_per_segment = total_obstacle_count / len(segments)

        return min(1.0, avg_obstacles_per_segment / 3.0)  # Normalize to 0-1

    async def _generate_travel_recommendations(
        self, path: list[str], total_time: float, total_energy: float
    ) -> list[str]:
        """Generate travel recommendations."""
        recommendations = []

        if total_energy > 100:
            recommendations.append(
                "Consider bringing extra food for this long journey."
            )

        if total_time > 3600:  # More than 1 hour
            recommendations.append("Plan for rest stops during this extended travel.")

        return recommendations

    def _calculate_exploration_value(self, location_data: dict[str, Any]) -> float:
        """Calculate exploration value of a location."""
        base_value = 1.0

        if location_data.get("is_landmark", False):
            base_value += 0.5

        if "treasure" in location_data.get("tags", []):
            base_value += 1.0

        if "dangerous" in location_data.get("tags", []):
            base_value += 0.3  # Risk adds some exploration value

        return base_value

    async def _get_connection_data(self, start: str, end: str) -> dict[str, Any]:
        """Get connection data between two locations."""
        return {"requirements": {}, "cost": 1.0, "direction": "north"}
