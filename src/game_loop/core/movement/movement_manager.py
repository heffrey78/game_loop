"""
Movement Manager for handling player and entity movement through the game world.

This module provides movement validation, location transitions, pathfinding,
and navigation assistance for entities in the game world.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from game_loop.core.command_handlers.physical_action_processor import (
    PhysicalActionResult,
    PhysicalActionType,
)

logger = logging.getLogger(__name__)


class MovementManager:
    """
    Handle player and entity movement through the game world.
    
    This class manages direction parsing, location transitions, pathfinding,
    movement constraints, and travel calculations.
    """

    def __init__(
        self,
        world_state_manager: Any = None,
        location_service: Any = None,
        physics_engine: Any = None,
    ):
        """
        Initialize the movement manager.

        Args:
            world_state_manager: Manager for world state access and updates
            location_service: Service for location data and management
            physics_engine: Physics engine for movement constraints
        """
        self.world_state = world_state_manager
        self.locations = location_service
        self.physics = physics_engine
        self._movement_cache: Dict[str, Any] = {}
        self._pathfinding_cache: Dict[str, List[str]] = {}
        self._direction_aliases = self._initialize_direction_aliases()

    async def process_movement_command(
        self, player_id: str, direction: str, context: Dict[str, Any]
    ) -> PhysicalActionResult:
        """
        Process a movement command from the player.

        Args:
            player_id: ID of the player attempting to move
            direction: Direction of movement (north, south, etc.)
            context: Current game context and state

        Returns:
            PhysicalActionResult containing movement outcome
        """
        try:
            # Get player's current location
            player_state = context.get("player_state", {})
            current_location = player_state.get("current_location")
            
            if not current_location:
                return PhysicalActionResult(
                    success=False,
                    action_type=PhysicalActionType.MOVEMENT,
                    affected_entities=[player_id],
                    state_changes={},
                    energy_cost=0.0,
                    time_elapsed=0.0,
                    side_effects=[],
                    description="Movement failed.",
                    error_message="Current location not found.",
                )

            # Parse and normalize direction
            normalized_direction = self.parse_direction_input(direction)
            if not normalized_direction:
                return PhysicalActionResult(
                    success=False,
                    action_type=PhysicalActionType.MOVEMENT,
                    affected_entities=[player_id],
                    state_changes={},
                    energy_cost=0.0,
                    time_elapsed=0.0,
                    side_effects=[],
                    description="Movement failed.",
                    error_message=f"Invalid direction: {direction}",
                )

            # Get available exits
            available_exits = await self.get_available_exits(current_location, player_state)
            
            # Find matching exit
            target_location = None
            for exit_info in available_exits:
                if exit_info["direction"] == normalized_direction:
                    target_location = exit_info["target_location"]
                    break

            if not target_location:
                return PhysicalActionResult(
                    success=False,
                    action_type=PhysicalActionType.MOVEMENT,
                    affected_entities=[player_id],
                    state_changes={},
                    energy_cost=0.0,
                    time_elapsed=0.0,
                    side_effects=[],
                    description=f"You cannot go {normalized_direction} from here.",
                    error_message=f"No exit {normalized_direction} from {current_location}",
                )

            # Validate movement
            is_valid, error_msg = await self.validate_movement(
                current_location, target_location, player_state
            )

            if not is_valid:
                return PhysicalActionResult(
                    success=False,
                    action_type=PhysicalActionType.MOVEMENT,
                    affected_entities=[player_id],
                    state_changes={},
                    energy_cost=0.0,
                    time_elapsed=0.0,
                    side_effects=[],
                    description="Movement blocked.",
                    error_message=error_msg,
                )

            # Calculate travel time and energy cost
            travel_time = await self.calculate_travel_time(
                current_location, target_location, "walking"
            )
            energy_cost = self._calculate_movement_energy_cost(travel_time, normalized_direction)

            # Check movement obstacles
            obstacles = await self.check_movement_obstacles(
                current_location, normalized_direction, player_state
            )

            if obstacles:
                return PhysicalActionResult(
                    success=False,
                    action_type=PhysicalActionType.MOVEMENT,
                    affected_entities=[player_id],
                    state_changes={},
                    energy_cost=0.0,
                    time_elapsed=0.0,
                    side_effects=[],
                    description=f"Your path is blocked: {', '.join(obstacles)}",
                    error_message=f"Obstacles prevent movement: {obstacles}",
                )

            # Execute the movement
            transition_result = await self.handle_location_transition(
                player_id, current_location, target_location
            )

            # Apply movement effects
            movement_result = PhysicalActionResult(
                success=True,
                action_type=PhysicalActionType.MOVEMENT,
                affected_entities=[player_id],
                state_changes={
                    "player_location": target_location,
                    "previous_location": current_location,
                    **transition_result.get("state_changes", {}),
                },
                energy_cost=energy_cost,
                time_elapsed=travel_time,
                side_effects=transition_result.get("side_effects", []),
                description=f"You move {normalized_direction} to {target_location}.",
            )

            await self.apply_movement_effects(player_id, movement_result)
            return movement_result

        except Exception as e:
            logger.error(f"Error processing movement command: {e}")
            return PhysicalActionResult(
                success=False,
                action_type=PhysicalActionType.MOVEMENT,
                affected_entities=[player_id],
                state_changes={},
                energy_cost=0.0,
                time_elapsed=0.0,
                side_effects=[],
                description="Movement failed due to an error.",
                error_message=str(e),
            )

    async def validate_movement(
        self, from_location: str, to_location: str, player_state: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that movement between locations is possible.

        Args:
            from_location: Starting location ID
            to_location: Target location ID
            player_state: Current player state

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check if locations exist
            if not from_location or not to_location:
                return False, "Invalid location specified."

            # Check location capacity
            location_full = not await self._validate_location_capacity(to_location)
            if location_full:
                return False, f"Location {to_location} is at capacity."

            # Check player movement ability
            if player_state.get("movement_disabled", False):
                return False, "Movement is currently disabled."

            # Check energy requirements
            if player_state.get("energy", 100) < 5:  # Minimum energy for movement
                return False, "Insufficient energy to move."

            # Check for movement restrictions
            movement_restrictions = player_state.get("movement_restrictions", [])
            if "all" in movement_restrictions:
                return False, "All movement is restricted."

            if to_location in movement_restrictions:
                return False, f"Movement to {to_location} is restricted."

            return True, None

        except Exception as e:
            logger.error(f"Error validating movement: {e}")
            return False, f"Movement validation error: {str(e)}"

    async def find_path(
        self,
        start_location: str,
        target_location: str,
        constraints: Dict[str, Any],
    ) -> Optional[List[str]]:
        """
        Find a path between two locations.

        Args:
            start_location: Starting location ID
            target_location: Target location ID
            constraints: Movement constraints and preferences

        Returns:
            List of location IDs representing the path, or None if no path found
        """
        try:
            # Check cache first
            cache_key = f"{start_location}:{target_location}:{hash(str(constraints))}"
            if cache_key in self._pathfinding_cache:
                return self._pathfinding_cache[cache_key]

            # Simple pathfinding implementation (BFS)
            # In a full implementation, this would use A* or similar algorithms
            if start_location == target_location:
                return [start_location]

            visited = set()
            queue = [(start_location, [start_location])]
            max_depth = 10  # Prevent infinite loops
            
            while queue:
                current_location, path = queue.pop(0)
                
                # Prevent infinite pathfinding
                if len(path) > max_depth:
                    continue
                
                if current_location in visited:
                    continue
                    
                visited.add(current_location)
                
                # Get connected locations
                exits = await self._get_location_exits(current_location)
                
                for exit_info in exits:
                    next_location = exit_info.get("target_location")
                    if not next_location:
                        continue
                        
                    if next_location == target_location:
                        final_path = path + [next_location]
                        self._pathfinding_cache[cache_key] = final_path
                        return final_path
                    
                    if next_location not in visited and len(path) < max_depth:
                        queue.append((next_location, path + [next_location]))

            # No path found
            return None

        except Exception as e:
            logger.error(f"Error finding path: {e}")
            return None

    async def calculate_travel_time(
        self, from_location: str, to_location: str, movement_type: str
    ) -> float:
        """
        Calculate time required for movement.

        Args:
            from_location: Starting location ID
            to_location: Target location ID
            movement_type: Type of movement (walking, running, etc.)

        Returns:
            Travel time in seconds
        """
        try:
            # Base travel times by movement type
            base_times = {
                "walking": 5.0,
                "running": 3.0,
                "crawling": 10.0,
                "swimming": 8.0,
                "climbing": 15.0,
            }

            base_time = base_times.get(movement_type, 5.0)

            # Add modifiers based on location characteristics
            # In a full implementation, this would consider terrain, weather, etc.
            terrain_modifier = 1.0
            
            # Simulate different terrain difficulties
            if "mountain" in to_location.lower():
                terrain_modifier = 1.5
            elif "swamp" in to_location.lower():
                terrain_modifier = 2.0
            elif "road" in to_location.lower():
                terrain_modifier = 0.8

            return base_time * terrain_modifier

        except Exception as e:
            logger.error(f"Error calculating travel time: {e}")
            return 5.0  # Default travel time

    async def get_available_exits(
        self, location_id: str, player_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Get list of available exits from current location.

        Args:
            location_id: Current location ID
            player_state: Current player state

        Returns:
            List of exit information dictionaries
        """
        try:
            exits = await self._get_location_exits(location_id)
            available_exits = []

            for exit_info in exits:
                # Check if player can use this exit
                can_use = True
                
                # Check access requirements
                requirements = exit_info.get("requirements", {})
                if requirements:
                    # Check if player meets requirements
                    for req_type, req_value in requirements.items():
                        player_value = player_state.get(req_type, 0)
                        if player_value < req_value:
                            can_use = False
                            break

                # Check if exit is blocked
                if exit_info.get("blocked", False):
                    can_use = False

                if can_use:
                    available_exits.append(exit_info)

            return available_exits

        except Exception as e:
            logger.error(f"Error getting available exits: {e}")
            return []

    async def handle_location_transition(
        self, player_id: str, from_location: str, to_location: str
    ) -> Dict[str, Any]:
        """
        Handle the transition between locations.

        Args:
            player_id: ID of the player moving
            from_location: Starting location ID
            to_location: Target location ID

        Returns:
            Dictionary containing transition results
        """
        try:
            transition_result = {
                "success": True,
                "state_changes": {},
                "side_effects": [],
                "events": [],
            }

            # Update location state
            await self._update_location_state(
                from_location, {"departed_players": [player_id]}
            )
            await self._update_location_state(
                to_location, {"arrived_players": [player_id]}
            )

            # Add location-specific effects
            location_effects = await self._get_location_arrival_effects(to_location)
            transition_result["side_effects"].extend(location_effects)

            # Check for location-based events
            location_events = await self._check_location_events(to_location, player_id)
            transition_result["events"].extend(location_events)

            return transition_result

        except Exception as e:
            logger.error(f"Error handling location transition: {e}")
            return {
                "success": False,
                "error": str(e),
                "state_changes": {},
                "side_effects": [],
                "events": [],
            }

    async def check_movement_obstacles(
        self, location_id: str, direction: str, player_state: Dict[str, Any]
    ) -> List[str]:
        """
        Check for obstacles preventing movement.

        Args:
            location_id: Current location ID
            direction: Direction of intended movement
            player_state: Current player state

        Returns:
            List of obstacle descriptions
        """
        try:
            obstacles = []

            # Check for physical obstacles
            location_obstacles = await self._get_location_obstacles(location_id, direction)
            obstacles.extend(location_obstacles)

            # Check for ability-based obstacles
            if direction in ["up"] and not player_state.get("can_climb", True):
                obstacles.append("You cannot climb in your current state.")

            if direction in ["down"] and player_state.get("afraid_of_heights", False):
                obstacles.append("Your fear of heights prevents you from going down.")

            # Check for item-based obstacles
            required_items = await self._get_movement_required_items(location_id, direction)
            for item in required_items:
                if item not in player_state.get("inventory", []):
                    obstacles.append(f"You need a {item} to go {direction}.")

            return obstacles

        except Exception as e:
            logger.error(f"Error checking movement obstacles: {e}")
            return [f"Error checking obstacles: {str(e)}"]

    async def apply_movement_effects(
        self, player_id: str, movement_result: PhysicalActionResult
    ) -> None:
        """
        Apply effects of movement on player state.

        Args:
            player_id: ID of the player who moved
            movement_result: Result of the movement action
        """
        try:
            if self.world_state:
                # Update player location in world state
                new_location = movement_result.state_changes.get("player_location")
                if new_location:
                    # In a full implementation, this would update the world state
                    logger.info(f"Player {player_id} moved to {new_location}")

                # Apply energy cost
                if movement_result.energy_cost > 0:
                    # In a full implementation, this would reduce player energy
                    logger.info(f"Player {player_id} expended {movement_result.energy_cost} energy")

        except Exception as e:
            logger.error(f"Error applying movement effects: {e}")

    def parse_direction_input(self, direction_input: str) -> Optional[str]:
        """
        Parse player direction input into standardized direction.

        Args:
            direction_input: Raw direction input from player

        Returns:
            Standardized direction string or None if invalid
        """
        normalized = direction_input.lower().strip()
        return self._direction_aliases.get(normalized)

    async def _update_location_state(
        self, location_id: str, changes: Dict[str, Any]
    ) -> None:
        """Update location state after movement events."""
        try:
            if self.locations:
                # In a full implementation, this would update location state
                logger.info(f"Location {location_id} state updated: {changes}")
        except Exception as e:
            logger.error(f"Error updating location state: {e}")

    async def _validate_location_capacity(self, location_id: str) -> bool:
        """Check if location can accommodate another entity."""
        try:
            # In a full implementation, this would check actual location capacity
            # For now, assume all locations can accommodate players
            return True
        except Exception as e:
            logger.error(f"Error validating location capacity: {e}")
            return False

    def _normalize_direction(self, direction: str) -> str:
        """Normalize direction string to standard format."""
        return self.parse_direction_input(direction) or direction.lower()

    def _initialize_direction_aliases(self) -> Dict[str, str]:
        """Initialize direction aliases and synonyms."""
        return {
            # Cardinal directions
            "north": "north", "n": "north",
            "south": "south", "s": "south",
            "east": "east", "e": "east",
            "west": "west", "w": "west",
            
            # Ordinal directions
            "northeast": "northeast", "ne": "northeast",
            "northwest": "northwest", "nw": "northwest",
            "southeast": "southeast", "se": "southeast",
            "southwest": "southwest", "sw": "southwest",
            
            # Vertical directions
            "up": "up", "u": "up", "upward": "up", "upwards": "up",
            "down": "down", "d": "down", "downward": "down", "downwards": "down",
            
            # Special directions
            "in": "in", "into": "in", "inside": "in",
            "out": "out", "outside": "out", "exit": "out",
            "forward": "forward", "forth": "forward", "ahead": "forward",
            "back": "back", "backward": "back", "backwards": "back",
            "left": "left", "right": "right",
        }

    def _calculate_movement_energy_cost(
        self, travel_time: float, direction: str
    ) -> float:
        """Calculate energy cost for movement."""
        base_cost = 5.0
        
        # Different directions have different energy costs
        direction_modifiers = {
            "up": 2.0,      # Climbing costs more energy
            "down": 1.2,    # Going down costs slightly more due to care needed
            "north": 1.0,   # Standard directions
            "south": 1.0,
            "east": 1.0,
            "west": 1.0,
        }
        
        modifier = direction_modifiers.get(direction, 1.0)
        time_factor = max(1.0, travel_time / 5.0)  # Longer travel = more energy
        
        return base_cost * modifier * time_factor

    async def _get_location_exits(self, location_id: str) -> List[Dict[str, Any]]:
        """Get exits from a location."""
        # Simulate location exits - in a full implementation, this would
        # query the location service or world state
        
        # Create a finite set of known locations to prevent infinite expansion
        known_locations = {
            "forest_clearing": ["forest_path", "mountain_base"],
            "forest_path": ["forest_clearing", "village"],
            "mountain_base": ["forest_clearing", "mountain_peak"],
            "village": ["forest_path", "town_square"],
            "mountain_peak": ["mountain_base"],
            "town_square": ["village"],
        }
        
        # Get connected locations for this location
        connected = known_locations.get(location_id, [])
        
        exits = []
        for i, target in enumerate(connected):
            direction = ["north", "south", "east", "west"][i % 4]
            exits.append({
                "direction": direction,
                "target_location": target,
                "description": f"A path leading {direction}",
                "requirements": {},
                "blocked": False,
            })
        
        return exits

    async def _get_location_arrival_effects(self, location_id: str) -> List[str]:
        """Get effects that happen when arriving at a location."""
        effects = []
        
        # Sample location-based effects
        if "dark" in location_id.lower():
            effects.append("The area is quite dark, making it hard to see.")
        if "cold" in location_id.lower():
            effects.append("A chill runs through you as you enter this cold area.")
        if "warm" in location_id.lower():
            effects.append("The warmth of this area is welcoming.")
            
        return effects

    async def _check_location_events(
        self, location_id: str, player_id: str
    ) -> List[Dict[str, Any]]:
        """Check for events triggered by arriving at a location."""
        events = []
        
        # Sample events
        if "treasure" in location_id.lower():
            events.append({
                "type": "discovery",
                "description": "You notice something glinting in the corner.",
                "trigger": "arrival",
            })
            
        return events

    async def _get_location_obstacles(
        self, location_id: str, direction: str
    ) -> List[str]:
        """Get obstacles in a specific direction from a location."""
        obstacles = []
        
        # Sample obstacles
        if direction == "north" and "blocked" in location_id.lower():
            obstacles.append("A fallen tree blocks the northern path.")
        if direction == "up" and "no_climb" in location_id.lower():
            obstacles.append("The walls are too smooth to climb.")
            
        return obstacles

    async def _get_movement_required_items(
        self, location_id: str, direction: str
    ) -> List[str]:
        """Get items required for movement in a specific direction."""
        required_items = []
        
        # Sample requirements
        if direction == "up" and "cliff" in location_id.lower():
            required_items.append("rope")
        if direction in ["north", "south"] and "locked_gate" in location_id.lower():
            required_items.append("key")
            
        return required_items