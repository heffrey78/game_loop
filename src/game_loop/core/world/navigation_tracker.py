"""
Player navigation tracker for breadcrumb trails and landmark-based navigation.

This module tracks player movement and provides navigation assistance.
"""

import logging
from datetime import datetime
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)


class PlayerNavigationTracker:
    """Track player movement and provide navigation assistance."""

    def __init__(self, state_manager: Any) -> None:
        self.state_manager = state_manager
        self.location_history: list[dict[str, Any]] = []
        self.visited_locations: set[UUID] = set()
        self.landmarks: dict[str, UUID] = {}
        self.max_history = 50  # Keep last 50 movements

    async def track_movement(
        self, player_id: UUID, from_location: UUID, to_location: UUID, direction: str
    ) -> None:
        """Record player movement for breadcrumb system."""
        try:
            movement_record = {
                "player_id": player_id,
                "from_location": from_location,
                "to_location": to_location,
                "direction": direction,
                "timestamp": datetime.now(),
            }

            self.location_history.append(movement_record)
            self.visited_locations.add(to_location)

            # Trim history if too long
            if len(self.location_history) > self.max_history:
                self.location_history = self.location_history[-self.max_history :]

            # Check if destination should be marked as landmark
            await self._check_for_landmark(to_location)

            # Persist to database for session recovery
            await self._persist_navigation_data(player_id)

        except Exception as e:
            logger.error(f"Error tracking movement: {e}")

    async def _check_for_landmark(self, location_id: UUID) -> None:
        """Check if location should be registered as a landmark."""
        try:
            location = await self.state_manager.get_location_details(location_id)
            if not location:
                return

            location_name = getattr(location, "name", "").lower()

            # Define landmark keywords
            landmark_keywords = {
                "entrance": ["reception", "lobby", "entrance", "foyer", "entry"],
                "hub": ["central", "main", "hub", "plaza", "atrium"],
                "unique": ["library", "cafeteria", "garden", "workshop", "office"],
                "vertical": ["stairwell", "elevator", "stairs", "staircase"],
                "special": ["security", "server", "archive", "vault"],
            }

            # Check if location matches landmark criteria
            for landmark_type, keywords in landmark_keywords.items():
                for keyword in keywords:
                    if keyword in location_name:
                        landmark_key = f"{landmark_type}_{keyword}"
                        self.landmarks[landmark_key] = location_id
                        self.landmarks[keyword] = location_id  # Also store by keyword

                        # If this is an entrance/lobby, mark as primary landmark
                        if keyword in ["reception", "lobby", "entrance"]:
                            self.landmarks["start"] = location_id
                            self.landmarks["beginning"] = location_id

                        logger.info(f"Registered landmark: {keyword} -> {location_id}")
                        break

        except Exception as e:
            logger.error(f"Error checking for landmark: {e}")

    async def get_return_path(
        self, player_id: UUID, target_location_name: str | None = None
    ) -> list[dict[str, Any]] | None:
        """Get path back to specific location or starting point."""
        try:
            current_location = await self._get_current_location(player_id)
            if not current_location:
                return None

            # Determine target location
            target_location = None

            if target_location_name:
                # Look for landmark by name
                target_location = await self._find_landmark_by_name(
                    target_location_name
                )

            if not target_location:
                # Default to starting location
                target_location = self.landmarks.get("start")
                if not target_location:
                    # Find the first location in history as fallback
                    if self.location_history:
                        target_location = self.location_history[0]["from_location"]

            if not target_location or target_location == current_location:
                return None

            # Use connection manager to find path
            if hasattr(self.state_manager, "connection_manager"):
                path = await self.state_manager.connection_manager.find_path_between_locations(
                    current_location, target_location
                )
                return path

            return None

        except Exception as e:
            logger.error(f"Error getting return path: {e}")
            return None

    async def _find_landmark_by_name(self, name: str) -> UUID | None:
        """Find landmark location by name or keyword."""
        name_lower = name.lower()

        # Direct lookup
        if name_lower in self.landmarks:
            return self.landmarks[name_lower]

        # Partial match
        for landmark_key, location_id in self.landmarks.items():
            if name_lower in landmark_key or landmark_key in name_lower:
                return location_id

        return None

    async def _get_current_location(self, player_id: UUID) -> UUID | None:
        """Get current location for player."""
        try:
            player_state = await self.state_manager.player_tracker.get_player_state(
                player_id
            )
            return (
                getattr(player_state, "current_location_id", None)
                if player_state
                else None
            )
        except Exception as e:
            logger.error(f"Error getting current location: {e}")
            return None

    def get_breadcrumb_trail(self, steps_back: int = 5) -> list[dict[str, Any]]:
        """Get recent movement history for retracing steps."""
        if not self.location_history:
            return []

        return self.location_history[-steps_back:]

    def get_reverse_directions(self, steps_back: int = 5) -> list[str]:
        """Get reverse directions for retracing steps."""
        trail = self.get_breadcrumb_trail(steps_back)
        reverse_directions = []

        direction_reverses = {
            "north": "south",
            "south": "north",
            "east": "west",
            "west": "east",
            "up": "down",
            "down": "up",
        }

        # Reverse the trail and get opposite directions
        for movement in reversed(trail):
            direction = movement["direction"]
            reverse_dir = direction_reverses.get(direction, direction)
            reverse_directions.append(reverse_dir)

        return reverse_directions

    async def handle_navigation_command(
        self, command: str, player_id: UUID
    ) -> dict[str, Any] | None:
        """Handle navigation commands like 'go to reception' or 'retrace steps'."""
        try:
            command_lower = command.lower()

            # Handle retrace commands
            if any(
                word in command_lower
                for word in ["retrace", "back", "return", "backtrack"]
            ):
                return await self._handle_retrace_command(command_lower, player_id)

            # Handle landmark navigation
            elif any(
                phrase in command_lower
                for phrase in ["go to", "head to", "navigate to", "find"]
            ):
                return await self._handle_landmark_navigation(command_lower, player_id)

            return None

        except Exception as e:
            logger.error(f"Error handling navigation command: {e}")
            return None

    async def _handle_retrace_command(
        self, command: str, player_id: UUID
    ) -> dict[str, Any] | None:
        """Handle commands to retrace steps."""
        try:
            # Extract number of steps if specified
            steps = 1
            words = command.split()
            for i, word in enumerate(words):
                if word.isdigit():
                    steps = int(word)
                    break
                elif word in ["few", "several"]:
                    steps = 3
                    break

            reverse_directions = self.get_reverse_directions(steps)
            if not reverse_directions:
                return {"success": False, "message": "No recent movement to retrace."}

            return {
                "success": True,
                "type": "retrace",
                "directions": reverse_directions,
                "message": f"To retrace your steps: {', '.join(reverse_directions)}",
            }

        except Exception as e:
            logger.error(f"Error handling retrace command: {e}")
            return None

    async def _handle_landmark_navigation(
        self, command: str, player_id: UUID
    ) -> dict[str, Any] | None:
        """Handle landmark navigation commands."""
        try:
            # Extract target from command
            target_keywords = self._extract_navigation_keywords(command)

            if not target_keywords:
                return {
                    "success": False,
                    "message": "I don't recognize that location. Try 'go to reception' or 'go to lobby'.",
                }

            # Find matching landmark
            target_location = None
            matched_keyword = None

            for keyword in target_keywords:
                target_location = await self._find_landmark_by_name(keyword)
                if target_location:
                    matched_keyword = keyword
                    break

            if not target_location:
                available_landmarks = list(self.landmarks.keys())
                return {
                    "success": False,
                    "message": f"I don't know how to get to '{target_keywords[0]}'. Available locations: {', '.join(available_landmarks[:5])}",
                }

            # Get path to landmark
            path = await self.get_return_path(player_id, matched_keyword)
            if not path:
                return {
                    "success": False,
                    "message": f"I can't find a path to {matched_keyword}.",
                }

            # Convert path to directions
            directions = [step["direction"] for step in path]

            return {
                "success": True,
                "type": "landmark_navigation",
                "target": matched_keyword,
                "directions": directions,
                "message": f"To reach {matched_keyword}: {', '.join(directions)}",
            }

        except Exception as e:
            logger.error(f"Error handling landmark navigation: {e}")
            return None

    def _extract_navigation_keywords(self, command: str) -> list[str]:
        """Extract location keywords from navigation commands."""
        keywords = []
        command_lower = command.lower()

        # Remove navigation command words
        nav_words = ["go to", "head to", "navigate to", "find", "return to", "back to"]
        for nav_word in nav_words:
            command_lower = command_lower.replace(nav_word, "").strip()

        # Extract potential location words
        words = command_lower.split()

        # Common location keywords
        location_keywords = [
            "reception",
            "lobby",
            "entrance",
            "foyer",
            "start",
            "beginning",
            "library",
            "cafeteria",
            "garden",
            "workshop",
            "office",
            "stairwell",
            "elevator",
            "stairs",
            "security",
            "archive",
        ]

        for word in words:
            if word in location_keywords:
                keywords.append(word)

        # If no specific keywords found, try the whole remaining phrase
        if not keywords and command_lower.strip():
            keywords.append(command_lower.strip())

        return keywords

    async def _persist_navigation_data(self, player_id: UUID) -> None:
        """Persist navigation data to database for session recovery."""
        try:
            # Store in player state as JSON
            navigation_data = {
                "landmarks": {key: str(value) for key, value in self.landmarks.items()},
                "visited_locations": [str(loc_id) for loc_id in self.visited_locations],
                "recent_history": self.location_history[
                    -10:
                ],  # Store last 10 movements
            }

            # This would be stored in player state JSON field
            # Implementation depends on player state storage system
            # For now, just log the data
            logger.debug(f"Navigation data for player {player_id}: {navigation_data}")

        except Exception as e:
            logger.error(f"Error persisting navigation data: {e}")

    async def load_navigation_data(self, player_id: UUID) -> None:
        """Load navigation data from database for session recovery."""
        try:
            # Load from player state JSON field
            # Implementation depends on player state storage system
            # This would restore landmarks and visited locations
            logger.debug(f"Loading navigation data for player {player_id}")

        except Exception as e:
            logger.error(f"Error loading navigation data: {e}")

    def get_navigation_summary(self) -> dict[str, Any]:
        """Get summary of current navigation state."""
        return {
            "total_movements": len(self.location_history),
            "visited_locations": len(self.visited_locations),
            "landmarks_discovered": len(self.landmarks),
            "landmark_names": list(self.landmarks.keys()),
            "recent_movements": len(self.get_breadcrumb_trail(5)),
        }
