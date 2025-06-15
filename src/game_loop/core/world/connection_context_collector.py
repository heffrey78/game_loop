"""
Connection Context Collector for World Connection Management System.

This module gathers contextual information needed for intelligent connection
generation, including geographic analysis, existing patterns, and narrative context.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from game_loop.core.models.connection_models import ConnectionGenerationContext
from game_loop.database.session_factory import DatabaseSessionFactory
from game_loop.state.models import Location, WorldState

logger = logging.getLogger(__name__)


class ConnectionContextCollector:
    """Gathers contextual information for intelligent connection generation."""

    def __init__(
        self, world_state: WorldState, session_factory: DatabaseSessionFactory
    ):
        """Initialize context collector."""
        self.world_state = world_state
        self.session_factory = session_factory

    async def collect_generation_context(
        self, source_location_id: UUID, target_location_id: UUID, purpose: str
    ) -> ConnectionGenerationContext:
        """Collect comprehensive context for connection generation."""
        try:
            # Retrieve locations
            source_location = self.world_state.locations.get(source_location_id)
            target_location = self.world_state.locations.get(target_location_id)

            if not source_location or not target_location:
                raise ValueError(
                    f"Location not found: source={source_location_id}, target={target_location_id}"
                )

            # Analyze geographic relationship
            geographic_analysis = await self.analyze_geographic_relationship(
                source_location, target_location
            )

            # Get existing connection patterns
            connection_patterns = await self.get_existing_connection_patterns(
                source_location_id
            )

            # Determine narrative requirements
            narrative_requirements = await self.determine_narrative_requirements(
                source_location, target_location, purpose
            )

            # Analyze terrain constraints
            terrain_constraints = self._analyze_terrain_constraints(
                source_location, target_location, geographic_analysis
            )

            # Determine distance preference
            distance_preference = self._determine_distance_preference(
                geographic_analysis, purpose
            )

            # Get existing connections for context
            existing_connections = await self._get_existing_connections_list(
                source_location_id
            )

            # Create generation context
            context = ConnectionGenerationContext(
                source_location=source_location,
                target_location=target_location,
                generation_purpose=purpose,
                distance_preference=distance_preference,
                terrain_constraints=terrain_constraints,
                narrative_context=narrative_requirements,
                existing_connections=existing_connections,
                world_state_snapshot=self._create_world_snapshot(),
            )

            logger.info(
                f"Collected connection context: {source_location.name} -> {target_location.name} ({purpose})"
            )
            return context

        except Exception as e:
            logger.error(f"Error collecting generation context: {e}")
            # Create minimal fallback context
            return self._create_fallback_context(
                source_location_id, target_location_id, purpose
            )

    async def analyze_geographic_relationship(
        self, source: Location, target: Location
    ) -> dict[str, Any]:
        """Analyze geographic relationship between locations."""
        try:
            analysis = {
                "estimated_distance": "medium",
                "terrain_compatibility": 0.7,
                "elevation_change": "minimal",
                "natural_barriers": [],
                "connection_feasibility": 0.8,
            }

            # Get theme information
            source_theme = source.state_flags.get("theme", "Unknown")
            target_theme = target.state_flags.get("theme", "Unknown")

            # Analyze theme compatibility
            theme_compatibility = self._calculate_theme_compatibility(
                source_theme, target_theme
            )
            analysis["theme_compatibility"] = theme_compatibility

            # Determine terrain types
            source_terrain = self._infer_terrain_from_theme(source_theme)
            target_terrain = self._infer_terrain_from_theme(target_theme)

            analysis["source_terrain"] = source_terrain
            analysis["target_terrain"] = target_terrain

            # Calculate terrain compatibility
            terrain_compat = self._calculate_terrain_compatibility(
                source_terrain, target_terrain
            )
            analysis["terrain_compatibility"] = terrain_compat

            # Estimate distance based on location characteristics
            estimated_distance = self._estimate_distance(source, target)
            analysis["estimated_distance"] = estimated_distance

            # Identify potential barriers
            barriers = self._identify_natural_barriers(
                source_terrain, target_terrain, source_theme, target_theme
            )
            analysis["natural_barriers"] = barriers

            # Calculate overall feasibility
            feasibility = self._calculate_connection_feasibility(
                theme_compatibility, terrain_compat, len(barriers)
            )
            analysis["connection_feasibility"] = feasibility

            return analysis

        except Exception as e:
            logger.error(f"Error in geographic analysis: {e}")
            return {
                "estimated_distance": "medium",
                "terrain_compatibility": 0.5,
                "theme_compatibility": 0.5,
                "connection_feasibility": 0.5,
                "natural_barriers": [],
            }

    async def get_existing_connection_patterns(
        self, location_id: UUID
    ) -> dict[str, Any]:
        """Analyze existing connection patterns for consistency."""
        try:
            patterns: dict[str, Any] = {
                "connection_count": 0,
                "connection_types": [],
                "average_difficulty": 2.0,
                "common_features": [],
                "pattern_consistency": 0.8,
            }

            # Get connections from the location
            location = self.world_state.locations.get(location_id)
            if not location:
                return patterns

            # Analyze existing connections in the location's connection data
            # Note: In a real implementation, this would query the database
            # For now, we'll analyze based on the location's connections dict
            connections = location.connections

            if connections:
                patterns["connection_count"] = len(connections)

                # Analyze connection directions to infer types
                directions = list(connections.keys())
                patterns["connection_directions"] = directions

                # Infer common patterns
                if "north" in directions and "south" in directions:
                    patterns["common_features"].append("north-south_corridor")
                if "east" in directions and "west" in directions:
                    patterns["common_features"].append("east-west_passage")

                # Set pattern consistency based on connection count
                if len(connections) <= 2:
                    patterns["pattern_consistency"] = 0.9
                elif len(connections) <= 4:
                    patterns["pattern_consistency"] = 0.7
                else:
                    patterns["pattern_consistency"] = 0.5

            return patterns

        except Exception as e:
            logger.error(f"Error analyzing connection patterns: {e}")
            return {
                "connection_count": 0,
                "connection_types": [],
                "pattern_consistency": 0.5,
            }

    async def determine_narrative_requirements(
        self, source: Location, target: Location, purpose: str
    ) -> dict[str, Any]:
        """Determine narrative requirements for connection."""
        try:
            requirements = {
                "story_significance": "minor",
                "required_atmosphere": "neutral",
                "special_requirements": [],
                "difficulty_preference": "medium",
                "thematic_elements": [],
            }

            # Purpose-based requirements
            if purpose == "quest_path":
                requirements["story_significance"] = "major"
                requirements["difficulty_preference"] = "varied"
                requirements["special_requirements"].append("memorable_features")
            elif purpose == "exploration":
                requirements["required_atmosphere"] = "mysterious"
                requirements["special_requirements"].append("discovery_elements")
            elif purpose == "narrative_enhancement":
                requirements["story_significance"] = "major"
                requirements["thematic_elements"].append("atmospheric_details")

            # Theme-based atmosphere
            source_theme = source.state_flags.get("theme", "Unknown")
            target_theme = target.state_flags.get("theme", "Unknown")

            atmosphere = self._determine_thematic_atmosphere(source_theme, target_theme)
            requirements["required_atmosphere"] = atmosphere

            # Add thematic elements
            thematic_elements = self._get_thematic_elements(source_theme, target_theme)
            requirements["thematic_elements"].extend(thematic_elements)

            # Determine difficulty based on themes
            difficulty = self._determine_narrative_difficulty(
                source_theme, target_theme
            )
            requirements["difficulty_preference"] = difficulty

            return requirements

        except Exception as e:
            logger.error(f"Error determining narrative requirements: {e}")
            return {
                "story_significance": "minor",
                "required_atmosphere": "neutral",
                "difficulty_preference": "medium",
            }

    def _analyze_terrain_constraints(
        self,
        source: Location,
        target: Location,
        geographic_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """Analyze terrain constraints for connection generation."""
        constraints = {
            "source_terrain": geographic_analysis.get("source_terrain", "generic"),
            "target_terrain": geographic_analysis.get("target_terrain", "generic"),
            "barriers": geographic_analysis.get("natural_barriers", []),
            "elevation_change": geographic_analysis.get("elevation_change", "minimal"),
            "preferred_types": [],
            "restricted_types": [],
        }

        # Determine preferred connection types based on terrain
        source_terrain = constraints["source_terrain"]
        target_terrain = constraints["target_terrain"]

        if "underground" in [source_terrain, target_terrain]:
            constraints["preferred_types"].extend(["tunnel", "passage"])
        if "mountain" in [source_terrain, target_terrain]:
            constraints["preferred_types"].extend(["bridge", "path"])
        if "water" in constraints["barriers"]:
            constraints["preferred_types"].append("bridge")
            constraints["restricted_types"].extend(["path", "road"])

        return constraints

    def _determine_distance_preference(
        self, geographic_analysis: dict[str, Any], purpose: str
    ) -> str:
        """Determine appropriate distance preference."""
        estimated_distance = geographic_analysis.get("estimated_distance", "medium")

        # Purpose adjustments
        if purpose == "quest_path":
            return "variable"  # Quests can have any distance
        elif purpose == "exploration":
            return "medium"  # Exploration prefers medium distances
        elif purpose == "expand_world":
            return estimated_distance  # Use geographic estimate
        else:
            return "medium"  # Default to medium

    async def _get_existing_connections_list(self, location_id: UUID) -> list[str]:
        """Get list of existing connection types from location."""
        try:
            location = self.world_state.locations.get(location_id)
            if not location or not location.connections:
                return []

            # Extract connection information
            # Note: This is simplified - in a real implementation,
            # we'd query the connections database table
            connection_info = []
            for direction in location.connections:
                connection_info.append(f"direction_{direction}")

            return connection_info

        except Exception as e:
            logger.error(f"Error getting existing connections: {e}")
            return []

    def _create_world_snapshot(self) -> dict[str, Any]:
        """Create snapshot of relevant world state."""
        try:
            snapshot = {
                "total_locations": len(self.world_state.locations),
                "world_age": "established",  # Could be calculated from world state
                "dominant_themes": self._get_dominant_themes(),
                "connectivity_level": "medium",  # Could be calculated
            }

            return snapshot

        except Exception as e:
            logger.error(f"Error creating world snapshot: {e}")
            return {"total_locations": 0, "world_age": "new"}

    def _create_fallback_context(
        self, source_location_id: UUID, target_location_id: UUID, purpose: str
    ) -> ConnectionGenerationContext:
        """Create minimal fallback context for error cases."""
        # Create minimal location objects
        fallback_source = Location(
            location_id=source_location_id,
            name="Unknown Location",
            description="A mysterious place",
            connections={},
            objects={},
            npcs={},
            state_flags={"theme": "Unknown"},
        )

        fallback_target = Location(
            location_id=target_location_id,
            name="Unknown Location",
            description="A mysterious place",
            connections={},
            objects={},
            npcs={},
            state_flags={"theme": "Unknown"},
        )

        return ConnectionGenerationContext(
            source_location=fallback_source,
            target_location=fallback_target,
            generation_purpose=purpose,
            distance_preference="medium",
        )

    def _calculate_theme_compatibility(self, theme1: str, theme2: str) -> float:
        """Calculate compatibility between two themes."""
        # Theme compatibility matrix
        compatibility_matrix = {
            ("Forest", "Mountain"): 0.8,
            ("Forest", "Village"): 0.9,
            ("Village", "City"): 0.8,
            ("City", "City"): 0.9,
            ("Dungeon", "Cave"): 0.9,
            ("Mountain", "Cave"): 0.7,
            ("Forest", "Forest"): 1.0,
            ("Village", "Village"): 1.0,
        }

        # Try direct lookup
        key = (theme1, theme2)
        if key in compatibility_matrix:
            return compatibility_matrix[key]

        # Try reverse lookup
        reverse_key = (theme2, theme1)
        if reverse_key in compatibility_matrix:
            return compatibility_matrix[reverse_key]

        # Same theme
        if theme1 == theme2:
            return 1.0

        # Default compatibility
        return 0.5

    def _infer_terrain_from_theme(self, theme: str) -> str:
        """Infer terrain type from location theme."""
        terrain_mapping = {
            "Forest": "forest",
            "Mountain": "mountain",
            "Village": "grassland",
            "City": "urban",
            "Dungeon": "underground",
            "Cave": "underground",
            "Grassland": "grassland",
            "Desert": "desert",
            "Swamp": "wetland",
            "Coast": "coastal",
        }

        return terrain_mapping.get(theme, "generic")

    def _calculate_terrain_compatibility(self, terrain1: str, terrain2: str) -> float:
        """Calculate compatibility between terrain types."""
        # High compatibility pairs
        high_compat = [
            ("forest", "grassland"),
            ("grassland", "mountain"),
            ("mountain", "cave"),
            ("urban", "grassland"),
        ]

        # Medium compatibility pairs
        medium_compat = [
            ("forest", "mountain"),
            ("underground", "cave"),
            ("desert", "mountain"),
        ]

        # Same terrain
        if terrain1 == terrain2:
            return 1.0

        # Check compatibility lists
        pair = (terrain1, terrain2)
        reverse_pair = (terrain2, terrain1)

        if pair in high_compat or reverse_pair in high_compat:
            return 0.8
        elif pair in medium_compat or reverse_pair in medium_compat:
            return 0.6
        else:
            return 0.4

    def _estimate_distance(self, source: Location, target: Location) -> str:
        """Estimate distance between locations based on characteristics."""
        # In a real implementation, this could use actual coordinates
        # For now, we'll estimate based on theme and description similarity

        source_theme = source.state_flags.get("theme", "Unknown")
        target_theme = target.state_flags.get("theme", "Unknown")

        # Same theme suggests closer locations
        if source_theme == target_theme:
            return "short"

        # Compatible themes suggest medium distance
        compatibility = self._calculate_theme_compatibility(source_theme, target_theme)
        if compatibility > 0.7:
            return "medium"
        else:
            return "long"

    def _identify_natural_barriers(
        self,
        source_terrain: str,
        target_terrain: str,
        source_theme: str,
        target_theme: str,
    ) -> list[str]:
        """Identify potential natural barriers between locations."""
        barriers = []

        # Terrain-based barriers
        if source_terrain == "mountain" or target_terrain == "mountain":
            barriers.append("mountain_range")

        if "water" in [source_terrain, target_terrain] or "Coast" in [
            source_theme,
            target_theme,
        ]:
            barriers.append("water")

        if source_terrain == "underground" and target_terrain != "underground":
            barriers.append("elevation_change")

        if "Desert" in [source_theme, target_theme] and "Forest" in [
            source_theme,
            target_theme,
        ]:
            barriers.append("climate_zone")

        return barriers

    def _calculate_connection_feasibility(
        self,
        theme_compatibility: float,
        terrain_compatibility: float,
        barrier_count: int,
    ) -> float:
        """Calculate overall connection feasibility."""
        base_feasibility = (theme_compatibility + terrain_compatibility) / 2

        # Reduce feasibility for each barrier
        barrier_penalty = min(barrier_count * 0.1, 0.3)

        return max(0.1, base_feasibility - barrier_penalty)

    def _get_dominant_themes(self) -> list[str]:
        """Get dominant themes in the world."""
        theme_counts = {}

        for location in self.world_state.locations.values():
            theme = location.state_flags.get("theme", "Unknown")
            theme_counts[theme] = theme_counts.get(theme, 0) + 1

        # Return themes sorted by frequency
        sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        return [theme for theme, count in sorted_themes[:3]]

    def _determine_thematic_atmosphere(
        self, source_theme: str, target_theme: str
    ) -> str:
        """Determine appropriate atmosphere for connection."""
        atmosphere_mapping = {
            "Forest": "peaceful",
            "Mountain": "majestic",
            "Village": "welcoming",
            "City": "bustling",
            "Dungeon": "foreboding",
            "Cave": "mysterious",
        }

        source_atmosphere = atmosphere_mapping.get(source_theme, "neutral")
        target_atmosphere = atmosphere_mapping.get(target_theme, "neutral")

        # Combine or choose dominant atmosphere
        if source_atmosphere == target_atmosphere:
            return source_atmosphere
        elif "foreboding" in [source_atmosphere, target_atmosphere]:
            return "cautious"
        elif "peaceful" in [source_atmosphere, target_atmosphere]:
            return "serene"
        else:
            return "transitional"

    def _get_thematic_elements(self, source_theme: str, target_theme: str) -> list[str]:
        """Get thematic elements for connection description."""
        elements = []

        theme_elements = {
            "Forest": ["natural_sounds", "forest_canopy", "woodland_creatures"],
            "Mountain": ["mountain_winds", "rocky_terrain", "distant_peaks"],
            "Village": ["rural_charm", "well_worn_paths", "peaceful_atmosphere"],
            "City": ["urban_sounds", "architectural_details", "city_life"],
            "Dungeon": ["ancient_construction", "mysterious_echoes", "hidden_dangers"],
        }

        # Add elements from both themes
        elements.extend(theme_elements.get(source_theme, []))
        elements.extend(theme_elements.get(target_theme, []))

        # Remove duplicates and limit
        return list(set(elements))[:3]

    def _determine_narrative_difficulty(
        self, source_theme: str, target_theme: str
    ) -> str:
        """Determine narrative difficulty preference."""
        difficulty_mapping = {
            "Dungeon": "high",
            "Cave": "medium_high",
            "Mountain": "medium",
            "Forest": "medium",
            "Village": "low",
            "City": "low",
        }

        source_difficulty = difficulty_mapping.get(source_theme, "medium")
        target_difficulty = difficulty_mapping.get(target_theme, "medium")

        # Return the higher difficulty
        difficulty_order = ["low", "medium", "medium_high", "high"]

        source_index = (
            difficulty_order.index(source_difficulty)
            if source_difficulty in difficulty_order
            else 1
        )
        target_index = (
            difficulty_order.index(target_difficulty)
            if target_difficulty in difficulty_order
            else 1
        )

        max_index = max(source_index, target_index)
        return difficulty_order[max_index]
