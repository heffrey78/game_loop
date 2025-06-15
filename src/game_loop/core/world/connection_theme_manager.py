"""
Connection Theme Manager for World Connection Management System.

This module manages connection themes, passage archetypes, and terrain-based
connection types for creating thematically appropriate connections.
"""

from __future__ import annotations

import logging

from game_loop.core.models.connection_models import (
    ConnectionArchetype,
    ConnectionGenerationContext,
    ConnectionProperties,
)
from game_loop.database.session_factory import DatabaseSessionFactory
from game_loop.state.models import WorldState

logger = logging.getLogger(__name__)


class ConnectionThemeManager:
    """Manages connection themes, archetypes, and terrain compatibility."""

    def __init__(
        self, world_state: WorldState, session_factory: DatabaseSessionFactory
    ):
        """Initialize theme manager with connection archetypes."""
        self.world_state = world_state
        self.session_factory = session_factory
        self._archetype_cache: dict[str, ConnectionArchetype] = {}
        self._terrain_compatibility: dict[tuple[str, str], float] = {}
        self._theme_affinities: dict[str, dict[str, float]] = {}

        # Initialize default archetypes and terrain data
        self._initialize_default_archetypes()
        self._initialize_terrain_compatibility()
        self._initialize_theme_affinities()

    async def determine_connection_type(
        self, context: ConnectionGenerationContext
    ) -> str:
        """Determine most appropriate connection type for given context."""
        try:
            source_theme = context.source_location.state_flags.get("theme", "Unknown")
            target_theme = context.target_location.state_flags.get("theme", "Unknown")

            # Get available types for this theme combination
            available_types = await self.get_available_connection_types(
                source_theme, target_theme
            )

            if not available_types:
                logger.warning(
                    f"No suitable connection types for {source_theme} -> {target_theme}, using fallback"
                )
                return "passage"

            # Purpose-based preferences
            purpose_preferences = {
                "expand_world": ["passage", "path", "road"],
                "quest_path": ["portal", "bridge", "gateway"],
                "exploration": ["tunnel", "stairway", "passage"],
                "narrative_enhancement": ["portal", "teleporter", "gateway"],
                "player_request": available_types,
            }

            preferred = purpose_preferences.get(
                context.generation_purpose, available_types
            )

            # Find intersection of preferred and available
            suitable = [t for t in preferred if t in available_types]

            if suitable:
                # Consider terrain constraints and distance preference
                best_type = self._select_best_type(suitable, context)
                return best_type
            elif available_types:
                return available_types[0]
            else:
                return "passage"

        except Exception as e:
            logger.error(f"Error determining connection type: {e}")
            return "passage"

    async def get_available_connection_types(
        self, source_theme: str, target_theme: str
    ) -> list[str]:
        """Get connection types suitable for connecting given themes."""
        try:
            # Get affinities for both themes
            source_affinities = self._theme_affinities.get(source_theme, {})
            target_affinities = self._theme_affinities.get(target_theme, {})

            # Find connection types with good affinity for both themes
            suitable_types = []
            for archetype_name, archetype in self._archetype_cache.items():
                source_affinity = archetype.theme_compatibility.get(source_theme, 0.0)
                target_affinity = archetype.theme_compatibility.get(target_theme, 0.0)

                # Require minimum affinity for both themes
                if source_affinity >= 0.3 and target_affinity >= 0.3:
                    connection_type = archetype.typical_properties.connection_type
                    if connection_type not in suitable_types:
                        suitable_types.append(connection_type)

            # Ensure we have at least some basic types
            if not suitable_types:
                # Fallback based on theme characteristics
                if source_theme in ["Dungeon", "Cave", "Underground"]:
                    suitable_types = ["tunnel", "passage", "stairway"]
                elif source_theme in ["City", "Town", "Village"]:
                    suitable_types = ["road", "path", "gateway"]
                elif source_theme in ["Forest", "Mountain", "Wilderness"]:
                    suitable_types = ["path", "passage", "bridge"]
                else:
                    suitable_types = ["passage", "path"]

            return suitable_types

        except Exception as e:
            logger.error(f"Error getting available connection types: {e}")
            return ["passage", "path"]

    async def get_terrain_compatibility(
        self, source_terrain: str, target_terrain: str
    ) -> float:
        """Calculate terrain compatibility score for connection feasibility."""
        try:
            # Direct lookup first
            key = (source_terrain.lower(), target_terrain.lower())
            if key in self._terrain_compatibility:
                return self._terrain_compatibility[key]

            # Reverse lookup
            reverse_key = (target_terrain.lower(), source_terrain.lower())
            if reverse_key in self._terrain_compatibility:
                return self._terrain_compatibility[reverse_key]

            # Calculate based on terrain similarity
            return self._calculate_terrain_similarity(source_terrain, target_terrain)

        except Exception as e:
            logger.error(f"Error calculating terrain compatibility: {e}")
            return 0.5  # Neutral compatibility

    async def generate_theme_appropriate_description(
        self, connection_type: str, context: ConnectionGenerationContext
    ) -> str:
        """Generate theme-consistent connection description."""
        try:
            archetype = self.get_connection_archetype(connection_type)
            if not archetype:
                return f"A {connection_type} connecting the two areas."

            # Get theme-specific description elements
            source_theme = context.source_location.state_flags.get("theme", "Unknown")
            target_theme = context.target_location.state_flags.get("theme", "Unknown")

            # Build description based on archetype and themes
            base_description = archetype.description

            # Add theme-specific elements
            source_elements = self._get_theme_description_elements(source_theme)
            target_elements = self._get_theme_description_elements(target_theme)

            # Combine elements into a cohesive description
            description = self._combine_description_elements(
                base_description, source_elements, target_elements, connection_type
            )

            return description

        except Exception as e:
            logger.error(f"Error generating theme description: {e}")
            return f"A {connection_type} that connects the areas."

    def get_connection_archetype(
        self, connection_type: str
    ) -> ConnectionArchetype | None:
        """Retrieve archetype definition for connection type."""
        return self._archetype_cache.get(connection_type)

    def _select_best_type(
        self, suitable_types: list[str], context: ConnectionGenerationContext
    ) -> str:
        """Select the best connection type from suitable options."""
        try:
            # Score each type based on context
            type_scores = {}

            for conn_type in suitable_types:
                archetype = self.get_connection_archetype(conn_type)
                if not archetype:
                    continue

                score = 0.0

                # Distance preference scoring
                typical_travel_time = archetype.typical_properties.travel_time
                if context.distance_preference == "short" and typical_travel_time <= 30:
                    score += 0.3
                elif (
                    context.distance_preference == "medium"
                    and 30 <= typical_travel_time <= 120
                ):
                    score += 0.3
                elif (
                    context.distance_preference == "long" and typical_travel_time > 120
                ):
                    score += 0.3

                # Terrain compatibility scoring
                source_terrain = context.terrain_constraints.get(
                    "source_terrain", "generic"
                )
                target_terrain = context.terrain_constraints.get(
                    "target_terrain", "generic"
                )

                source_affinity = archetype.terrain_affinities.get(source_terrain, 0.0)
                target_affinity = archetype.terrain_affinities.get(target_terrain, 0.0)
                score += (source_affinity + target_affinity) * 0.2

                # Rarity consideration (prefer common types slightly)
                if archetype.rarity == "common":
                    score += 0.1
                elif archetype.rarity == "uncommon":
                    score += 0.05

                type_scores[conn_type] = score

            # Return the highest scoring type
            if type_scores:
                return max(type_scores.keys(), key=lambda t: type_scores[t])
            else:
                return suitable_types[0]

        except Exception as e:
            logger.error(f"Error selecting best connection type: {e}")
            return suitable_types[0] if suitable_types else "passage"

    def _initialize_default_archetypes(self):
        """Initialize default connection archetypes."""
        # Passage archetype
        self._archetype_cache["passage"] = ConnectionArchetype(
            name="passage",
            description="A simple passage connecting two areas",
            typical_properties=ConnectionProperties(
                connection_type="passage",
                difficulty=2,
                travel_time=30,
                description="A narrow passage winds between the areas",
                visibility="visible",
                requirements=[],
            ),
            terrain_affinities={
                "underground": 0.9,
                "indoor": 0.8,
                "mountain": 0.7,
                "forest": 0.6,
            },
            theme_compatibility={
                "Dungeon": 0.9,
                "Cave": 0.9,
                "Underground": 0.9,
                "Castle": 0.7,
                "Forest": 0.6,
                "Mountain": 0.7,
            },
            generation_templates={
                "basic": "A {type} connecting {source} and {target}",
                "detailed": "A winding {type} carved through the {terrain}",
            },
            rarity="common",
        )

        # Bridge archetype
        self._archetype_cache["bridge"] = ConnectionArchetype(
            name="bridge",
            description="A bridge spanning between elevated areas",
            typical_properties=ConnectionProperties(
                connection_type="bridge",
                difficulty=3,
                travel_time=45,
                description="A sturdy bridge spans across the gap",
                visibility="visible",
                requirements=[],
            ),
            terrain_affinities={
                "river": 0.9,
                "canyon": 0.9,
                "mountain": 0.8,
                "valley": 0.7,
            },
            theme_compatibility={
                "Mountain": 0.9,
                "Forest": 0.7,
                "Village": 0.8,
                "City": 0.8,
                "Wilderness": 0.6,
            },
            generation_templates={
                "basic": "A {type} spanning between {source} and {target}",
                "detailed": "An ancient stone {type} arching over the {terrain}",
            },
            rarity="common",
        )

        # Portal archetype
        self._archetype_cache["portal"] = ConnectionArchetype(
            name="portal",
            description="A magical portal providing instant travel",
            typical_properties=ConnectionProperties(
                connection_type="portal",
                difficulty=1,
                travel_time=5,
                description="A shimmering portal hums with magical energy",
                visibility="visible",
                requirements=["magical_attunement"],
            ),
            terrain_affinities={
                "magical": 0.9,
                "tower": 0.8,
                "shrine": 0.8,
                "ruins": 0.7,
            },
            theme_compatibility={
                "Magical": 0.9,
                "Tower": 0.9,
                "Shrine": 0.8,
                "Ruins": 0.7,
                "City": 0.5,
            },
            generation_templates={
                "basic": "A magical {type} linking {source} and {target}",
                "detailed": "A swirling {type} crackling with arcane energy",
            },
            rarity="rare",
        )

        # Path archetype
        self._archetype_cache["path"] = ConnectionArchetype(
            name="path",
            description="A natural path worn by travelers",
            typical_properties=ConnectionProperties(
                connection_type="path",
                difficulty=2,
                travel_time=60,
                description="A well-worn path leads between the areas",
                visibility="visible",
                requirements=[],
            ),
            terrain_affinities={
                "forest": 0.9,
                "grassland": 0.9,
                "mountain": 0.8,
                "wilderness": 0.8,
            },
            theme_compatibility={
                "Forest": 0.9,
                "Mountain": 0.8,
                "Village": 0.8,
                "Wilderness": 0.9,
                "Grassland": 0.9,
            },
            generation_templates={
                "basic": "A {type} winding between {source} and {target}",
                "detailed": "A meandering {type} through the {terrain}",
            },
            rarity="common",
        )

        # Tunnel archetype
        self._archetype_cache["tunnel"] = ConnectionArchetype(
            name="tunnel",
            description="An underground tunnel burrowed between areas",
            typical_properties=ConnectionProperties(
                connection_type="tunnel",
                difficulty=4,
                travel_time=90,
                description="A dark tunnel burrows through the earth",
                visibility="hidden",
                requirements=[],
            ),
            terrain_affinities={
                "underground": 0.9,
                "mountain": 0.8,
                "cave": 0.9,
                "mine": 0.9,
            },
            theme_compatibility={
                "Underground": 0.9,
                "Cave": 0.9,
                "Mine": 0.9,
                "Dungeon": 0.8,
                "Mountain": 0.7,
            },
            generation_templates={
                "basic": "A {type} carved between {source} and {target}",
                "detailed": "A rough-hewn {type} through solid rock",
            },
            rarity="uncommon",
        )

        # Road archetype
        self._archetype_cache["road"] = ConnectionArchetype(
            name="road",
            description="A constructed road for easy travel",
            typical_properties=ConnectionProperties(
                connection_type="road",
                difficulty=1,
                travel_time=45,
                description="A paved road connects the settlements",
                visibility="visible",
                requirements=[],
            ),
            terrain_affinities={
                "urban": 0.9,
                "grassland": 0.8,
                "plains": 0.9,
                "settled": 0.9,
            },
            theme_compatibility={
                "City": 0.9,
                "Town": 0.9,
                "Village": 0.9,
                "Grassland": 0.7,
                "Plains": 0.8,
            },
            generation_templates={
                "basic": "A {type} running between {source} and {target}",
                "detailed": "A well-maintained {type} with stone markers",
            },
            rarity="common",
        )

    def _initialize_terrain_compatibility(self):
        """Initialize terrain compatibility matrix."""
        # High compatibility (0.8-1.0)
        high_compat = [
            ("forest", "forest"),
            ("mountain", "mountain"),
            ("grassland", "grassland"),
            ("urban", "urban"),
            ("underground", "underground"),
            ("forest", "grassland"),
            ("village", "grassland"),
            ("city", "urban"),
        ]

        # Medium compatibility (0.5-0.7)
        medium_compat = [
            ("forest", "mountain"),
            ("grassland", "mountain"),
            ("village", "forest"),
            ("city", "village"),
            ("underground", "cave"),
            ("mountain", "cave"),
        ]

        # Low compatibility (0.2-0.4)
        low_compat = [
            ("urban", "forest"),
            ("underground", "grassland"),
            ("cave", "grassland"),
            ("water", "mountain"),
        ]

        # Populate compatibility matrix
        for terrain1, terrain2 in high_compat:
            self._terrain_compatibility[(terrain1, terrain2)] = 0.9

        for terrain1, terrain2 in medium_compat:
            self._terrain_compatibility[(terrain1, terrain2)] = 0.6

        for terrain1, terrain2 in low_compat:
            self._terrain_compatibility[(terrain1, terrain2)] = 0.3

    def _initialize_theme_affinities(self):
        """Initialize theme affinities for connection types."""
        self._theme_affinities = {
            "Forest": {
                "path": 0.9,
                "bridge": 0.7,
                "passage": 0.6,
                "tunnel": 0.3,
            },
            "Mountain": {
                "path": 0.8,
                "bridge": 0.9,
                "tunnel": 0.7,
                "passage": 0.7,
            },
            "City": {
                "road": 0.9,
                "gateway": 0.8,
                "passage": 0.6,
                "portal": 0.5,
            },
            "Village": {
                "path": 0.8,
                "road": 0.7,
                "bridge": 0.8,
                "passage": 0.6,
            },
            "Dungeon": {
                "passage": 0.9,
                "tunnel": 0.8,
                "stairway": 0.8,
                "door": 0.7,
            },
            "Cave": {
                "tunnel": 0.9,
                "passage": 0.9,
                "bridge": 0.4,
                "path": 0.3,
            },
        }

    def _calculate_terrain_similarity(self, terrain1: str, terrain2: str) -> float:
        """Calculate similarity between two terrain types."""
        # Normalize terrain names
        terrain1 = terrain1.lower()
        terrain2 = terrain2.lower()

        # Same terrain
        if terrain1 == terrain2:
            return 1.0

        # Natural vs artificial
        natural = ["forest", "mountain", "grassland", "wilderness", "water", "cave"]
        artificial = ["city", "urban", "village", "road", "building"]

        terrain1_natural = terrain1 in natural
        terrain2_natural = terrain2 in natural

        if terrain1_natural == terrain2_natural:
            return 0.6  # Same category
        else:
            return 0.3  # Different categories

    def _get_theme_description_elements(self, theme: str) -> dict[str, list[str]]:
        """Get description elements for a theme."""
        theme_elements = {
            "Forest": {
                "materials": ["wooden", "moss-covered", "vine-wrapped"],
                "atmosphere": ["peaceful", "mysterious", "ancient"],
                "features": ["fallen logs", "hanging branches", "forest sounds"],
            },
            "Mountain": {
                "materials": ["stone", "rocky", "weathered"],
                "atmosphere": ["windswept", "majestic", "challenging"],
                "features": ["mountain winds", "distant peaks", "rocky outcrops"],
            },
            "City": {
                "materials": ["paved", "cobblestone", "brick"],
                "atmosphere": ["bustling", "organized", "civilized"],
                "features": ["street lamps", "signposts", "urban sounds"],
            },
            "Village": {
                "materials": ["well-worn", "rustic", "handmade"],
                "atmosphere": ["quaint", "peaceful", "welcoming"],
                "features": ["flower boxes", "worn stones", "village charm"],
            },
            "Dungeon": {
                "materials": ["ancient stone", "iron-bound", "torchlit"],
                "atmosphere": ["foreboding", "mysterious", "dangerous"],
                "features": ["echoing footsteps", "flickering shadows", "cold drafts"],
            },
        }

        return theme_elements.get(
            theme,
            {
                "materials": ["weathered", "simple"],
                "atmosphere": ["quiet", "unremarkable"],
                "features": ["basic construction"],
            },
        )

    def _combine_description_elements(
        self,
        base_description: str,
        source_elements: dict[str, list[str]],
        target_elements: dict[str, list[str]],
        connection_type: str,
    ) -> str:
        """Combine elements into a cohesive description."""
        try:
            # Get a material and atmosphere from source elements
            source_material = source_elements.get("materials", ["simple"])[0]
            source_atmosphere = source_elements.get("atmosphere", ["quiet"])[0]

            # Create enhanced description
            if connection_type == "bridge":
                return f"A {source_material} bridge with a {source_atmosphere} presence spans gracefully between the areas."
            elif connection_type == "path":
                return f"A {source_atmosphere} path winds its way through {source_material} terrain connecting the locations."
            elif connection_type == "tunnel":
                return f"A {source_atmosphere} tunnel carved through {source_material} rock provides passage between the areas."
            elif connection_type == "portal":
                return f"A {source_atmosphere} magical portal shimmers with arcane energy, offering instant travel."
            elif connection_type == "road":
                return f"A well-maintained {source_material} road provides easy travel with a {source_atmosphere} journey."
            else:  # passage
                return f"A {source_atmosphere} passage through {source_material} architecture connects the locations."

        except Exception as e:
            logger.error(f"Error combining description elements: {e}")
            return base_description
