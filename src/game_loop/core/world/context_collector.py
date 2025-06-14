"""
Location context collector for gathering and analyzing context for location generation.
"""

import logging
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from ...database.session_factory import DatabaseSessionFactory
from ...state.models import WorldState
from ..models.location_models import (
    AdjacentLocationContext,
    EnrichedContext,
    LocationGenerationContext,
    NarrativeContext,
    PlayerLocationPreferences,
)
from ..models.navigation_models import ExpansionPoint

logger = logging.getLogger(__name__)


class LocationContextCollector:
    """Collects and analyzes context for location generation from various sources."""

    def __init__(
        self, world_state: WorldState, session_factory: DatabaseSessionFactory
    ):
        self.world_state = world_state
        self.session_factory = session_factory
        self._preference_cache: dict[UUID, PlayerLocationPreferences] = {}
        self._context_cache: dict[str, LocationGenerationContext] = {}

    async def collect_expansion_context(
        self, expansion_point: ExpansionPoint
    ) -> LocationGenerationContext:
        """Collect comprehensive context for location generation."""
        logger.debug(f"Collecting context for expansion at {expansion_point.direction}")

        # Check cache first
        cache_key = f"{expansion_point.location_id}_{expansion_point.direction}"
        if cache_key in self._context_cache:
            cached_context = self._context_cache[cache_key]
            # Use cache if less than 5 minutes old
            if hasattr(cached_context, "_cached_at"):
                if datetime.now() - cached_context._cached_at < timedelta(minutes=5):
                    return cached_context

        # Gather adjacent locations
        adjacent_locations = await self.gather_adjacent_context(
            expansion_point.location_id, expansion_point.direction
        )

        # Get player preferences (use first player for now)
        player_preferences = await self._get_default_player_preferences()

        # Get world themes
        world_themes = await self._get_world_themes()

        # Extract narrative context
        narrative_context = await self.extract_narrative_context(
            expansion_point.location_id,
            expansion_point.context.get("location_name", ""),
        )

        # Create context
        context = LocationGenerationContext(
            expansion_point=expansion_point,
            adjacent_locations=adjacent_locations,
            player_preferences=player_preferences,
            world_themes=world_themes,
            narrative_context=narrative_context,
            generation_constraints=self._get_generation_constraints(),
            world_rules=self._get_world_rules(),
        )

        # Cache the context
        context._cached_at = datetime.now()  # Add timestamp
        self._context_cache[cache_key] = context

        return context

    async def analyze_player_preferences(
        self, player_id: UUID
    ) -> PlayerLocationPreferences:
        """Analyze player's location preferences from history."""
        if player_id in self._preference_cache:
            return self._preference_cache[player_id]

        logger.debug(f"Analyzing preferences for player {player_id}")

        try:
            # Query player's location visit history
            async with self.session_factory.get_session() as session:
                query = """
                    SELECT l.name, l.description, l.theme_id, lt.name as theme_name,
                           COUNT(*) as visit_count, MAX(gsh.timestamp) as last_visit
                    FROM game_state_history gsh
                    JOIN locations l ON gsh.location_id = l.location_id
                    LEFT JOIN location_themes lt ON l.theme_id = lt.theme_id
                    WHERE gsh.player_id = :player_id
                      AND gsh.timestamp > :cutoff_date
                    GROUP BY l.location_id, l.name, l.description, l.theme_id, lt.name
                    ORDER BY visit_count DESC, last_visit DESC
                """

                cutoff_date = datetime.now() - timedelta(days=30)
                result = await session.execute(
                    query, {"player_id": player_id, "cutoff_date": cutoff_date}
                )

                # Analyze visit patterns
                visit_data = result.fetchall()
                preferences = self._analyze_visit_patterns(visit_data)

        except Exception as e:
            logger.error(f"Error analyzing player preferences: {e}")
            preferences = self._get_default_preferences()

        self._preference_cache[player_id] = preferences
        return preferences

    def _analyze_visit_patterns(
        self, visit_data: list[Any]
    ) -> PlayerLocationPreferences:
        """Analyze visit patterns to determine preferences."""
        theme_counts = {}
        environment_types = []
        total_visits = 0

        for row in visit_data:
            theme_name = row.theme_name or "Unknown"
            visit_count = row.visit_count

            theme_counts[theme_name] = theme_counts.get(theme_name, 0) + visit_count
            total_visits += visit_count

            # Infer environment types from location names/descriptions
            location_name = row.name.lower()
            if any(word in location_name for word in ["forest", "tree", "wood"]):
                environment_types.append("forest")
            elif any(word in location_name for word in ["village", "town", "market"]):
                environment_types.append("village")
            elif any(word in location_name for word in ["mountain", "peak", "cliff"]):
                environment_types.append("mountain")
            elif any(word in location_name for word in ["river", "stream", "water"]):
                environment_types.append("riverside")

        # Determine preferred themes
        if theme_counts:
            sorted_themes = sorted(
                theme_counts.items(), key=lambda x: x[1], reverse=True
            )
            preferred_themes = [theme for theme, count in sorted_themes[:3]]
        else:
            preferred_themes = []

        # Determine interaction style based on visit frequency
        avg_visits = total_visits / max(len(visit_data), 1)
        if avg_visits > 5:
            interaction_style = "explorer"
        elif avg_visits > 2:
            interaction_style = "balanced"
        else:
            interaction_style = "casual"

        # Determine complexity preference
        complexity_level = "medium"  # Default
        if len(visit_data) > 20:
            complexity_level = "high"
        elif len(visit_data) < 5:
            complexity_level = "low"

        return PlayerLocationPreferences(
            environments=list(set(environment_types[:5])),
            interaction_style=interaction_style,
            complexity_level=complexity_level,
            preferred_themes=preferred_themes,
            avoided_themes=[],  # Would need negative feedback data
            exploration_preference="balanced",
            social_preference="balanced",
        )

    def _get_default_preferences(self) -> PlayerLocationPreferences:
        """Get default player preferences."""
        return PlayerLocationPreferences(
            environments=["forest", "village"],
            interaction_style="balanced",
            complexity_level="medium",
            preferred_themes=["Forest", "Village"],
            avoided_themes=[],
            exploration_preference="balanced",
            social_preference="balanced",
        )

    async def _get_default_player_preferences(self) -> PlayerLocationPreferences:
        """Get default player preferences for system use."""
        # For now, return default preferences
        # In a real implementation, this would analyze the active player
        return self._get_default_preferences()

    async def gather_adjacent_context(
        self, location_id: UUID, direction: str
    ) -> list[AdjacentLocationContext]:
        """Gather context from locations adjacent to expansion point."""
        adjacent_contexts = []

        if location_id not in self.world_state.locations:
            logger.warning(f"Location {location_id} not found in world state")
            return adjacent_contexts

        current_location = self.world_state.locations[location_id]

        # Get all connected locations
        for conn_direction, connected_id in current_location.connections.items():
            if connected_id in self.world_state.locations:
                connected_location = self.world_state.locations[connected_id]

                # Extract theme information
                theme_name = "Unknown"
                if (
                    hasattr(connected_location, "theme_id")
                    and connected_location.theme_id
                ):
                    theme_name = await self._get_theme_name(connected_location.theme_id)
                else:
                    # Infer theme from location characteristics
                    theme_name = self._infer_theme_from_location(connected_location)

                adjacent_context = AdjacentLocationContext(
                    location_id=connected_id,
                    direction=conn_direction,
                    name=connected_location.name,
                    description=connected_location.description,
                    theme=theme_name,
                    short_description=self._create_short_description(
                        connected_location
                    ),
                    objects=(
                        list(connected_location.objects.keys())
                        if connected_location.objects
                        else []
                    ),
                    npcs=(
                        list(connected_location.npcs.keys())
                        if connected_location.npcs
                        else []
                    ),
                )

                adjacent_contexts.append(adjacent_context)

        logger.debug(f"Gathered {len(adjacent_contexts)} adjacent location contexts")
        return adjacent_contexts

    async def _get_theme_name(self, theme_id: UUID) -> str:
        """Get theme name from theme ID."""
        try:
            async with self.session_factory.get_session() as session:
                query = "SELECT name FROM location_themes WHERE theme_id = :theme_id"
                result = await session.execute(query, {"theme_id": theme_id})
                row = result.fetchone()
                return row.name if row else "Unknown"
        except Exception as e:
            logger.error(f"Error getting theme name: {e}")
            return "Unknown"

    def _infer_theme_from_location(self, location) -> str:
        """Infer theme from location characteristics."""
        name_lower = location.name.lower()
        desc_lower = location.description.lower()

        # Check for theme keywords
        if any(
            word in name_lower + desc_lower
            for word in ["forest", "tree", "wood", "grove"]
        ):
            return "Forest"
        elif any(
            word in name_lower + desc_lower
            for word in ["village", "town", "market", "square"]
        ):
            return "Village"
        elif any(
            word in name_lower + desc_lower
            for word in ["mountain", "peak", "cliff", "summit"]
        ):
            return "Mountain"
        elif any(
            word in name_lower + desc_lower
            for word in ["river", "stream", "water", "bank"]
        ):
            return "Riverside"
        elif any(
            word in name_lower + desc_lower
            for word in ["ruin", "ancient", "crumbling", "old"]
        ):
            return "Ruins"
        else:
            return "Forest"  # Default theme

    def _create_short_description(self, location) -> str:
        """Create a short description from full description."""
        desc = location.description
        if len(desc) <= 50:
            return desc

        # Take first sentence or first 50 chars
        first_sentence = desc.split(".")[0]
        if len(first_sentence) <= 50:
            return first_sentence + "."
        else:
            return desc[:47] + "..."

    async def extract_narrative_context(
        self, player_id: UUID, location_area: str
    ) -> NarrativeContext | None:
        """Extract relevant narrative context for generation."""
        try:
            # For now, return basic narrative context
            # In a full implementation, this would analyze:
            # - Active quests in the area
            # - Recent player actions
            # - World events
            # - Story progression

            return NarrativeContext(
                current_quests=[],
                story_themes=["exploration", "discovery"],
                player_actions=["moving", "exploring"],
                world_events=[],
                narrative_tension="neutral",
            )

        except Exception as e:
            logger.error(f"Error extracting narrative context: {e}")
            return None

    async def _get_world_themes(self) -> list:
        """Get available world themes."""
        # Import here to avoid circular imports
        from .theme_manager import LocationThemeManager

        try:
            theme_manager = LocationThemeManager(self.world_state, self.session_factory)
            return await theme_manager.get_all_themes()
        except Exception as e:
            logger.error(f"Error getting world themes: {e}")
            return []

    def _get_generation_constraints(self) -> dict[str, Any]:
        """Get constraints for location generation."""
        return {
            "max_objects": 5,
            "max_npcs": 3,
            "min_description_length": 100,
            "max_description_length": 500,
            "required_connections": 1,
            "avoid_duplicate_names": True,
        }

    def _get_world_rules(self) -> list[str]:
        """Get world consistency rules."""
        return [
            "Locations must have logical connections",
            "Themes should transition naturally",
            "Objects should fit the environment",
            "NPCs should have reasonable motivations",
            "Descriptions should be engaging and immersive",
            "Avoid modern anachronisms",
            "Maintain consistent tone and style",
        ]

    async def enrich_context(
        self, base_context: LocationGenerationContext
    ) -> EnrichedContext:
        """Enhance context with additional analysis."""
        logger.debug("Enriching location generation context")

        # Analyze historical patterns
        historical_patterns = await self._analyze_historical_patterns(base_context)

        # Analyze player behavior
        player_behavior = await self._analyze_player_behavior(base_context)

        # Analyze world consistency factors
        consistency_factors = self._analyze_world_consistency(base_context)

        # Generate hints and priority elements
        hints = self._generate_generation_hints(base_context)
        priorities = self._identify_priority_elements(base_context)

        return EnrichedContext(
            base_context=base_context,
            historical_patterns=historical_patterns,
            player_behavior_analysis=player_behavior,
            world_consistency_factors=consistency_factors,
            generation_hints=hints,
            priority_elements=priorities,
        )

    async def _analyze_historical_patterns(
        self, context: LocationGenerationContext
    ) -> dict[str, Any]:
        """Analyze historical location generation patterns."""
        try:
            async with self.session_factory.get_session() as session:
                # Get recent location generation history
                query = """
                    SELECT generation_context, generated_content, validation_result
                    FROM location_generation_history
                    WHERE created_at > :cutoff_date
                    ORDER BY created_at DESC
                    LIMIT 50
                """

                cutoff_date = datetime.now() - timedelta(days=7)
                result = await session.execute(query, {"cutoff_date": cutoff_date})

                patterns = {
                    "common_themes": [],
                    "successful_combinations": [],
                    "generation_frequency": {},
                    "validation_scores": [],
                }

                for row in result:
                    # Analyze patterns from historical data
                    if row.validation_result:
                        patterns["validation_scores"].append(
                            row.validation_result.get("overall_score", 0)
                        )

                return patterns

        except Exception as e:
            logger.error(f"Error analyzing historical patterns: {e}")
            return {"error": str(e)}

    async def _analyze_player_behavior(
        self, context: LocationGenerationContext
    ) -> dict[str, Any]:
        """Analyze player behavior relevant to location generation."""
        return {
            "exploration_style": context.player_preferences.exploration_preference,
            "social_interaction": context.player_preferences.social_preference,
            "complexity_comfort": context.player_preferences.complexity_level,
            "preferred_environments": context.player_preferences.environments,
        }

    def _analyze_world_consistency(
        self, context: LocationGenerationContext
    ) -> dict[str, Any]:
        """Analyze world consistency factors."""
        adjacent_themes = [loc.theme for loc in context.adjacent_locations]

        return {
            "adjacent_theme_diversity": len(set(adjacent_themes)),
            "dominant_theme": (
                max(set(adjacent_themes), key=adjacent_themes.count)
                if adjacent_themes
                else None
            ),
            "theme_transitions_needed": len(set(adjacent_themes)) > 1,
            "expansion_direction": context.expansion_point.direction,
        }

    def _generate_generation_hints(
        self, context: LocationGenerationContext
    ) -> list[str]:
        """Generate hints for location generation."""
        hints = []

        # Theme-based hints
        adjacent_themes = [loc.theme for loc in context.adjacent_locations]
        if adjacent_themes:
            dominant_theme = max(set(adjacent_themes), key=adjacent_themes.count)
            hints.append(
                f"Consider theme consistency with dominant {dominant_theme} theme"
            )

        # Direction-based hints
        direction = context.expansion_point.direction
        if direction in ["north", "up"]:
            hints.append("Consider elevated or northward locations")
        elif direction in ["south", "down"]:
            hints.append("Consider lower or southward locations")

        # Player preference hints
        prefs = context.player_preferences
        if prefs.preferred_themes:
            hints.append(f"Player enjoys {', '.join(prefs.preferred_themes)} themes")

        return hints

    def _identify_priority_elements(
        self, context: LocationGenerationContext
    ) -> list[str]:
        """Identify priority elements for generation."""
        priorities = []

        # Always prioritize theme consistency
        priorities.append("theme_consistency")

        # Prioritize based on player preferences
        if context.player_preferences.complexity_level == "high":
            priorities.append("rich_interactions")

        if context.player_preferences.social_preference == "high":
            priorities.append("npc_presence")

        # Prioritize based on expansion context
        if context.expansion_point.priority > 0.7:
            priorities.append("high_quality_generation")

        return priorities

    def clear_cache(self) -> None:
        """Clear context caches."""
        self._preference_cache.clear()
        self._context_cache.clear()
        logger.debug("Context collector caches cleared")
