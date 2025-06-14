"""
Location theme manager for ensuring consistency across the generated world.
"""

import logging

from ...database.session_factory import DatabaseSessionFactory
from ...state.models import WorldState
from ..models.location_models import (
    AdjacentLocationContext,
    LocationGenerationContext,
    LocationTheme,
    ThemeContent,
    ThemeTransitionRules,
)

logger = logging.getLogger(__name__)


class LocationThemeManager:
    """Manages location themes and ensures consistency across the generated world."""

    def __init__(
        self, world_state: WorldState, session_factory: DatabaseSessionFactory
    ):
        self.world_state = world_state
        self.session_factory = session_factory
        self._theme_cache: dict[str, LocationTheme] = {}
        self._transition_cache: dict[tuple[str, str], ThemeTransitionRules] = {}

    async def determine_location_theme(
        self, context: LocationGenerationContext
    ) -> LocationTheme:
        """Determine appropriate theme for new location based on context."""
        logger.debug(
            f"Determining theme for location at {context.expansion_point.direction}"
        )

        # Analyze adjacent themes
        adjacent_themes = self._extract_adjacent_themes(context.adjacent_locations)

        # Check for theme consistency patterns
        dominant_theme = self._find_dominant_theme(adjacent_themes)

        # Consider player preferences
        preferred_themes = context.player_preferences.preferred_themes
        avoided_themes = context.player_preferences.avoided_themes

        # Select theme based on analysis
        selected_theme = await self._select_optimal_theme(
            dominant_theme, preferred_themes, avoided_themes, context.world_themes
        )

        logger.debug(f"Selected theme: {selected_theme.name}")
        return selected_theme

    def _extract_adjacent_themes(
        self, adjacent_locations: list[AdjacentLocationContext]
    ) -> list[str]:
        """Extract themes from adjacent locations."""
        themes = []
        for location in adjacent_locations:
            if location.theme:
                themes.append(location.theme)
        return themes

    def _find_dominant_theme(self, themes: list[str]) -> str | None:
        """Find the most common theme among adjacent locations."""
        if not themes:
            return None

        theme_counts = {}
        for theme in themes:
            theme_counts[theme] = theme_counts.get(theme, 0) + 1

        return max(theme_counts, key=theme_counts.get)

    async def _select_optimal_theme(
        self,
        dominant_theme: str | None,
        preferred_themes: list[str],
        avoided_themes: list[str],
        available_themes: list[LocationTheme],
    ) -> LocationTheme:
        """Select the optimal theme considering all factors."""

        # Create scoring for each theme
        theme_scores = {}

        for theme in available_themes:
            score = 0.0

            # Bonus for dominant adjacent theme
            if dominant_theme and theme.name == dominant_theme:
                score += 3.0

            # Bonus for player preferences
            if theme.name in preferred_themes:
                score += 2.0

            # Penalty for avoided themes
            if theme.name in avoided_themes:
                score -= 5.0

            # Consider transition compatibility
            if dominant_theme:
                transition_score = await self._calculate_transition_score(
                    dominant_theme, theme.name
                )
                score += transition_score

            theme_scores[theme.name] = score

        # Select theme with highest score
        best_theme_name = max(theme_scores, key=theme_scores.get)
        return next(
            theme for theme in available_themes if theme.name == best_theme_name
        )

    async def _calculate_transition_score(
        self, from_theme: str, to_theme: str
    ) -> float:
        """Calculate compatibility score for theme transition."""
        try:
            transition_rules = await self.get_theme_transition_rules(
                from_theme, to_theme
            )
            return transition_rules.compatibility_score
        except Exception as e:
            logger.warning(f"Could not calculate transition score: {e}")
            return 0.0

    async def validate_theme_consistency(
        self, theme: LocationTheme, adjacent_themes: list[LocationTheme]
    ) -> bool:
        """Validate theme consistency with surrounding areas."""
        logger.debug(f"Validating consistency for theme: {theme.name}")

        if not adjacent_themes:
            return True

        # Check transition rules for each adjacent theme
        for adjacent_theme in adjacent_themes:
            try:
                transition_rules = await self.get_theme_transition_rules(
                    adjacent_theme.name, theme.name
                )

                if transition_rules.compatibility_score < 0.3:
                    logger.warning(
                        f"Low compatibility between {adjacent_theme.name} and {theme.name}: "
                        f"{transition_rules.compatibility_score}"
                    )
                    return False

            except Exception as e:
                logger.error(f"Error validating theme consistency: {e}")
                return False

        return True

    async def get_theme_transition_rules(
        self, from_theme: str, to_theme: str
    ) -> ThemeTransitionRules:
        """Get rules for transitioning between themes."""
        cache_key = (from_theme, to_theme)

        if cache_key in self._transition_cache:
            return self._transition_cache[cache_key]

        # Query database for transition rules
        transition_rules = await self._load_transition_rules(from_theme, to_theme)

        if not transition_rules:
            # Generate default transition rules
            transition_rules = self._generate_default_transition_rules(
                from_theme, to_theme
            )

        self._transition_cache[cache_key] = transition_rules
        return transition_rules

    async def _load_transition_rules(
        self, from_theme: str, to_theme: str
    ) -> ThemeTransitionRules | None:
        """Load transition rules from database."""
        try:
            async with self.session_factory.get_session() as session:
                # Query theme_transitions table
                query = """
                    SELECT tt.compatibility_score, tt.transition_rules, tt.is_valid
                    FROM theme_transitions tt
                    JOIN location_themes lt1 ON tt.from_theme_id = lt1.theme_id
                    JOIN location_themes lt2 ON tt.to_theme_id = lt2.theme_id
                    WHERE lt1.name = :from_theme AND lt2.name = :to_theme
                """

                result = await session.execute(
                    query, {"from_theme": from_theme, "to_theme": to_theme}
                )

                row = result.fetchone()
                if row:
                    return ThemeTransitionRules(
                        from_theme=from_theme,
                        to_theme=to_theme,
                        compatibility_score=row.compatibility_score,
                        transition_requirements=row.transition_rules.get(
                            "requirements", []
                        ),
                        forbidden_elements=row.transition_rules.get("forbidden", []),
                        required_elements=row.transition_rules.get("required", []),
                        transition_description=row.transition_rules.get(
                            "description", ""
                        ),
                    )
        except Exception as e:
            logger.error(f"Error loading transition rules: {e}")

        return None

    def _generate_default_transition_rules(
        self, from_theme: str, to_theme: str
    ) -> ThemeTransitionRules:
        """Generate default transition rules for theme pair."""

        # Simple compatibility matrix
        compatibility_matrix = {
            ("Forest", "Riverside"): 0.9,
            ("Forest", "Mountain"): 0.7,
            ("Forest", "Village"): 0.8,
            ("Forest", "Ruins"): 0.6,
            ("Village", "Riverside"): 0.8,
            ("Village", "Mountain"): 0.5,
            ("Village", "Ruins"): 0.4,
            ("Mountain", "Ruins"): 0.7,
            ("Mountain", "Riverside"): 0.6,
            ("Riverside", "Ruins"): 0.5,
        }

        # Get compatibility score (symmetric)
        score = compatibility_matrix.get((from_theme, to_theme))
        if score is None:
            score = compatibility_matrix.get((to_theme, from_theme), 0.5)

        return ThemeTransitionRules(
            from_theme=from_theme,
            to_theme=to_theme,
            compatibility_score=score,
            transition_requirements=[],
            forbidden_elements=[],
            required_elements=[],
            transition_description=f"Transition from {from_theme} to {to_theme}",
        )

    async def generate_theme_specific_content(
        self, theme: LocationTheme, location_type: str
    ) -> ThemeContent:
        """Generate theme-specific content elements."""
        logger.debug(
            f"Generating content for theme: {theme.name}, type: {location_type}"
        )

        # Base content from theme
        base_objects = theme.typical_objects.copy()
        base_npcs = theme.typical_npcs.copy()

        # Add location-type specific elements
        type_specific_content = self._get_type_specific_content(location_type)

        # Combine and filter
        objects = base_objects + type_specific_content.get("objects", [])
        npcs = base_npcs + type_specific_content.get("npcs", [])

        # Generate descriptions
        descriptions = self._generate_theme_descriptions(theme, location_type)

        return ThemeContent(
            theme_name=theme.name,
            objects=objects[:5],  # Limit to reasonable number
            npcs=npcs[:3],
            descriptions=descriptions,
            atmospheric_elements=theme.visual_elements,
            special_features=type_specific_content.get("special_features", []),
        )

    def _get_type_specific_content(self, location_type: str) -> dict[str, list[str]]:
        """Get content specific to a location type."""
        type_content = {
            "clearing": {
                "objects": ["campfire ring", "log benches", "wildflowers"],
                "npcs": ["traveler", "hermit"],
                "special_features": ["peaceful resting spot"],
            },
            "crossroads": {
                "objects": ["signpost", "milestone", "traveler's pack"],
                "npcs": ["merchant", "guard", "pilgrim"],
                "special_features": ["multiple paths converge"],
            },
            "overlook": {
                "objects": ["viewing platform", "telescope", "bench"],
                "npcs": ["lookout", "artist"],
                "special_features": ["scenic vista"],
            },
            "cave": {
                "objects": ["rock formations", "torch brackets", "pools"],
                "npcs": ["explorer", "bat colony"],
                "special_features": ["echoing chambers"],
            },
        }

        return type_content.get(location_type, {})

    def _generate_theme_descriptions(
        self, theme: LocationTheme, location_type: str
    ) -> list[str]:
        """Generate thematic descriptions for the location."""
        descriptions = []

        # Base theme description
        descriptions.append(theme.description)

        # Atmosphere-based descriptions
        if theme.atmosphere:
            descriptions.append(f"The atmosphere here is {theme.atmosphere}.")

        # Visual elements
        if theme.visual_elements:
            visual_desc = f"You notice {', '.join(theme.visual_elements[:3])}."
            descriptions.append(visual_desc)

        return descriptions

    async def load_theme_by_name(self, theme_name: str) -> LocationTheme | None:
        """Load a theme by name from database or cache."""
        if theme_name in self._theme_cache:
            return self._theme_cache[theme_name]

        try:
            async with self.session_factory.get_session() as session:
                query = """
                    SELECT theme_id, name, description, visual_elements, atmosphere,
                           typical_objects, typical_npcs, generation_parameters,
                           parent_theme_id, created_at
                    FROM location_themes
                    WHERE name = :theme_name
                """

                result = await session.execute(query, {"theme_name": theme_name})
                row = result.fetchone()

                if row:
                    theme = LocationTheme(
                        name=row.name,
                        description=row.description,
                        visual_elements=row.visual_elements or [],
                        atmosphere=row.atmosphere or "",
                        typical_objects=row.typical_objects or [],
                        typical_npcs=row.typical_npcs or [],
                        generation_parameters=row.generation_parameters or {},
                        theme_id=row.theme_id,
                        parent_theme_id=row.parent_theme_id,
                        created_at=row.created_at,
                    )

                    self._theme_cache[theme_name] = theme
                    return theme

        except Exception as e:
            logger.error(f"Error loading theme {theme_name}: {e}")

        return None

    async def get_all_themes(self) -> list[LocationTheme]:
        """Get all available themes."""
        try:
            async with self.session_factory.get_session() as session:
                query = """
                    SELECT theme_id, name, description, visual_elements, atmosphere,
                           typical_objects, typical_npcs, generation_parameters,
                           parent_theme_id, created_at
                    FROM location_themes
                    ORDER BY name
                """

                result = await session.execute(query)
                themes = []

                for row in result:
                    theme = LocationTheme(
                        name=row.name,
                        description=row.description,
                        visual_elements=row.visual_elements or [],
                        atmosphere=row.atmosphere or "",
                        typical_objects=row.typical_objects or [],
                        typical_npcs=row.typical_npcs or [],
                        generation_parameters=row.generation_parameters or {},
                        theme_id=row.theme_id,
                        parent_theme_id=row.parent_theme_id,
                        created_at=row.created_at,
                    )
                    themes.append(theme)

                    # Cache the theme
                    self._theme_cache[theme.name] = theme

                return themes

        except Exception as e:
            logger.error(f"Error loading all themes: {e}")
            return []

    def clear_cache(self) -> None:
        """Clear theme and transition caches."""
        self._theme_cache.clear()
        self._transition_cache.clear()
        logger.debug("Theme manager caches cleared")
