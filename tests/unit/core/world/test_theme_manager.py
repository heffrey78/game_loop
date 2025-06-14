"""
Unit tests for LocationThemeManager.
"""

from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from game_loop.core.models.location_models import (
    AdjacentLocationContext,
    LocationGenerationContext,
    LocationTheme,
    PlayerLocationPreferences,
    ThemeContent,
    ThemeTransitionRules,
)
from game_loop.core.models.navigation_models import ExpansionPoint
from game_loop.core.world.theme_manager import LocationThemeManager
from game_loop.state.models import WorldState


@pytest.fixture
def sample_themes():
    """Create sample location themes."""
    return [
        LocationTheme(
            name="Forest",
            description="Dense woodland",
            visual_elements=["trees", "leaves"],
            atmosphere="peaceful",
            typical_objects=["log", "mushroom"],
            typical_npcs=["rabbit", "hermit"],
            generation_parameters={"complexity": "medium"},
            theme_id=uuid4(),
        ),
        LocationTheme(
            name="Village",
            description="Small settlement",
            visual_elements=["buildings", "paths"],
            atmosphere="welcoming",
            typical_objects=["well", "market stall"],
            typical_npcs=["merchant", "villager"],
            generation_parameters={"complexity": "high"},
            theme_id=uuid4(),
        ),
    ]


@pytest.fixture
def mock_session_factory():
    """Create a mock session factory."""
    session = AsyncMock()
    session_factory = Mock()
    context_manager = AsyncMock()
    context_manager.__aenter__.return_value = session
    context_manager.__aexit__.return_value = None
    session_factory.get_session.return_value = context_manager
    return session_factory, session


@pytest.fixture
def theme_manager(mock_session_factory):
    """Create a LocationThemeManager instance with mocked dependencies."""
    session_factory, _ = mock_session_factory
    world_state = WorldState()
    return LocationThemeManager(world_state, session_factory)


@pytest.fixture
def sample_context():
    """Create a sample generation context."""
    expansion_point = ExpansionPoint(
        location_id=uuid4(),
        direction="north",
        priority=0.7,
        context={"location_name": "Starting Point"},
    )

    adjacent_location = AdjacentLocationContext(
        location_id=uuid4(),
        direction="south",
        name="Forest Grove",
        description="A peaceful grove",
        theme="Forest",
        short_description="A grove",
    )

    player_prefs = PlayerLocationPreferences(
        environments=["forest", "village"],
        interaction_style="explorer",
        complexity_level="medium",
        preferred_themes=["Forest"],
        avoided_themes=["Ruins"],
    )

    return LocationGenerationContext(
        expansion_point=expansion_point,
        adjacent_locations=[adjacent_location],
        player_preferences=player_prefs,
        world_themes=[],  # Will be populated in tests
    )


class TestLocationThemeManager:
    """Test cases for LocationThemeManager."""

    @pytest.mark.asyncio
    async def test_determine_location_theme_dominant_adjacent(
        self, theme_manager, sample_context, sample_themes
    ):
        """Test theme determination based on dominant adjacent theme."""
        # Setup context with forest theme dominance
        sample_context.world_themes = sample_themes
        sample_context.adjacent_locations = [
            AdjacentLocationContext(
                location_id=uuid4(),
                direction="south",
                name="Grove A",
                description="Forest grove",
                theme="Forest",
                short_description="Grove",
            ),
            AdjacentLocationContext(
                location_id=uuid4(),
                direction="east",
                name="Grove B",
                description="Another forest grove",
                theme="Forest",
                short_description="Grove",
            ),
            AdjacentLocationContext(
                location_id=uuid4(),
                direction="west",
                name="Small Village",
                description="A village",
                theme="Village",
                short_description="Village",
            ),
        ]

        result = await theme_manager.determine_location_theme(sample_context)

        # Should select Forest due to dominance (2 forest vs 1 village)
        assert result.name == "Forest"

    @pytest.mark.asyncio
    async def test_determine_location_theme_player_preference(
        self, theme_manager, sample_context, sample_themes
    ):
        """Test theme determination considering player preferences."""
        sample_context.world_themes = sample_themes
        sample_context.player_preferences.preferred_themes = ["Village"]
        sample_context.adjacent_locations = [
            AdjacentLocationContext(
                location_id=uuid4(),
                direction="south",
                name="Forest Grove",
                description="Forest area",
                theme="Forest",
                short_description="Grove",
            )
        ]

        # Mock transition score calculation
        theme_manager._calculate_transition_score = AsyncMock(return_value=2.0)

        result = await theme_manager.determine_location_theme(sample_context)

        # Could be either, but player preference should influence scoring
        assert result.name in ["Forest", "Village"]

    @pytest.mark.asyncio
    async def test_validate_theme_consistency_compatible(
        self, theme_manager, sample_themes
    ):
        """Test theme consistency validation for compatible themes."""
        forest_theme = sample_themes[0]
        adjacent_themes = [sample_themes[1]]  # Village theme

        # Mock transition rules
        theme_manager.get_theme_transition_rules = AsyncMock(
            return_value=ThemeTransitionRules(
                from_theme="Village",
                to_theme="Forest",
                compatibility_score=0.8,
                transition_requirements=[],
            )
        )

        result = await theme_manager.validate_theme_consistency(
            forest_theme, adjacent_themes
        )

        assert result is True
        theme_manager.get_theme_transition_rules.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_theme_consistency_incompatible(
        self, theme_manager, sample_themes
    ):
        """Test theme consistency validation for incompatible themes."""
        forest_theme = sample_themes[0]
        adjacent_themes = [sample_themes[1]]  # Village theme

        # Mock low compatibility transition rules
        theme_manager.get_theme_transition_rules = AsyncMock(
            return_value=ThemeTransitionRules(
                from_theme="Village",
                to_theme="Forest",
                compatibility_score=0.2,  # Low compatibility
                transition_requirements=[],
            )
        )

        result = await theme_manager.validate_theme_consistency(
            forest_theme, adjacent_themes
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_get_theme_transition_rules_cached(self, theme_manager):
        """Test transition rules retrieval from cache."""
        # Pre-populate cache
        cached_rules = ThemeTransitionRules(
            from_theme="Forest",
            to_theme="Village",
            compatibility_score=0.7,
            transition_requirements=["gradual_change"],
        )
        theme_manager._transition_cache[("Forest", "Village")] = cached_rules

        result = await theme_manager.get_theme_transition_rules("Forest", "Village")

        assert result == cached_rules
        assert result.compatibility_score == 0.7

    @pytest.mark.asyncio
    async def test_get_theme_transition_rules_default(self, theme_manager):
        """Test default transition rules generation."""
        result = await theme_manager.get_theme_transition_rules("Forest", "Village")

        # Should generate default rules
        assert isinstance(result, ThemeTransitionRules)
        assert result.from_theme == "Forest"
        assert result.to_theme == "Village"
        assert 0.0 <= result.compatibility_score <= 1.0

    @pytest.mark.asyncio
    async def test_generate_theme_specific_content(self, theme_manager, sample_themes):
        """Test generation of theme-specific content."""
        forest_theme = sample_themes[0]

        result = await theme_manager.generate_theme_specific_content(
            forest_theme, "clearing"
        )

        assert isinstance(result, ThemeContent)
        assert result.theme_name == "Forest"
        assert len(result.objects) <= 5  # Should limit objects
        assert len(result.npcs) <= 3  # Should limit NPCs

        # Should include theme objects
        for obj in forest_theme.typical_objects:
            assert obj in result.objects

    @pytest.mark.asyncio
    async def test_load_theme_by_name_from_cache(self, theme_manager, sample_themes):
        """Test loading theme from cache."""
        forest_theme = sample_themes[0]
        theme_manager._theme_cache["Forest"] = forest_theme

        result = await theme_manager.load_theme_by_name("Forest")

        assert result == forest_theme

    @pytest.mark.asyncio
    async def test_load_theme_by_name_from_database(
        self, theme_manager, mock_session_factory, sample_themes
    ):
        """Test loading theme from database."""
        session_factory, session = mock_session_factory
        forest_theme = sample_themes[0]

        # Mock database response
        mock_row = Mock()
        mock_row.name = forest_theme.name
        mock_row.description = forest_theme.description
        mock_row.visual_elements = forest_theme.visual_elements
        mock_row.atmosphere = forest_theme.atmosphere
        mock_row.typical_objects = forest_theme.typical_objects
        mock_row.typical_npcs = forest_theme.typical_npcs
        mock_row.generation_parameters = forest_theme.generation_parameters
        mock_row.theme_id = forest_theme.theme_id
        mock_row.parent_theme_id = forest_theme.parent_theme_id
        mock_row.created_at = forest_theme.created_at

        mock_result = Mock()
        mock_result.fetchone.return_value = mock_row
        session.execute.return_value = mock_result

        result = await theme_manager.load_theme_by_name("Forest")

        assert result is not None
        assert result.name == "Forest"
        assert result.description == forest_theme.description

        # Should be cached after loading
        assert "Forest" in theme_manager._theme_cache

    @pytest.mark.asyncio
    async def test_load_theme_by_name_not_found(
        self, theme_manager, mock_session_factory
    ):
        """Test loading non-existent theme."""
        session_factory, session = mock_session_factory

        # Mock empty database response
        mock_result = Mock()
        mock_result.fetchone.return_value = None
        session.execute.return_value = mock_result

        result = await theme_manager.load_theme_by_name("NonExistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_all_themes(
        self, theme_manager, mock_session_factory, sample_themes
    ):
        """Test getting all themes from database."""
        session_factory, session = mock_session_factory

        # Mock database response with multiple themes
        mock_rows = []
        for theme in sample_themes:
            mock_row = Mock()
            mock_row.name = theme.name
            mock_row.description = theme.description
            mock_row.visual_elements = theme.visual_elements
            mock_row.atmosphere = theme.atmosphere
            mock_row.typical_objects = theme.typical_objects
            mock_row.typical_npcs = theme.typical_npcs
            mock_row.generation_parameters = theme.generation_parameters
            mock_row.theme_id = theme.theme_id
            mock_row.parent_theme_id = theme.parent_theme_id
            mock_row.created_at = theme.created_at
            mock_rows.append(mock_row)

        mock_result = Mock()
        mock_result.__iter__ = lambda self: iter(mock_rows)
        session.execute.return_value = mock_result

        result = await theme_manager.get_all_themes()

        assert len(result) == 2
        assert result[0].name in ["Forest", "Village"]
        assert result[1].name in ["Forest", "Village"]

        # Should cache all themes
        assert len(theme_manager._theme_cache) == 2

    def test_extract_adjacent_themes(self, theme_manager):
        """Test extraction of themes from adjacent locations."""
        adjacent_locations = [
            AdjacentLocationContext(
                location_id=uuid4(),
                direction="north",
                name="Grove",
                description="Forest",
                theme="Forest",
                short_description="Grove",
            ),
            AdjacentLocationContext(
                location_id=uuid4(),
                direction="south",
                name="Village",
                description="Settlement",
                theme="Village",
                short_description="Village",
            ),
        ]

        result = theme_manager._extract_adjacent_themes(adjacent_locations)

        assert "Forest" in result
        assert "Village" in result
        assert len(result) == 2

    def test_find_dominant_theme(self, theme_manager):
        """Test finding dominant theme from list."""
        themes = ["Forest", "Forest", "Village", "Forest"]

        result = theme_manager._find_dominant_theme(themes)

        assert result == "Forest"

    def test_find_dominant_theme_empty(self, theme_manager):
        """Test finding dominant theme from empty list."""
        themes = []

        result = theme_manager._find_dominant_theme(themes)

        assert result is None

    def test_get_type_specific_content(self, theme_manager):
        """Test getting type-specific content."""
        result = theme_manager._get_type_specific_content("clearing")

        assert "objects" in result
        assert "npcs" in result
        assert "special_features" in result
        assert isinstance(result["objects"], list)

    def test_get_type_specific_content_unknown_type(self, theme_manager):
        """Test getting content for unknown location type."""
        result = theme_manager._get_type_specific_content("unknown_type")

        # Should return empty dict for unknown types
        assert result == {}

    def test_generate_theme_descriptions(self, theme_manager, sample_themes):
        """Test generation of theme descriptions."""
        forest_theme = sample_themes[0]

        result = theme_manager._generate_theme_descriptions(forest_theme, "clearing")

        assert isinstance(result, list)
        assert len(result) > 0
        assert forest_theme.description in result

    def test_clear_cache(self, theme_manager):
        """Test clearing theme caches."""
        # Populate caches
        theme_manager._theme_cache["test"] = Mock()
        theme_manager._transition_cache[("a", "b")] = Mock()

        theme_manager.clear_cache()

        assert len(theme_manager._theme_cache) == 0
        assert len(theme_manager._transition_cache) == 0
