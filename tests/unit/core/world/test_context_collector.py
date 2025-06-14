"""
Unit tests for LocationContextCollector.
"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from game_loop.core.models.location_models import (
    AdjacentLocationContext,
    EnrichedContext,
    LocationGenerationContext,
    NarrativeContext,
    PlayerLocationPreferences,
)
from game_loop.core.models.navigation_models import ExpansionPoint
from game_loop.core.world.context_collector import LocationContextCollector
from game_loop.state.models import Location, NonPlayerCharacter, WorldObject, WorldState


@pytest.fixture
def sample_location():
    """Create a sample location."""
    # Create proper objects with UUIDs
    log_id = uuid4()
    mushroom_id = uuid4()
    hermit_id = uuid4()

    log = WorldObject(
        object_id=log_id,
        name="fallen log",
        description="A large fallen tree trunk",
        is_takeable=False,
    )

    mushroom = WorldObject(
        object_id=mushroom_id,
        name="mushroom",
        description="A colorful mushroom growing near the log",
        is_takeable=True,
    )

    hermit = NonPlayerCharacter(
        npc_id=hermit_id,
        name="forest hermit",
        description="A wise old hermit living in the woods",
        dialogue_state="friendly",
    )

    return Location(
        location_id=uuid4(),
        name="Forest Grove",
        description="A peaceful grove surrounded by tall trees and dappled sunlight.",
        connections={"north": uuid4(), "east": uuid4()},
        objects={log_id: log, mushroom_id: mushroom},
        npcs={hermit_id: hermit},
        state_flags={"visit_count": 5, "type": "clearing"},
    )


@pytest.fixture
def world_state_with_locations(sample_location):
    """Create a world state with sample locations."""
    world_state = WorldState()

    # Add the main location
    world_state.locations[sample_location.location_id] = sample_location

    # Add connected locations
    connected_ids = list(sample_location.connections.values())
    for i, location_id in enumerate(connected_ids):
        connected_location = Location(
            location_id=location_id,
            name=f"Connected Location {i+1}",
            description=f"Description for location {i+1}",
            connections={},
            objects={},
            npcs={},
            state_flags={"theme": "Forest" if i == 0 else "Village"},
        )
        world_state.locations[location_id] = connected_location

    return world_state


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
def context_collector(world_state_with_locations, mock_session_factory):
    """Create a LocationContextCollector instance."""
    session_factory, _ = mock_session_factory
    return LocationContextCollector(world_state_with_locations, session_factory)


@pytest.fixture
def sample_expansion_point(sample_location):
    """Create a sample expansion point."""
    return ExpansionPoint(
        location_id=sample_location.location_id,
        direction="west",
        priority=0.8,
        context={"location_name": sample_location.name},
    )


class TestLocationContextCollector:
    """Test cases for LocationContextCollector."""

    @pytest.mark.asyncio
    async def test_collect_expansion_context(
        self, context_collector, sample_expansion_point
    ):
        """Test collecting comprehensive context for location generation."""
        # Mock theme manager for getting world themes
        with patch.object(context_collector, "_get_world_themes", return_value=[]):
            result = await context_collector.collect_expansion_context(
                sample_expansion_point
            )

        assert isinstance(result, LocationGenerationContext)
        assert result.expansion_point == sample_expansion_point
        assert isinstance(result.adjacent_locations, list)
        assert isinstance(result.player_preferences, PlayerLocationPreferences)
        assert isinstance(result.world_themes, list)

    @pytest.mark.asyncio
    async def test_collect_expansion_context_cached(
        self, context_collector, sample_expansion_point
    ):
        """Test context collection with caching."""
        # First call
        with patch.object(context_collector, "_get_world_themes", return_value=[]):
            result1 = await context_collector.collect_expansion_context(
                sample_expansion_point
            )

        # Add to cache manually with recent timestamp
        cache_key = (
            f"{sample_expansion_point.location_id}_{sample_expansion_point.direction}"
        )
        result1._cached_at = datetime.now()
        context_collector._context_cache[cache_key] = result1

        # Second call should use cache
        result2 = await context_collector.collect_expansion_context(
            sample_expansion_point
        )

        assert result2 is result1  # Should be the exact same object from cache

    @pytest.mark.asyncio
    async def test_gather_adjacent_context(
        self, context_collector, sample_location, world_state_with_locations
    ):
        """Test gathering context from adjacent locations."""
        result = await context_collector.gather_adjacent_context(
            sample_location.location_id, "west"
        )

        assert isinstance(result, list)
        assert len(result) == 2  # Should have 2 connected locations

        for adj_context in result:
            assert isinstance(adj_context, AdjacentLocationContext)
            assert adj_context.name is not None
            assert adj_context.direction in ["north", "east"]
            assert adj_context.theme in ["Forest", "Village"]

    @pytest.mark.asyncio
    async def test_gather_adjacent_context_location_not_found(self, context_collector):
        """Test gathering context when location is not found."""
        non_existent_id = uuid4()

        result = await context_collector.gather_adjacent_context(
            non_existent_id, "north"
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_analyze_player_preferences_from_database(
        self, context_collector, mock_session_factory
    ):
        """Test analyzing player preferences from visit history."""
        session_factory, session = mock_session_factory
        player_id = uuid4()

        # Mock database response
        forest_grove = Mock()
        forest_grove.name = "Forest Grove"
        forest_grove.theme_name = "Forest"
        forest_grove.visit_count = 5
        forest_grove.last_visit = datetime.now()

        village_square = Mock()
        village_square.name = "Village Square"
        village_square.theme_name = "Village"
        village_square.visit_count = 3
        village_square.last_visit = datetime.now()

        mountain_peak = Mock()
        mountain_peak.name = "Mountain Peak"
        mountain_peak.theme_name = "Mountain"
        mountain_peak.visit_count = 1
        mountain_peak.last_visit = datetime.now()

        mock_rows = [forest_grove, village_square, mountain_peak]

        mock_result = Mock()
        mock_result.fetchall.return_value = mock_rows
        session.execute.return_value = mock_result

        result = await context_collector.analyze_player_preferences(player_id)

        assert isinstance(result, PlayerLocationPreferences)
        assert "Forest" in result.preferred_themes  # Most visited
        assert result.interaction_style in ["explorer", "balanced", "casual"]

        # Should be cached
        assert player_id in context_collector._preference_cache

    @pytest.mark.asyncio
    async def test_analyze_player_preferences_cached(self, context_collector):
        """Test player preference analysis with caching."""
        player_id = uuid4()
        cached_prefs = PlayerLocationPreferences(
            environments=["forest"],
            interaction_style="explorer",
            complexity_level="high",
        )

        # Pre-populate cache
        context_collector._preference_cache[player_id] = cached_prefs

        result = await context_collector.analyze_player_preferences(player_id)

        assert result is cached_prefs

    def test_analyze_visit_patterns(self, context_collector):
        """Test analysis of visit patterns to determine preferences."""
        # Create mock objects that properly simulate database rows
        forest_deep = Mock()
        forest_deep.name = "Deep Forest"
        forest_deep.theme_name = "Forest"
        forest_deep.visit_count = 8
        forest_deep.last_visit = datetime.now()

        forest_grove = Mock()
        forest_grove.name = "Forest Grove"
        forest_grove.theme_name = "Forest"
        forest_grove.visit_count = 5
        forest_grove.last_visit = datetime.now()

        village_market = Mock()
        village_market.name = "Village Market"
        village_market.theme_name = "Village"
        village_market.visit_count = 2
        village_market.last_visit = datetime.now()

        visit_data = [forest_deep, forest_grove, village_market]

        result = context_collector._analyze_visit_patterns(visit_data)

        assert isinstance(result, PlayerLocationPreferences)
        assert "Forest" in result.preferred_themes
        assert "forest" in result.environments
        assert result.interaction_style == "balanced"  # Average visits = 5.0

    def test_analyze_visit_patterns_empty(self, context_collector):
        """Test analysis with empty visit data."""
        result = context_collector._analyze_visit_patterns([])

        assert isinstance(result, PlayerLocationPreferences)
        assert result.complexity_level == "low"  # Default for empty data (< 5 items)

    def test_infer_theme_from_location(self, context_collector):
        """Test theme inference from location characteristics."""
        # Forest location
        forest_location = Mock()
        forest_location.name = "Deep Forest Grove"
        forest_location.description = "Surrounded by towering trees"

        result = context_collector._infer_theme_from_location(forest_location)
        assert result == "Forest"

        # Village location
        village_location = Mock()
        village_location.name = "Market Square"
        village_location.description = "Bustling village center"

        result = context_collector._infer_theme_from_location(village_location)
        assert result == "Village"

        # Unknown location should default to Forest
        unknown_location = Mock()
        unknown_location.name = "Mysterious Place"
        unknown_location.description = "An unusual location"

        result = context_collector._infer_theme_from_location(unknown_location)
        assert result == "Forest"

    def test_create_short_description(self, context_collector):
        """Test creation of short descriptions."""
        # Short description (no change needed)
        location = Mock()
        location.description = "A small grove"

        result = context_collector._create_short_description(location)
        assert result == "A small grove"

        # Long description (should be truncated)
        location.description = "A very long description that exceeds fifty characters and should be truncated with ellipsis"

        result = context_collector._create_short_description(location)
        assert len(result) <= 50
        assert result.endswith("...")

        # First sentence extraction
        location.description = "First sentence. Second sentence continues here with more text to exceed fifty characters."

        result = context_collector._create_short_description(location)
        assert result == "First sentence."

    @pytest.mark.asyncio
    async def test_extract_narrative_context(self, context_collector):
        """Test extraction of narrative context."""
        player_id = uuid4()
        location_area = "Forest Grove"

        result = await context_collector.extract_narrative_context(
            player_id, location_area
        )

        assert isinstance(result, NarrativeContext)
        assert isinstance(result.current_quests, list)
        assert isinstance(result.story_themes, list)
        assert result.narrative_tension in ["neutral", "low", "medium", "high"]

    @pytest.mark.asyncio
    async def test_enrich_context(self, context_collector, sample_expansion_point):
        """Test context enrichment with additional analysis."""
        # Create base context
        base_context = LocationGenerationContext(
            expansion_point=sample_expansion_point,
            adjacent_locations=[],
            player_preferences=PlayerLocationPreferences(
                environments=["forest"],
                interaction_style="balanced",
                complexity_level="medium",
            ),
            world_themes=[],
        )

        # Mock analysis methods
        context_collector._analyze_historical_patterns = AsyncMock(
            return_value={"pattern": "data"}
        )
        context_collector._analyze_player_behavior = AsyncMock(
            return_value={"behavior": "data"}
        )

        result = await context_collector.enrich_context(base_context)

        assert isinstance(result, EnrichedContext)
        assert result.base_context == base_context
        assert isinstance(result.generation_hints, list)
        assert isinstance(result.priority_elements, list)

    def test_generate_generation_hints(self, context_collector):
        """Test generation of hints for location generation."""
        # Create context with adjacent locations
        context = LocationGenerationContext(
            expansion_point=ExpansionPoint(
                location_id=uuid4(), direction="north", priority=0.8, context={}
            ),
            adjacent_locations=[
                AdjacentLocationContext(
                    location_id=uuid4(),
                    direction="south",
                    name="Forest Grove",
                    description="Grove",
                    theme="Forest",
                    short_description="Grove",
                )
            ],
            player_preferences=PlayerLocationPreferences(
                environments=["forest"],
                interaction_style="balanced",
                complexity_level="medium",
                preferred_themes=["Forest"],
            ),
            world_themes=[],
        )

        result = context_collector._generate_generation_hints(context)

        assert isinstance(result, list)
        assert len(result) > 0

        # Should include hints about theme consistency
        theme_hints = [h for h in result if "Forest" in h and "theme" in h]
        assert len(theme_hints) > 0

        # Should include direction hints
        direction_hints = [h for h in result if "north" in h]
        assert len(direction_hints) > 0

    def test_identify_priority_elements(self, context_collector):
        """Test identification of priority elements."""
        context = LocationGenerationContext(
            expansion_point=ExpansionPoint(
                location_id=uuid4(),
                direction="north",
                priority=0.9,  # High priority
                context={},
            ),
            adjacent_locations=[],
            player_preferences=PlayerLocationPreferences(
                environments=["forest"],
                interaction_style="balanced",
                complexity_level="high",  # High complexity
                social_preference="high",  # High social
            ),
            world_themes=[],
        )

        result = context_collector._identify_priority_elements(context)

        assert isinstance(result, list)
        assert "theme_consistency" in result  # Always prioritized
        assert "rich_interactions" in result  # High complexity
        assert "npc_presence" in result  # High social preference
        assert "high_quality_generation" in result  # High priority

    def test_get_generation_constraints(self, context_collector):
        """Test getting generation constraints."""
        result = context_collector._get_generation_constraints()

        assert isinstance(result, dict)
        assert "max_objects" in result
        assert "max_npcs" in result
        assert "min_description_length" in result
        assert result["max_objects"] == 5
        assert result["max_npcs"] == 3

    def test_get_world_rules(self, context_collector):
        """Test getting world consistency rules."""
        result = context_collector._get_world_rules()

        assert isinstance(result, list)
        assert len(result) > 0

        # Should contain basic world rules
        rule_text = " ".join(result)
        assert "logical" in rule_text.lower()
        assert "consistent" in rule_text.lower()

    def test_clear_cache(self, context_collector):
        """Test clearing context caches."""
        # Populate caches
        context_collector._preference_cache[uuid4()] = Mock()
        context_collector._context_cache["test_key"] = Mock()

        context_collector.clear_cache()

        assert len(context_collector._preference_cache) == 0
        assert len(context_collector._context_cache) == 0

    def test_get_default_preferences(self, context_collector):
        """Test getting default player preferences."""
        result = context_collector._get_default_preferences()

        assert isinstance(result, PlayerLocationPreferences)
        assert "forest" in result.environments
        assert result.interaction_style == "balanced"
        assert result.complexity_level == "medium"
