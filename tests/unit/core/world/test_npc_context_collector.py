"""
Unit tests for NPC Context Collector.
"""

from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from game_loop.core.models.location_models import LocationTheme
from game_loop.core.models.npc_models import NPCGenerationContext
from game_loop.core.world.npc_context_collector import NPCContextCollector
from game_loop.state.models import Location, NonPlayerCharacter, WorldState


@pytest.fixture
def world_state():
    """Create a world state with sample locations for testing."""
    world_state = WorldState()

    # Create main location
    main_location_id = uuid4()
    connected_location_id = uuid4()

    main_location = Location(
        location_id=main_location_id,
        name="Village Center",
        description="The heart of a bustling village",
        connections={"north": connected_location_id},
        objects={},
        npcs={},
        state_flags={"theme": "Village", "type": "village"},
    )

    connected_location = Location(
        location_id=connected_location_id,
        name="Village Outskirts",
        description="The edge of the village",
        connections={"south": main_location_id},
        objects={},
        npcs={},
        state_flags={"theme": "Village"},
    )

    world_state.locations[main_location_id] = main_location
    world_state.locations[connected_location_id] = connected_location

    return world_state, main_location_id, connected_location_id


@pytest.fixture
def mock_session_factory():
    """Create a mock session factory."""
    session_factory = Mock()
    session_factory.get_session = AsyncMock()
    return session_factory


@pytest.fixture
def context_collector(world_state, mock_session_factory):
    """Create a context collector for testing."""
    ws, _, _ = world_state
    return NPCContextCollector(ws, mock_session_factory)


class TestNPCContextCollector:
    """Test NPCContextCollector functionality."""

    def test_initialization(self, context_collector):
        """Test context collector initialization."""
        assert context_collector.world_state is not None
        assert context_collector.session_factory is not None

    @pytest.mark.asyncio
    async def test_collect_generation_context_basic(
        self, context_collector, world_state
    ):
        """Test basic context collection."""
        ws, main_location_id, _ = world_state

        context = await context_collector.collect_generation_context(
            main_location_id, "populate_location"
        )

        assert isinstance(context, NPCGenerationContext)
        assert context.location.name == "Village Center"
        assert context.location_theme.name == "Village"
        assert context.generation_purpose == "populate_location"
        assert context.player_level == 3  # Default value
        assert isinstance(context.constraints, dict)

    @pytest.mark.asyncio
    async def test_collect_generation_context_invalid_location(self, context_collector):
        """Test context collection with invalid location ID."""
        invalid_location_id = uuid4()

        with pytest.raises(ValueError, match="Location .* not found in world state"):
            await context_collector.collect_generation_context(
                invalid_location_id, "populate_location"
            )

    @pytest.mark.asyncio
    async def test_analyze_location_needs_village(self, context_collector, world_state):
        """Test location needs analysis for village."""
        ws, main_location_id, _ = world_state
        location = ws.locations[main_location_id]

        analysis = await context_collector.analyze_location_needs(location)

        assert "recommended_archetypes" in analysis
        assert "merchant" in analysis["recommended_archetypes"]
        assert "innkeeper" in analysis["recommended_archetypes"]
        assert "guard" in analysis["recommended_archetypes"]
        assert "artisan" in analysis["recommended_archetypes"]

        assert analysis["max_npcs"] == 4  # Village can have more NPCs
        assert "merchant" in analysis["priority_roles"]
        assert "innkeeper" in analysis["priority_roles"]
        assert analysis["can_add_npcs"] is True  # No existing NPCs

    @pytest.mark.asyncio
    async def test_analyze_location_needs_with_existing_npcs(
        self, context_collector, world_state
    ):
        """Test location needs analysis with existing NPCs."""
        ws, main_location_id, _ = world_state
        location = ws.locations[main_location_id]

        # Add some existing NPCs
        npc1 = NonPlayerCharacter(
            npc_id=uuid4(), name="Village Guard", description="A guard"
        )
        npc2 = NonPlayerCharacter(
            npc_id=uuid4(), name="Local Merchant", description="A trader"
        )

        location.npcs[npc1.npc_id] = npc1
        location.npcs[npc2.npc_id] = npc2

        analysis = await context_collector.analyze_location_needs(location)

        assert analysis["current_npc_count"] == 2
        assert analysis["can_add_npcs"] is True  # Still below max for village
        assert len(analysis["existing_archetypes"]) == 2
        assert len(analysis["missing_archetypes"]) > 0

    @pytest.mark.asyncio
    async def test_gather_social_context_empty_location(
        self, context_collector, world_state
    ):
        """Test social context gathering for empty location."""
        ws, main_location_id, _ = world_state

        social_context = await context_collector.gather_social_context(main_location_id)

        assert social_context["npc_relationships"] == {}
        assert social_context["social_hierarchy"] == []
        assert social_context["conflict_potential"] == 0.0
        assert social_context["cooperation_level"] == 0.5
        assert social_context["community_mood"] == "neutral"
        assert social_context["leadership"] is None

    @pytest.mark.asyncio
    async def test_gather_social_context_with_npcs(
        self, context_collector, world_state
    ):
        """Test social context gathering with NPCs."""
        ws, main_location_id, _ = world_state
        location = ws.locations[main_location_id]

        # Add NPCs
        guard = NonPlayerCharacter(
            npc_id=uuid4(), name="Village Guard", description="A guard"
        )
        merchant = NonPlayerCharacter(
            npc_id=uuid4(), name="Merchant Bob", description="A trader"
        )
        smith = NonPlayerCharacter(
            npc_id=uuid4(), name="Blacksmith", description="A smith"
        )

        location.npcs[guard.npc_id] = guard
        location.npcs[merchant.npc_id] = merchant
        location.npcs[smith.npc_id] = smith

        social_context = await context_collector.gather_social_context(main_location_id)

        assert social_context["cooperation_level"] == 0.7  # 3+ NPCs
        assert social_context["community_mood"] == "collaborative"
        assert len(social_context["social_hierarchy"]) == 3
        assert social_context["leadership"] == "Village Guard"  # Guard should be leader

    @pytest.mark.asyncio
    async def test_analyze_player_preferences(self, context_collector):
        """Test player preference analysis."""
        player_id = uuid4()

        preferences = await context_collector.analyze_player_preferences(player_id)

        # Should return default preferences since no history analysis is implemented
        assert "preferred_archetypes" in preferences
        assert "interaction_style" in preferences
        assert "complexity_preference" in preferences
        assert preferences["interaction_style"] == "friendly"
        assert preferences["complexity_preference"] == "moderate"

    @pytest.mark.asyncio
    async def test_collect_world_knowledge(self, context_collector, world_state):
        """Test world knowledge collection."""
        ws, main_location_id, _ = world_state
        location = ws.locations[main_location_id]

        world_knowledge = await context_collector.collect_world_knowledge(location)

        assert "current_events" in world_knowledge
        assert "historical_events" in world_knowledge
        assert "notable_locations" in world_knowledge
        assert "cultural_context" in world_knowledge

        # Should include connected locations
        assert len(world_knowledge["notable_locations"]) == 1
        connected_loc = world_knowledge["notable_locations"][0]
        assert connected_loc["name"] == "Village Outskirts"
        assert connected_loc["direction"] == "north"
        assert connected_loc["theme"] == "Village"

        # Should have cultural context
        cultural = world_knowledge["cultural_context"]
        assert cultural["primary_theme"] == "Village"
        assert (
            "communal" in cultural["regional_characteristics"]
        )  # Village characteristic

    @pytest.mark.asyncio
    async def test_determine_location_theme(self, context_collector, world_state):
        """Test location theme determination."""
        ws, main_location_id, _ = world_state
        location = ws.locations[main_location_id]

        theme = await context_collector._determine_location_theme(location)

        assert isinstance(theme, LocationTheme)
        assert theme.name == "Village"
        assert theme.description == "Village themed area"
        assert "village" in theme.visual_elements

    @pytest.mark.asyncio
    async def test_get_nearby_npcs(self, context_collector, world_state):
        """Test getting nearby NPCs."""
        ws, main_location_id, connected_location_id = world_state

        # Add NPCs to both locations
        main_npc = NonPlayerCharacter(
            npc_id=uuid4(), name="Main NPC", description="In main location"
        )
        connected_npc = NonPlayerCharacter(
            npc_id=uuid4(), name="Connected NPC", description="In connected location"
        )

        ws.locations[main_location_id].npcs[main_npc.npc_id] = main_npc
        ws.locations[connected_location_id].npcs[connected_npc.npc_id] = connected_npc

        nearby_npcs = await context_collector._get_nearby_npcs(main_location_id)

        assert len(nearby_npcs) == 2
        npc_names = [npc.name for npc in nearby_npcs]
        assert "Main NPC" in npc_names
        assert "Connected NPC" in npc_names

    @pytest.mark.asyncio
    async def test_create_world_snapshot(self, context_collector, world_state):
        """Test world snapshot creation."""
        ws, _, _ = world_state

        snapshot = await context_collector._create_world_snapshot()

        assert snapshot["total_locations"] == 2
        assert snapshot["total_npcs"] == 0  # No NPCs added yet
        assert "Village" in snapshot["themes_present"]
        assert snapshot["world_complexity"] == "developing"

    @pytest.mark.asyncio
    async def test_determine_player_level(self, context_collector):
        """Test player level determination."""
        level = await context_collector._determine_player_level()

        assert isinstance(level, int)
        assert level == 3  # Default implementation

    @pytest.mark.asyncio
    async def test_gather_generation_constraints_village(
        self, context_collector, world_state
    ):
        """Test generation constraints for village location."""
        ws, main_location_id, _ = world_state
        location = ws.locations[main_location_id]

        constraints = await context_collector._gather_generation_constraints(
            location, "populate_location", []
        )

        assert constraints["max_npcs_per_location"] == 4  # Village type
        assert constraints["avoid_duplicate_archetypes"] is True
        assert constraints["maintain_theme_consistency"] is True
        assert constraints["current_npc_count"] == 0
        assert constraints["can_add_npc"] is True

    @pytest.mark.asyncio
    async def test_gather_generation_constraints_quest_related(
        self, context_collector, world_state
    ):
        """Test generation constraints for quest-related purpose."""
        ws, main_location_id, _ = world_state
        location = ws.locations[main_location_id]

        constraints = await context_collector._gather_generation_constraints(
            location, "quest_related", []
        )

        assert constraints["ensure_quest_capability"] is True
        assert constraints["knowledge_requirements"] == "specialized"

    def test_get_regional_characteristics_village(self, context_collector):
        """Test getting regional characteristics for village theme."""
        characteristics = context_collector._get_regional_characteristics("Village")

        expected = ["communal", "agricultural", "traditional", "close-knit"]
        assert characteristics == expected

    def test_get_regional_characteristics_forest(self, context_collector):
        """Test getting regional characteristics for forest theme."""
        characteristics = context_collector._get_regional_characteristics("Forest")

        expected = ["natural", "peaceful", "wild", "ancient"]
        assert characteristics == expected

    def test_get_regional_characteristics_unknown(self, context_collector):
        """Test getting regional characteristics for unknown theme."""
        characteristics = context_collector._get_regional_characteristics(
            "UnknownTheme"
        )

        assert characteristics == ["neutral"]

    def test_get_common_knowledge_village(self, context_collector):
        """Test getting common knowledge for village theme."""
        knowledge = context_collector._get_common_knowledge("Village")

        expected = ["local_families", "crop_seasons", "market_days", "local_customs"]
        assert knowledge == expected

    def test_get_common_knowledge_forest(self, context_collector):
        """Test getting common knowledge for forest theme."""
        knowledge = context_collector._get_common_knowledge("Forest")

        expected = [
            "local_wildlife",
            "safe_paths",
            "seasonal_changes",
            "natural_resources",
        ]
        assert knowledge == expected

    def test_calculate_exploration_level(self, context_collector, world_state):
        """Test exploration level calculation."""
        ws, _, _ = world_state

        # With 2 locations, should return 0.2
        exploration_level = context_collector._calculate_exploration_level()
        assert exploration_level == 0.2

        # Add more locations to test different thresholds
        for i in range(5):  # Add 5 more locations (total 7)
            new_location = Location(
                location_id=uuid4(),
                name=f"Location {i}",
                description="Test location",
                connections={},
                objects={},
                npcs={},
                state_flags={},
            )
            ws.locations[new_location.location_id] = new_location

        exploration_level = context_collector._calculate_exploration_level()
        assert exploration_level == 0.6  # 7 locations should give 0.6 (6<7<=10)
